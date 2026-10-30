//! Explicit prepared native sampler. This is not a default or a routing policy.

use super::{noise, LogitsProcessor, Sampling};
use lloom_tensor::metal_backend::sampling::{
    GuardedSample, PreparedSampler, PreparedSamplerSlots, SampleGuard,
};
use lloom_tensor::{DType, Device, Result, Storage, Tensor};

/// A request owns its sticky error and a bounded number of reusable uploads.
/// The returned token borrows its slot through the last consumer encoding.
pub struct PreparedNativeSampler {
    guard: SampleGuard,
    slots: PreparedSamplerSlots,
    next: usize,
    rows: usize,
    vocab: usize,
    device: Device,
}

/// The existing device forcing rule, with request-prepared guarded state.
pub struct PreparedNativeBudget {
    forcing: super::budget::DeviceForcing,
    vocab: usize,
}

impl PreparedNativeBudget {
    pub fn new(budget: super::ThinkingBudget, device: &Device, vocab: usize) -> Result<Self> {
        device.as_metal_device()?;
        budget.check_native_vocab(vocab)?;
        let mut forcing = super::budget::DeviceForcing::new(budget, device)?;
        forcing.prepare_guarded()?;
        Ok(Self { forcing, vocab })
    }

    pub(crate) fn check_for(&self, device: &Device, vocab: usize) -> Result<()> {
        if self.vocab != vocab {
            lloom_tensor::bail!("native thinking budget: prepared vocabulary mismatch");
        }
        self.forcing.check_guarded_device(device)
    }

    pub(crate) fn live(&self) -> bool {
        self.forcing.live()
    }

    pub(crate) fn feed(&mut self, token: &mut GuardedSample<'_>) -> Result<()> {
        self.forcing.feed_guarded(token)
    }
}

impl PreparedNativeSampler {
    /// Logical device payload and host noise scratch, respectively. Physical
    /// rounding, native metadata, snapshots and model outputs remain the
    /// existing caller's quote, not a quota created by this adapter.
    pub fn payload_bytes(rows: usize, vocab: usize, slots: usize) -> Result<(usize, usize)> {
        let sampler = PreparedSampler::payload_bytes(rows, vocab)?;
        let noise = rows.checked_mul(vocab).and_then(|n| n.checked_mul(4));
        let device = noise
            .and_then(|n| sampler.checked_add(n))
            .and_then(|n| n.checked_mul(slots))
            .and_then(|n| n.checked_add(16));
        let host = noise.and_then(|n| n.checked_mul(slots));
        match (device, host) {
            (Some(device), Some(host))
                if slots > 0 && device <= isize::MAX as usize && host <= isize::MAX as usize =>
            {
                Ok((device, host))
            }
            _ => lloom_tensor::bail!("native sampling: prepared payload overflows or has no slots"),
        }
    }

    pub fn new(device: &Device, rows: usize, vocab: usize, slots: usize) -> Result<Self> {
        Self::payload_bytes(rows, vocab, slots)?;
        let metal = device.as_metal_device()?;
        metal
            .commands()
            .check_failed()
            .map_err(lloom_tensor::MetalError::from)?;
        if !metal.commands().executor().is_classical() {
            lloom_tensor::bail!("native sampling: captured generation ownership is not admitted");
        }
        let prepared = PreparedSamplerSlots::with_noise(metal, rows, vocab, slots)?;
        Ok(Self {
            guard: SampleGuard::new(metal)?,
            slots: prepared,
            next: 0,
            rows,
            vocab,
            device: device.clone(),
        })
    }

    fn check_rows(&self, logits: &Tensor, rows: usize) -> Result<()> {
        if rows == 0
            || rows > self.rows
            || logits.dim(lloom_tensor::D::Minus1)? != self.vocab
            || logits.elem_count().checked_div(self.vocab) != Some(rows)
            || !logits.device().same_device(&self.device)
        {
            lloom_tensor::bail!("native sampling: prepared row/vocabulary/device mismatch");
        }
        Ok(())
    }

    pub(crate) fn check_output(&self, output: &Tensor, vocab: usize) -> Result<()> {
        let (_, layout) = output.storage_and_layout();
        if !layout.is_contiguous()
            || self.vocab != vocab
            || self.rows != 1
            || !matches!(output.dims(), [1, 1, _])
            || !self.device.same_device(output.device())
        {
            lloom_tensor::bail!("native ring: one prepared row on the output device required");
        }
        Ok(())
    }

    pub(crate) fn check_sampling_kind(&self, sampling: &Sampling) -> Result<()> {
        let Some((temperature, _, p)) = sampling.distribution() else {
            lloom_tensor::bail!("native sampling: argmax retains its existing route");
        };
        super::distribution::validate(temperature, p)
    }

    fn check_sampling(&self, sampler: &LogitsProcessor) -> Result<()> {
        let Some((temperature, _, p)) = sampler.sampling.distribution() else {
            lloom_tensor::bail!("native sampling: argmax retains its existing route");
        };
        super::distribution::validate(temperature, p)?;
        self.slots[self.next].check_reuse()
    }

    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Retain once on the producer thread for deferred/final publication.
    pub fn publisher(&self) -> lloom_tensor::metal_backend::sampling::SamplePublisher {
        self.guard.publisher()
    }

    /// Poll at the driver's existing completion cadence, even when a failed
    /// sample cannot publish a ring marker. No commit or wait is introduced.
    pub fn check_error(&mut self) -> Result<bool> {
        self.guard.check_error()
    }

    /// Scalar draw, forcing, then processor-history update. The returned
    /// carrier holds both mutable workspace and budget borrows through its
    /// last consumer encoding; queued reads use normal device epoch custody.
    ///
    /// ```compile_fail
    /// use lloom::generation::native::{PreparedNativeBudget, PreparedNativeSampler};
    /// use lloom::generation::LogitsProcessor;
    /// use lloom_tensor::{Result, Tensor};
    /// fn cannot_refeed(a: &mut PreparedNativeSampler, b: &mut PreparedNativeSampler,
    ///     budget: &mut PreparedNativeBudget, p: &mut LogitsProcessor, x: &Tensor,
    ///     out: &Tensor) -> Result<()> {
    ///     let first = a.sample(p, x, Some(budget))?;
    ///     let second = b.sample(p, x, Some(budget))?;
    ///     first.copy_into_tensor(out, 0)?;
    ///     drop(second);
    ///     Ok(())
    /// }
    /// ```
    pub fn sample<'a>(
        &'a mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
        budget: Option<&'a mut PreparedNativeBudget>,
    ) -> Result<GuardedSample<'a>> {
        if sampler.digests() {
            lloom_tensor::bail!(
                "native scalar sampling: logits digest needs the explicit host route"
            );
        }
        if let Some(budget) = budget.as_ref() {
            budget.check_for(&self.device, self.vocab)?;
        }
        let mut token = self.sample_for_host_readback(sampler, logits)?;
        if let Some(budget) = budget {
            budget.feed(&mut token)?;
        }
        if let Some(stage) = sampler.stage.as_mut() {
            stage.record_guarded(&token)?;
        }
        Ok(token)
    }

    /// Shared scalar draw primitive, before fed-token history publication.
    /// sample() applies guarded device forcing/history immediately for Ring.
    /// NativeBlocking instead snapshots, waits at its existing ID boundary,
    /// validates, applies host forcing, then records the actual fed id once.
    /// A digest is permitted here only because the checked host strategy
    /// captures raw logits afterward; sample() retains its digest refusal.
    pub(crate) fn sample_for_host_readback<'a>(
        &'a mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<GuardedSample<'a>> {
        self.check_rows(logits, 1)?;
        self.check_sampling(sampler)?;
        let temperature = sampler.temperature();
        let processed = match sampler.stage.as_mut() {
            Some(stage) => stage.apply(logits, temperature)?,
            None => logits.clone(),
        };
        let key = sampler
            .position
            .map_or(noise::Key::Running, noise::Key::Position);
        self.draw(sampler, &processed, key)
    }

    /// A window does not advance processor history. Only commit_window's
    /// accepted prefix may do that; rejected rows consume no position key.
    pub fn window<'a>(
        &'a mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
        first: u64,
        inputs: Option<&Tensor>,
    ) -> Result<GuardedSample<'a>> {
        let rows = match logits.dims() {
            [1, _, _] => logits.squeeze(0)?,
            [_, _] => logits.clone(),
            _ => lloom_tensor::bail!("native sampling: window requires [1,t,vocab] or [t,vocab]"),
        };
        let count = rows.dim(0)?;
        self.check_rows(&rows, count)?;
        self.check_sampling(sampler)?;
        if u64::try_from(count - 1)
            .ok()
            .and_then(|n| first.checked_add(n))
            .is_none()
        {
            lloom_tensor::bail!("native sampling: absolute position overflow");
        }
        let temperature = sampler.temperature();
        let processed = match (sampler.stage.as_mut(), inputs) {
            (None, _) => rows.clone(),
            (Some(stage), Some(inputs)) => stage.apply_rows(&rows, inputs, temperature)?,
            (Some(_), None) => {
                lloom_tensor::bail!("native sampling: processed window needs its input ids")
            }
        };
        let token = self.draw(sampler, &processed, noise::Key::Window(first))?;
        sampler.window_logits = sampler.digest.is_some().then_some(rows);
        Ok(token)
    }

    fn draw<'a>(
        &'a mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
        key: noise::Key,
    ) -> Result<GuardedSample<'a>> {
        let Some((temperature, k, p)) = sampler.sampling.distribution() else {
            lloom_tensor::bail!("native sampling: argmax retains its existing route");
        };
        let method = if matches!(
            sampler.sampling,
            Sampling::GumbelSoftmax { .. } | Sampling::FilteredGumbel { .. }
        ) {
            noise::Method::Gumbel
        } else {
            noise::Method::Multinomial
        };
        let scores = logits.to_dtype(DType::F32)?.contiguous()?;
        let (storage, layout) = scores.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            unreachable!("device checked before processing")
        };
        let slot = self.next;
        self.next = (slot + 1) % self.slots.len();
        let seed = sampler.seed;
        self.slots[slot].draw_generated(
            &self.guard,
            storage,
            layout,
            temperature,
            k,
            p,
            method == noise::Method::Gumbel,
            |out, rows, vocab| noise::fill(out, rows, vocab, method, key, seed, &mut sampler.rng),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_quote_includes_every_noise_upload_and_host_scratch() -> Result<()> {
        // 12RV+36R sampler, 4RV upload and 4RV host noise; one 16-byte guard.
        assert_eq!(PreparedNativeSampler::payload_bytes(2, 3, 4)?, (688, 96));
        for (rows, vocab, slots) in [
            (0, 3, 1),
            (1, 0, 1),
            (1, 3, 0),
            (usize::MAX, 3, 1),
            (1, 3, usize::MAX),
        ] {
            assert!(PreparedNativeSampler::payload_bytes(rows, vocab, slots).is_err());
        }
        Ok(())
    }
}

/// One request's prepared verify-window transfers and folded embedding output.
/// This is separate from LogitsProcessor, whose Send behavior is unchanged.
/// Reuse requires the previous window's existing readback/completion boundary;
/// dropping the owner never resets its queued request's sticky guard.
pub struct PreparedNativeWindow {
    sampler: PreparedNativeSampler,
    readback: lloom_tensor::metal_backend::sampling::SampleReadback,
    ids: Tensor,
    embeds: Tensor,
    capacity: usize,
    poisoned: bool,
}

/// Internal-only envelope for the shared verify/fold consumer. Device consumers
/// use `guarded()`; ids may only join the existing window readback, whose whole
/// result remains enclosed until `with_window` checks status/error completion.
pub(crate) struct NativeWindowTokens<'a> {
    token: &'a GuardedSample<'a>,
    ids: Tensor,
    embeds: Tensor,
}

impl NativeWindowTokens<'_> {
    pub(crate) fn guarded(&self) -> &GuardedSample<'_> {
        self.token
    }
    pub(crate) fn ids_for_readback(&self) -> &Tensor {
        &self.ids
    }
    pub(crate) fn embedding_output(&self) -> &Tensor {
        &self.embeds
    }
}

impl PreparedNativeWindow {
    /// Caller-prepared model output uses the table's working dtype (packed
    /// get_rows writes f32); no full-table dequantization or hidden quota.
    pub fn new(
        device: &Device,
        capacity: usize,
        vocab: usize,
        slots: usize,
        embeddings: Tensor,
    ) -> Result<Self> {
        let (_, rows, _) = embeddings.dims3()?;
        if embeddings.dim(0)? != 1 || rows < capacity || !device.same_device(embeddings.device()) {
            lloom_tensor::bail!("native window: prepared embedding capacity/device mismatch");
        }
        let sampler = PreparedNativeSampler::new(device, capacity, vocab, slots)?;
        let readback = lloom_tensor::metal_backend::sampling::SampleReadback::new(
            device.as_metal_device()?,
            capacity,
        )?;
        let ids = Tensor::zeros(capacity, DType::U32, device)?;
        Ok(Self {
            sampler,
            readback,
            ids,
            embeds: embeddings,
            capacity,
            poisoned: false,
        })
    }

    /// Additional logical payload beside PreparedNativeSampler's quote.
    /// Existing model/output allocator and #985 admission still own physical
    /// rounding, metadata, namespace and future-emission custody.
    pub fn transfer_payload_bytes(capacity: usize) -> Result<usize> {
        capacity
            .checked_mul(6)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("native window transfer payload overflow".into())
            })
    }

    /// Request/window geometry only; readback-only targets need no sampled
    /// embedding consumer. The owning model separately checks any fold table.
    pub(crate) fn check_geometry(&self, device: &Device, vocab: usize, rows: usize) -> Result<()> {
        if self.poisoned || vocab != self.sampler.vocab || !self.sampler.device.same_device(device)
        {
            lloom_tensor::bail!(
                "native window: poisoned request or model/vocabulary/device mismatch"
            );
        }
        self.check_rows(rows)
    }

    pub(crate) fn check_layout(
        &self,
        quote: &lloom_tensor::quantized::GuardedEmbeddingLayout,
        rows: usize,
    ) -> Result<()> {
        self.check_geometry(&quote.device, quote.vocab, rows)?;
        let (b, capacity, hidden) = self.embeds.dims3()?;
        if b != 1
            || capacity < rows
            || hidden != quote.hidden
            || self.embeds.dtype() != quote.dtype
            || !self.embeds.device().same_device(&quote.device)
        {
            lloom_tensor::bail!("native window: prepared output differs from cold model layout");
        }
        Ok(())
    }

    pub(crate) fn check_rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            lloom_tensor::bail!("native window: row capacity exceeded before encoding");
        }
        Ok(())
    }

    pub(crate) fn check_sampling(&self, processor: &LogitsProcessor) -> Result<()> {
        self.sampler.check_sampling_kind(processor.sampling())?;
        if let Some(forcing) = &processor.window_forcing {
            forcing.check_vocab(self.sampler.vocab)?;
        }
        Ok(())
    }

    pub(crate) fn check_model(
        &self,
        device: &Device,
        vocab: usize,
        table: &lloom_tensor::quantized::QMatMul,
    ) -> Result<()> {
        if self.poisoned || vocab != self.sampler.vocab || !self.sampler.device.same_device(device)
        {
            lloom_tensor::bail!(
                "native window: poisoned request or model/vocabulary/device mismatch"
            );
        }
        table.check_guarded_embedding_output(&self.embeds)
    }

    pub(crate) fn with_window<T>(
        &mut self,
        processor: &mut LogitsProcessor,
        logits: &Tensor,
        first: u64,
        inputs: Option<&Tensor>,
        consume: impl FnOnce(NativeWindowTokens<'_>) -> Result<(Vec<u32>, T)>,
    ) -> Result<(Vec<u32>, T)> {
        if self.poisoned {
            lloom_tensor::bail!("native window: prior consumer failed");
        }
        let rows = match logits.dims() {
            [1, rows, _] | [rows, _] => *rows,
            _ => lloom_tensor::bail!("native window: expected single-sequence score rows"),
        };
        if rows == 0 || rows > self.capacity {
            lloom_tensor::bail!("native window: row capacity exceeded before encoding");
        }
        self.sampler.check_error()?;
        let ids = self.ids.narrow(0, 0, rows)?;
        let embeds = self.embeds.narrow(1, 0, rows)?;
        let token = self.sampler.window(processor, logits, first, inputs)?;
        self.poisoned = true;
        token.copy_into_tensor(&ids, 0)?;
        token.snapshot(&mut self.readback)?;
        let result = consume(NativeWindowTokens {
            token: &token,
            ids,
            embeds,
        })?;
        drop(token);
        // The shared consumer performed its EXISTING one-window wait. Reading
        // status is additional validation, never permission to index ids early.
        let checked = self.readback.poll()?.ok_or_else(|| {
            lloom_tensor::Error::Msg(
                "native window: consumer returned before its existing completion boundary".into(),
            )
        })?;
        if checked.ids().len() != rows || checked.ids() != result.0 {
            lloom_tensor::bail!("native window: completed rows differ from window-read ids");
        }
        self.sampler.check_error()?;
        self.poisoned = false;
        Ok(result)
    }
}

#[cfg(test)]
mod window_controls {
    use super::*;
    use lloom_tensor::quantized::QMatMul;
    fn device() -> Result<Device> {
        lloom_tensor::isolation::claim("prepared native window consumer controls");
        Device::new_metal(0)
    }
    fn processor() -> LogitsProcessor {
        LogitsProcessor::from_sampling(
            17,
            Sampling::TopK {
                temperature: 1.,
                k: 1,
            },
        )
    }

    #[test]
    fn scalar_host_digest_refusal_preserves_rng_history_and_prepared_slot() -> Result<()> {
        use rand::RngCore;
        let device = device()?;
        let mut native = PreparedNativeSampler::new(&device, 1, 3, 2)?;
        let mut processor = processor()
            .with_processors(super::super::Processors {
                presence_penalty: 10.,
                ..Default::default()
            })
            .with_logits_digest(true);
        let logits = Tensor::new(&[0f32, 3., 0.], &device)?;
        let mut before_rng = processor.rng.clone();
        assert!(native.sample(&mut processor, &logits, None).is_err());
        let mut after_rng = processor.rng.clone();
        assert_eq!(before_rng.next_u64(), after_rng.next_u64());
        assert_eq!(processor.digest_rows(), 0);
        assert_eq!(native.next, 0);
        processor = processor.with_logits_digest(false);
        let token = native.sample(&mut processor, &logits, None)?;
        let mut snapshot = lloom_tensor::metal_backend::sampling::SampleReadback::new(
            device.as_metal_device()?,
            1,
        )?;
        token.snapshot(&mut snapshot)?;
        drop(token);
        device.synchronize()?;
        assert_eq!(
            snapshot.poll()?.expect("completed").ids(),
            [1],
            "refusal did not record the high-presence-penalty token into history"
        );
        Ok(())
    }

    #[test]
    fn invalid_later_window_row_blocks_fold_embedding_and_all_returned_results() -> Result<()> {
        let device = device()?;
        let output = Tensor::full(-99f32, (1, 2, 3), &device)?;
        let table = QMatMul::Tensor(Tensor::eye(3, DType::F32, &device)?);
        let mut window = PreparedNativeWindow::new(&device, 2, 3, 2, output.clone())?;
        let bad = Tensor::new(&[[0f32, 3., 0.], [f32::NAN, 0., 0.]], &device)?;
        let mut processor = processor();
        let failed = window.with_window(&mut processor, &bad, 4, None, |tokens| {
            table.embedding_guarded_into(tokens.guarded(), tokens.embedding_output())?;
            Ok((tokens.ids_for_readback().to_vec1::<u32>()?, ()))
        });
        assert!(
            failed.is_err(),
            "no id vector/FoldRun may escape invalid later status"
        );
        device.synchronize()?;
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![-99.; 6]);
        let valid = Tensor::new(&[[0f32, 3., 0.], [0., 0., 3.]], &device)?;
        assert!(window
            .with_window(
                &mut processor,
                &valid,
                4,
                None,
                |_| -> Result<(Vec<u32>, ())> {
                    panic!("failed request cannot encode a fresh consumer")
                }
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn window_capacity_and_forced_vocabulary_are_checked_before_consumers() -> Result<()> {
        let device = device()?;
        let mut window = PreparedNativeWindow::new(
            &device,
            2,
            3,
            2,
            Tensor::zeros((1, 2, 3), DType::F32, &device)?,
        )?;
        assert!(window.check_rows(3).is_err());
        // A valid budget whose forced (closing) id 3 is outside vocabulary 3.
        let processor = processor().with_window_budget(Some(&super::super::ThinkingBudget::new(
            0,
            vec![3],
            vec![3],
        )?));
        assert!(window.check_sampling(&processor).is_err());
        let valid = Tensor::new(&[[0f32, 3., 0.]], &device)?;
        let mut processor = self::processor();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(Vec<u32>, ())> =
                window.with_window(&mut processor, &valid, 1, None, |_| {
                    panic!("consumer unwind")
                });
        }));
        assert!(unwind.is_err());
        assert!(window
            .with_window(
                &mut processor,
                &valid,
                1,
                None,
                |_| -> Result<(Vec<u32>, ())> { panic!("unwound request must remain poisoned") }
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn window_and_forcing_constructed_in_served_pass_keep_their_storage_afterward() -> Result<()> {
        use lloom_tensor::kernels::metal::{ArenaLayout, PassKey, PassKind};
        let d = device()?;
        let m = d.as_metal_device()?;
        eprintln!(
            "NATIVE_PERSISTENT_SETUP actual_api={}",
            m.commands().actual_command_api()
        );
        let embeds = Tensor::zeros((1, 1, 3), DType::F32, &d)?;
        let x = Tensor::new(&[1f32, 2., 3.], &d)?;
        d.synchronize()?;
        m.enable_pass_plans(ArenaLayout::Packed);
        let key = PassKey::new(PassKind::Decode { b: 1 });
        let mut owners = Vec::new();
        for _ in 0..5 {
            m.begin_pass(key)?;
            let body = (|| -> Result<_> {
                let window = PreparedNativeWindow::new(&d, 1, 3, 1, embeds.clone())?;
                let forcing = PreparedNativeBudget::new(
                    super::super::ThinkingBudget::new(0, vec![2], vec![2])?,
                    &d,
                    3,
                )?;
                let y = x.affine(2., 1.)?;
                Ok((window, forcing, y))
            })();
            let (window, forcing, y) = match body {
                Ok(v) => {
                    m.finish_pass()?;
                    v
                }
                Err(e) => {
                    m.abort_pass()?;
                    return Err(e);
                }
            };
            assert_eq!(y.to_vec1::<f32>()?, [3., 5., 7.]);
            owners.push((window, forcing));
        }
        assert!(
            m.pass_plan_counters().expect("enabled plans").served > 0,
            "constructor lifetime fixture never served a cached pass"
        );
        // All five owners remain alive across later served passes. Verify each
        // retained ids buffer, then actually use its native draw and forcing.
        for (mut window, mut forcing) in owners {
            assert_eq!(window.ids.to_vec1::<u32>()?, [0]);
            let logits = Tensor::new(&[0f32, 3., 0.], &d)?;
            let mut processor = processor();
            let token = window
                .sampler
                .sample(&mut processor, &logits, Some(&mut forcing))?;
            token.copy_into_tensor(&window.ids, 0)?;
            token.snapshot(&mut window.readback)?;
            drop(token);
            assert_eq!(
                window
                    .readback
                    .read_committed()?
                    .expect("own snapshot")
                    .ids(),
                [2]
            );
            assert_eq!(window.ids.to_vec1::<u32>()?, [2]);
            d.synchronize()?;
        }
        Ok(())
    }
}
