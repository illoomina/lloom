//! Request/window-owned reusable slots. Ring depth chooses how many slots to
//! prepare; a busy slot refuses instead of waiting or allocating a replacement.
#![allow(
    dead_code,
    reason = "private prepared sampler; native activation remains gated"
)]

use super::{
    guard::{disjoint, span, GuardedSample, SampleGuard},
    PrivateMultinomial,
};
use crate::backend::{BackendDevice, BackendStorage};
use crate::metal_backend::MetalDevice;
use crate::{DType, Layout, MetalError, MetalStorage, Result, Storage};
use std::cell::Cell;

pub enum Draw<'a> {
    Multinomial {
        uniforms: &'a MetalStorage,
        layout: &'a Layout,
    },
    Gumbel {
        variates: &'a MetalStorage,
        layout: &'a Layout,
    },
}

// Generated noise owns contiguous, zero-offset storage. Keep its geometry on
// the stack instead of rebuilding a shape/stride allocation for every draw.
enum NoiseInput<'a> {
    Supplied(Draw<'a>),
    Generated {
        storage: &'a MetalStorage,
        count: usize,
        gumbel: bool,
    },
}

pub struct PreparedSampler {
    kept: MetalStorage,
    selection_status: MetalStorage,
    weights: MetalStorage,
    result: PrivateMultinomial,
    capacity: usize,
    vocab: usize,
    last_use: Cell<Option<crate::retire::Epoch>>,
    poisoned: bool,
    noise: Option<PreparedNoise>,
}

// Geometry/scalars are immutable during the exclusive sampler borrow. Ambient
// custody is NOT certified across the user-supplied filler callback.
struct ValidatedRequest {
    vocab: usize,
    elements: usize,
    rows: usize,
    offset: usize,
    inv: f32,
    p: f64,
}

struct PreparedNoise {
    storage: MetalStorage,
    values: Vec<f32>,
}

impl PreparedSampler {
    /// Logical payload only; existing builder/admission owns physical rounding,
    /// metadata and namespace custody. Never a replacement allocator quote.
    pub fn payload_bytes(capacity: usize, vocab: usize) -> Result<usize> {
        if capacity == 0 || capacity > u32::MAX as usize || vocab == 0 || vocab > u32::MAX as usize
        {
            crate::bail!("prepared sampling: capacity/vocabulary");
        }
        capacity
            .checked_mul(vocab)
            .and_then(|n| n.checked_mul(12))
            .and_then(|n| capacity.checked_mul(36).and_then(|r| n.checked_add(r)))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("prepared sampling payload overflow".into()))
    }

    pub fn new(device: &MetalDevice, capacity: usize, vocab: usize) -> Result<Self> {
        Self::payload_bytes(capacity, vocab)?;
        // These slots survive their construction pass and every later draw.
        let _escape = crate::EscapeGuard::new();
        let elements = capacity * vocab;
        let allocate = |count: usize, dtype, label| {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(count, dtype)
                .with_label(label)
                .build()?;
            MetalStorage::new_allocated(buffer, device.clone(), count, dtype)
        };
        Ok(Self {
            kept: allocate(elements, DType::U32, "sampling_prepared_k")?,
            selection_status: allocate(capacity * 4, DType::U32, "sampling_prepared_k_status")?,
            weights: allocate(elements, DType::F32, "sampling_prepared_weights")?,
            result: PrivateMultinomial {
                ids: allocate(capacity, DType::U32, "sampling_prepared_ids")?,
                support: allocate(elements, DType::U32, "sampling_prepared_support")?,
                status: allocate(capacity * 4, DType::U32, "sampling_prepared_status")?,
                rows: capacity,
                vocab,
            },
            capacity,
            vocab,
            last_use: Cell::new(None),
            poisoned: false,
            noise: None,
        })
    }

    /// A noise upload shares the sampler slot's reuse epoch and exclusive
    /// borrow, so host writes never overlap an earlier device reader.
    pub fn with_noise(device: &MetalDevice, capacity: usize, vocab: usize) -> Result<Self> {
        let mut sampler = Self::new(device, capacity, vocab)?;
        let count = capacity * vocab;
        let slot = crate::metal_backend::HostSlot::new(device, count, DType::F32)?;
        let tensor = slot.tensor(count)?;
        let storage = {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                unreachable!("Metal upload")
            };
            storage.clone()
        };
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| crate::Error::Msg(format!("prepared sampling noise capacity: {e}")))?;
        values.resize(count, 0.);
        sampler.noise = Some(PreparedNoise { storage, values });
        Ok(sampler)
    }

    pub fn check_reuse(&self) -> Result<()> {
        if self.poisoned
            || self
                .last_use
                .get()
                .is_some_and(|e| !self.kept.device().gpu_clock().is_complete(e))
        {
            crate::bail!("prepared sampling: busy or poisoned slot");
        }
        let device = self.kept.device();
        device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        if !device.commands().executor().is_classical() {
            crate::bail!("prepared sampling: replay generation ownership not admitted");
        }
        Ok(())
    }

    fn check_request(
        &self,
        guard: &SampleGuard,
        scores: &MetalStorage,
        layout: &Layout,
        temperature: f64,
        p: f64,
    ) -> Result<ValidatedRequest> {
        self.check_reuse()?;
        let device = self.kept.device();
        let (vocab, elements, rows, offset, _) = scores.sampling_topk_geometry(layout)?;
        if vocab != self.vocab || rows > self.capacity || !scores.device().same_device(device) {
            crate::bail!("prepared sampling: capacity/device/vocabulary mismatch");
        }
        let inv = (1.0 / temperature) as f32;
        if !temperature.is_finite()
            || temperature <= 0.
            || !inv.is_finite()
            || !(0.0..=1.).contains(&p)
        {
            crate::bail!("prepared sampling: temperature or p");
        }
        self.check_custody(guard)?;
        Ok(ValidatedRequest {
            vocab,
            elements,
            rows,
            offset,
            inv,
            p,
        })
    }

    // Repeat only dynamic authority after arbitrary filler code. In particular
    // it may replace the executor, close an allocation context, or poison the
    // shared guard through a reentrant consumer. Nothing is uploaded first.
    fn check_custody(&self, guard: &SampleGuard) -> Result<()> {
        let device = self.kept.device();
        let origin = device.allocation_origin()?;
        for output in [
            &self.kept,
            &self.selection_status,
            &self.weights,
            &self.result.ids,
            &self.result.support,
            &self.result.status,
        ] {
            output
                .device()
                .declared_allocation_origin()
                .validate_reuse(&origin)?;
        }
        guard.check_device(device)
    }

    /// Generate noise directly into a prepared host scratch/upload pair. The
    /// filler retains its existing running or absolute-position RNG policy.
    /// Nothing here commits, waits, creates an RNG or allocates a replacement.
    #[allow(clippy::too_many_arguments, reason = "prepared draw and noise policy")]
    pub fn draw_generated<'a>(
        &'a mut self,
        guard: &'a SampleGuard,
        scores: &MetalStorage,
        layout: &Layout,
        temperature: f64,
        k: usize,
        p: f64,
        gumbel: bool,
        fill: impl FnOnce(&mut [f32], usize, usize) -> Result<()>,
    ) -> Result<GuardedSample<'a>> {
        let request = self.check_request(guard, scores, layout, temperature, p)?;
        let ValidatedRequest {
            vocab,
            elements,
            rows,
            ..
        } = request;
        let origin = self.kept.device().allocation_origin()?;
        let count = if gumbel { elements } else { rows };
        let noise = self
            .noise
            .as_mut()
            .ok_or_else(|| crate::Error::Msg("prepared sampling: no noise upload".into()))?;
        noise
            .storage
            .device()
            .declared_allocation_origin()
            .validate_reuse(&origin)?;
        fill(&mut noise.values[..count], rows, vocab)?;
        self.check_reuse()?;
        self.check_custody(guard)?;
        let noise = self.noise.as_mut().expect("validated prepared noise");
        noise
            .storage
            .device()
            .declared_allocation_origin()
            .validate_reuse(&self.kept.device().allocation_origin()?)?;
        let values = &noise.values[..count];
        // SAFETY: setup owns contiguous F32 Shared storage at offset zero;
        // the retained MetalStorage owns the same allocation as the cold tensor.
        // values is disjoint host scratch. check_reuse established completion
        // of previous reads; the returned carrier keeps the slot borrowed.
        if !unsafe { noise.storage.write_host_slice(0, values) } {
            crate::bail!("prepared sampling: mapped noise upload unavailable");
        }
        let input = noise.storage.clone();
        self.draw_inner(
            guard,
            scores,
            layout,
            request,
            k,
            NoiseInput::Generated {
                storage: &input,
                count,
                gumbel,
            },
        )
    }

    /// Reuses prepared sampling storage without a readback or wait. Encoder and
    /// dispatch bookkeeping retain their existing costs. The carrier extends
    /// the slot's last-use epoch through later consumers and snapshot copies.
    #[allow(
        clippy::too_many_arguments,
        reason = "private prepared draw parameters"
    )]
    pub fn draw<'a>(
        &'a mut self,
        guard: &'a SampleGuard,
        scores: &MetalStorage,
        layout: &Layout,
        temperature: f64,
        k: usize,
        p: f64,
        draw: Draw<'_>,
    ) -> Result<GuardedSample<'a>> {
        let request = self.check_request(guard, scores, layout, temperature, p)?;
        self.draw_inner(
            guard,
            scores,
            layout,
            request,
            k,
            NoiseInput::Supplied(draw),
        )
    }

    #[allow(clippy::too_many_arguments, reason = "shared prepared draw validation")]
    fn draw_inner<'a>(
        &'a mut self,
        guard: &'a SampleGuard,
        scores: &MetalStorage,
        layout: &Layout,
        request: ValidatedRequest,
        k: usize,
        noise: NoiseInput<'_>,
    ) -> Result<GuardedSample<'a>> {
        let ValidatedRequest {
            vocab,
            elements,
            rows,
            offset,
            inv,
            p,
        } = request;
        let device = self.kept.device();
        let (input, input_offset, count, gumbel) = match noise {
            NoiseInput::Supplied(draw) => {
                let (input, input_layout, gumbel) = match draw {
                    Draw::Multinomial { uniforms, layout } => (uniforms, layout, false),
                    Draw::Gumbel { variates, layout } => (variates, layout, true),
                };
                let count = input_layout
                    .dims()
                    .iter()
                    .rev()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| crate::Error::Msg("prepared sampling input overflow".into()))?;
                if !input_layout.is_contiguous() || (gumbel && input_layout.dims() != layout.dims())
                {
                    crate::bail!("prepared sampling: variate layout/device");
                }
                (input, input_layout.start_offset(), count, gumbel)
            }
            NoiseInput::Generated {
                storage,
                count,
                gumbel,
            } => (storage, 0, count, gumbel),
        };
        if count != if gumbel { elements } else { rows } || !input.device().same_device(device) {
            crate::bail!("prepared sampling: variate layout/device");
        }
        let input_range = span(input, DType::F32, input_offset, count)?;
        let score_range = span(scores, DType::F32, layout.start_offset(), elements)?;
        for output in [
            &self.kept,
            &self.selection_status,
            &self.weights,
            &self.result.ids,
            &self.result.support,
            &self.result.status,
        ] {
            let range = span(output, output.dtype, 0, output.count)?;
            disjoint(output, range, input, input_range)?;
            disjoint(output, range, scores, score_range)?;
        }
        let entry = guard.native_entry();
        self.poisoned = true;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            super::guard::require_classical_encoder(&encoder)?;
            self.last_use.set(Some(device.gpu_clock().current_epoch()));
            entry.note_epoch(device.gpu_clock().current_epoch());
            // SAFETY: preflight above checked all spans/scalars/devices; setup
            // owns disjoint scratch/results and reuse requires last completion.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_topk_mask(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (scores.buffer(), offset),
                    rows as u32,
                    vocab as u32,
                    k.min(vocab) as u32,
                    self.kept.buffer(),
                    self.selection_status.buffer(),
                )
            }
            .map_err(MetalError::from)?;
        }
        if rows != 0 {
            let encoder = device.command_encoder()?;
            super::guard::require_classical_encoder(&encoder)?;
            self.last_use.set(Some(device.gpu_clock().current_epoch()));
            entry.note_epoch(device.gpu_clock().current_epoch());
            let call = if gumbel {
                crate::kernels::kernels::sampling_topk::call_sampling_filtered_gumbel_f32
            } else {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial
            };
            // SAFETY: same complete preflight as K, unchanged selector outputs;
            // input offsets include their physical view base exactly once.
            unsafe {
                call(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (scores.buffer(), offset),
                    rows as u32,
                    vocab as u32,
                    inv,
                    p as f32,
                    u32::from(p != 1.0),
                    (input.buffer(), input_offset * 4),
                    self.kept.buffer(),
                    self.selection_status.buffer(),
                    self.weights.buffer(),
                    self.result.support.buffer(),
                    self.result.ids.buffer(),
                    self.result.status.buffer(),
                )
            }
            .map_err(MetalError::from)?;
        }
        let mut sample = self.result.clone();
        sample.rows = rows;
        let mut token = guard.bind(sample)?;
        self.last_use.set(Some(device.gpu_clock().current_epoch()));
        token.workspace_use = Some(&self.last_use);
        entry.finish();
        self.poisoned = false;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    fn device() -> Result<Device> {
        crate::isolation::claim("prepared sampler request custody");
        let mut tuning = crate::tuning::Tuning::default();
        tuning.from_env().expect("valid test environment");
        crate::tuning::install(tuning).expect("before device");
        let d = Device::new_metal(0)?;
        eprintln!(
            "PREPARED_CUSTODY_SETUP actual_api={}",
            d.as_metal_device()?.commands().actual_command_api()
        );
        Ok(d)
    }

    fn scores(d: &Device) -> Result<MetalStorage> {
        let t = Tensor::new(&[0f32; 3], d)?;
        let (s, _) = t.storage_and_layout();
        let Storage::Metal(s) = &*s else {
            unreachable!()
        };
        Ok(s.clone())
    }

    #[test]
    fn filler_executor_change_refuses_before_shared_upload() -> Result<()> {
        use crate::kernels::metal::executor::{DispatchRecord, Executor, ExecutorSlot};
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Count(Arc<AtomicUsize>);
        impl Executor for Count {
            fn dispatch(&self, _: &DispatchRecord) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed);
                true
            }
        }
        let d = device()?;
        let m = d.as_metal_device()?;
        let z = scores(&d)?;
        let guard = SampleGuard::new(m)?;
        let mut sampler = PreparedSampler::with_noise(m, 1, 3)?;
        m.wait_until_completed()?;
        // Exclusive, never submitted upload; the callback must not overwrite it.
        assert!(unsafe {
            sampler
                .noise
                .as_ref()
                .unwrap()
                .storage
                .write_host_slice(0, &[0.125f32; 3])
        });
        let count = Arc::new(AtomicUsize::new(0));
        let prior = m.commands().executor();
        let result = sampler
            .draw_generated(
                &guard,
                &z,
                &Layout::contiguous(3),
                1.,
                0,
                1.,
                false,
                |out, _, _| {
                    out[0] = 0.75;
                    m.set_executor(Arc::new(ExecutorSlot::Custom(Arc::new(Count(
                        count.clone(),
                    )))));
                    Ok(())
                },
            )
            .map(drop);
        m.set_executor(prior);
        m.wait_until_completed()?;
        let err = result.unwrap_err();
        assert!(err.to_string().contains("replay generation"), "{err}");
        assert_eq!(count.load(Ordering::Relaxed), 0);
        assert_eq!(
            sampler
                .noise
                .as_ref()
                .unwrap()
                .storage
                .peek_scalar::<f32>(0),
            Some(0.125)
        );
        assert!(sampler.last_use.get().is_none());
        Ok(())
    }

    #[test]
    fn filler_cancellation_refuses_before_shared_upload() -> Result<()> {
        use crate::metal_backend::MetalAllocationDomain;
        let base = device()?;
        let domain = crate::memory_admission::ModelDomain::isolated_numeric_fixture(64 << 20);
        let namespace = MetalAllocationDomain::new(base.as_metal_device()?, domain)?;
        let mut promise = Some(namespace.domain().reserve(16 << 20)?);
        let scope = namespace.scope(promise.as_ref().unwrap().context())?;
        scope.run(|m| {
            let d = Device::Metal(m.clone());
            let z = scores(&d)?;
            let guard = SampleGuard::new(m)?;
            let mut sampler = PreparedSampler::with_noise(m, 1, 3)?;
            m.wait_until_completed()?;
            // Setup is drained and no draw has acquired the upload.
            assert!(unsafe {
                sampler
                    .noise
                    .as_ref()
                    .unwrap()
                    .storage
                    .write_host_slice(0, &[0.125f32; 3])
            });
            let result = sampler
                .draw_generated(
                    &guard,
                    &z,
                    &Layout::contiguous(3),
                    1.,
                    0,
                    1.,
                    false,
                    |out, _, _| {
                        out[0] = 0.75;
                        drop(promise.take());
                        Ok(())
                    },
                )
                .map(drop);
            assert!(result.is_err(), "closed allocation context survived filler");
            assert_eq!(
                sampler
                    .noise
                    .as_ref()
                    .unwrap()
                    .storage
                    .peek_scalar::<f32>(0),
                Some(0.125)
            );
            assert!(sampler.last_use.get().is_none());
            Ok(())
        })
    }

    #[test]
    fn persistent_sampler_construction_during_served_pass_declares_all_six_escapes() -> Result<()> {
        use crate::kernels::metal::{ArenaLayout, PassKey, PassKind};
        let d = device()?;
        let m = d.as_metal_device()?;
        let x = Tensor::new(&[1f32, 2., 3.], &d)?;
        m.wait_until_completed()?;
        m.enable_pass_plans(ArenaLayout::Packed);
        let key = PassKey::new(PassKind::Decode { b: 1 });
        for turn in 0..5 {
            m.begin_pass(key.clone())?;
            let body = (|| -> Result<(PreparedSampler, Tensor)> {
                let before = m.arena_counters();
                let sampler = PreparedSampler::with_noise(m, 1, 3)?;
                if turn == 4 {
                    let before = before.expect("warmed pass must own an arena");
                    let after = m.arena_counters().expect("arena remains installed");
                    assert_eq!(
                        after.hits, before.hits,
                        "persistent slot served from ordinal"
                    );
                    assert!(
                        after.declined_escaped - before.declined_escaped >= 6,
                        "all six private slots must escape: {before:?} -> {after:?}"
                    );
                }
                let y = x.affine(2., 1.)?;
                Ok((sampler, y))
            })();
            let (mut sampler, y) = match body {
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
            // Use the retained constructor result after its pass closes.
            let guard = SampleGuard::new(m)?;
            let mut snapshot = super::super::readback::SampleReadback::new(m, 1)?;
            let z = scores(&d)?;
            let token = sampler.draw_generated(
                &guard,
                &z,
                &Layout::contiguous(3),
                1.,
                0,
                1.,
                false,
                |out, _, _| {
                    out[0] = 0.5;
                    Ok(())
                },
            )?;
            token.snapshot(&mut snapshot)?;
            drop(token);
            assert_eq!(snapshot.read_committed()?.expect("own snapshot").ids(), [1]);
            m.wait_until_completed()?;
        }
        Ok(())
    }
}
