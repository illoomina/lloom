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
use crate::{DType, Layout, MetalError, MetalStorage, Result};
use std::cell::Cell;

pub(super) enum Draw<'a> {
    Multinomial {
        uniforms: &'a MetalStorage,
        layout: &'a Layout,
    },
    Gumbel {
        variates: &'a MetalStorage,
        layout: &'a Layout,
    },
}

pub(super) struct PreparedSampler {
    kept: MetalStorage,
    selection_status: MetalStorage,
    weights: MetalStorage,
    result: PrivateMultinomial,
    capacity: usize,
    vocab: usize,
    last_use: Cell<Option<crate::retire::Epoch>>,
    poisoned: bool,
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
        })
    }

    /// No readback/wait/allocation on successful reuse. The carrier extends the
    /// slot's last-use epoch through every later consumer and snapshot copy.
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
        let device = self.kept.device();
        if self.poisoned
            || self
                .last_use
                .get()
                .is_some_and(|e| !device.gpu_clock().is_complete(e))
        {
            crate::bail!("prepared sampling: busy or poisoned slot");
        }
        device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        if !device.commands().executor().is_classical() {
            crate::bail!("prepared sampling: replay generation ownership not admitted");
        }
        let (vocab, elements, rows, offset, _) = scores.sampling_topk_geometry(layout)?;
        if vocab != self.vocab || rows > self.capacity || !scores.device().same_device(device) {
            crate::bail!("prepared sampling: capacity/device/vocabulary mismatch");
        }
        let inv = (1.0 / temperature) as f32;
        if !temperature.is_finite()
            || temperature <= 0.0
            || !inv.is_finite()
            || !(0.0..=1.0).contains(&p)
        {
            crate::bail!("prepared sampling: temperature or p");
        }
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
        if !input_layout.is_contiguous()
            || count != if gumbel { elements } else { rows }
            || (gumbel && input_layout.dims() != layout.dims())
            || !input.device().same_device(device)
        {
            crate::bail!("prepared sampling: variate layout/device");
        }
        let input_range = span(input, DType::F32, input_layout.start_offset(), count)?;
        let score_range = span(scores, DType::F32, layout.start_offset(), elements)?;
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
            let range = span(output, output.dtype, 0, output.count)?;
            disjoint(output, range, input, input_range)?;
            disjoint(output, range, scores, score_range)?;
        }
        // Bind verifies the request guard's device before any draw side effect.
        guard.check_device(device)?;
        self.poisoned = true;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            self.last_use.set(Some(device.gpu_clock().current_epoch()));
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
            self.last_use.set(Some(device.gpu_clock().current_epoch()));
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
                    (input.buffer(), input_layout.start_offset() * 4),
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
        self.poisoned = false;
        Ok(token)
    }
}
