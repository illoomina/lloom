use super::{MetalError, MetalStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{DType, Layout, Result};

/// Private diagnostics, deliberately not convertible to a sampled Tensor.
/// A status/sentinel here is not an asynchronous consumer-error protocol.
#[allow(dead_code, reason = "private source candidate, no public integration")]
#[derive(Clone)]
pub(crate) struct PrivateMultinomial {
    ids: MetalStorage,
    support: MetalStorage,
    status: MetalStorage,
    rows: usize,
    vocab: usize,
}

impl MetalStorage {
    /// Processed-f32 contract only: no processors, probability callback, RNG,
    /// greedy threshold or public routing. Uniforms are supplied per row by
    /// the caller; their absolute-position mapping is not redefined here.
    #[allow(dead_code, reason = "private source candidate, no public integration")]
    pub(crate) fn sampling_multinomial(
        &self,
        layout: &Layout,
        temperature: f64,
        k: usize,
        p: f64,
        uniforms: &Self,
        uniform_layout: &Layout,
    ) -> Result<PrivateMultinomial> {
        if !temperature.is_finite() || temperature <= 0.0 {
            crate::bail!("sampling_multinomial: temperature must be finite and positive");
        }
        let inv_t = (1.0 / temperature) as f32;
        if !inv_t.is_finite() {
            crate::bail!("sampling_multinomial: temperature reciprocal must fit finite f32");
        }
        if !(0.0..=1.0).contains(&p) {
            crate::bail!("sampling_multinomial: p must be in [0, 1]");
        }
        if uniforms.dtype != DType::F32 {
            crate::bail!("sampling_multinomial: uniforms must be f32");
        }
        if !self.device().same_device(uniforms.device()) {
            crate::bail!("sampling_multinomial: scores and uniforms must share a device");
        }
        let uniform_count = uniform_layout
            .dims()
            .iter()
            .rev()
            .try_fold(1usize, |n, &d| {
                n.checked_mul(d).ok_or_else(|| {
                    crate::Error::Msg("sampling_multinomial: uniform count overflow".into())
                })
            })?;
        // Layout's contiguity predicate multiplies dimensions in reverse.
        // Validate those products before invoking its unchecked arithmetic.
        if !uniform_layout.is_contiguous() {
            crate::bail!("sampling_multinomial: uniforms must be contiguous");
        }
        let bytes = |n: usize| {
            n.checked_mul(4)
                .ok_or_else(|| crate::Error::Msg("sampling_multinomial: byte span overflow".into()))
        };
        let uniform_start = uniform_layout.start_offset();
        let uniform_end = uniform_start.checked_add(uniform_count).ok_or_else(|| {
            crate::Error::Msg("sampling_multinomial: uniform span overflow".into())
        })?;
        let uniform_end_bytes = bytes(uniform_end)?;
        if uniform_end > uniforms.count
            || uniform_end_bytes > uniforms.buffer().length()
            || !uniforms.buffer().base_offset().is_multiple_of(4)
            || uniforms
                .buffer()
                .base_offset()
                .checked_add(uniform_end_bytes)
                .is_none()
        {
            crate::bail!("sampling_multinomial: uniform range exceeds storage");
        }
        // The selector's geometry check is the single score-layout validator.
        // Preflight row/uniform agreement before anything allocates or encodes.
        let (vocab, elements, rows, score_offset, status_elements) =
            self.sampling_topk_geometry(layout)?;
        if uniform_count != rows {
            crate::bail!("sampling_multinomial: invalid row/uniform geometry");
        }
        let device = self.device();
        let allocate = |count: usize, dtype, label| {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(count.max(1), dtype)
                .with_label(label)
                .build()?;
            Self::new_allocated(buffer, device.clone(), count, dtype)
        };
        let words = crate::kernels::kernels::sampling_topk::sampling_work_words(rows, vocab)
            .ok_or_else(|| crate::Error::Msg("sampling_multinomial: scratch overflow".into()))?;
        let work = allocate(words, DType::U32, "sampling_work")?;
        let scratch_ids = allocate(elements, DType::U32, "sampling_scratch_ids")?;
        let selection_status = allocate(status_elements, DType::U32, "sampling_topk_status")?;
        let weights = allocate(elements, DType::F32, "sampling_weights")?;
        let support = allocate(elements, DType::U32, "sampling_support")?;
        let ids = allocate(rows, DType::U32, "sampling_private_ids")?;
        let status = allocate(bytes(rows)?, DType::U32, "sampling_private_status")?;
        let uniform_offset = bytes(uniform_start)?;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            // SAFETY: the geometry check validated the full score span;
            // uniform spans and device match were checked before encoding.
            // Fresh outputs and scratch are disjoint, storage/pool retains
            // reuse authority through completion.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_select(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (self.buffer(), score_offset),
                    rows as u32,
                    vocab as u32,
                    k.min(vocab) as u32,
                    u32::from(p != 1.0),
                    work.buffer(),
                    scratch_ids.buffer(),
                    selection_status.buffer(),
                )
                .and_then(|()| {
                    crate::kernels::kernels::sampling_topk::call_sampling_multinomial(
                        device.metal_device(),
                        &encoder,
                        device.kernels(),
                        (self.buffer(), score_offset),
                        rows as u32,
                        vocab as u32,
                        inv_t,
                        p as f32,
                        u32::from(p != 1.0),
                        (uniforms.buffer(), uniform_offset),
                        work.buffer(),
                        scratch_ids.buffer(),
                        weights.buffer(),
                        support.buffer(),
                        ids.buffer(),
                        status.buffer(),
                    )
                })
            }
            .map_err(MetalError::from)?;
        }
        Ok(PrivateMultinomial {
            ids,
            support,
            status,
            rows,
            vocab,
        })
    }

    /// Private f32 draw candidate only. Gumbel variates must already exist as
    /// contiguous f32 rows in original-id order, using the existing caller's
    /// stream_key(seed, absolute_position) and full-vocabulary consumption.
    /// No RNG/noise preparation or public checked-token bridge is supplied.
    /// The centered separately-rounded f32 expression is shared with host
    /// generation, not an identity promise with the former f64 expression.
    #[allow(
        dead_code,
        reason = "private arithmetic candidate, no public integration"
    )]
    pub(crate) fn sampling_filtered_gumbel_f32(
        &self,
        layout: &Layout,
        temperature: f64,
        k: usize,
        p: f64,
        variates: &Self,
        variate_layout: &Layout,
    ) -> Result<PrivateMultinomial> {
        if !temperature.is_finite() || temperature <= 0.0 {
            crate::bail!("sampling_filtered_gumbel_f32: temperature must be finite and positive");
        }
        let inv_t = (1.0 / temperature) as f32;
        if !inv_t.is_finite() || !(0.0..=1.0).contains(&p) {
            crate::bail!("sampling_filtered_gumbel_f32: invalid reciprocal temperature or p");
        }
        // Reuse the selector's complete no-allocation score-layout preflight.
        let (vocab, elements, rows, score_offset, status_elements) =
            self.sampling_topk_geometry(layout)?;
        if variates.dtype != DType::F32 || !self.device().same_device(variates.device()) {
            crate::bail!("sampling_filtered_gumbel_f32: variates need f32 on the score device");
        }
        let count = variate_layout
            .dims()
            .iter()
            .rev()
            .try_fold(1usize, |n, &d| {
                n.checked_mul(d).ok_or_else(|| {
                    crate::Error::Msg("sampling_filtered_gumbel_f32: variate count overflow".into())
                })
            })?;
        if variate_layout.dims() != layout.dims()
            || count != elements
            || !variate_layout.is_contiguous()
        {
            crate::bail!("sampling_filtered_gumbel_f32: variates must match contiguous score rows");
        }
        let bytes = |n: usize| {
            n.checked_mul(4).ok_or_else(|| {
                crate::Error::Msg("sampling_filtered_gumbel_f32: byte span overflow".into())
            })
        };
        let start = variate_layout.start_offset();
        let end = start.checked_add(count).ok_or_else(|| {
            crate::Error::Msg("sampling_filtered_gumbel_f32: variate span overflow".into())
        })?;
        let end_bytes = bytes(end)?;
        let variate_offset = bytes(start)?;
        if end > variates.count
            || end_bytes > variates.buffer().length()
            || !variates.buffer().base_offset().is_multiple_of(4)
            || variates
                .buffer()
                .base_offset()
                .checked_add(end_bytes)
                .is_none()
        {
            crate::bail!("sampling_filtered_gumbel_f32: invalid physical variate span/alignment");
        }
        // All scalar/layout/device/address products are checked before the K
        // stage can allocate/encode; its status size check covers outputs too.
        let device = self.device();
        let allocate = |count: usize, dtype, label| {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(count.max(1), dtype)
                .with_label(label)
                .build()?;
            Self::new_allocated(buffer, device.clone(), count, dtype)
        };
        let words = crate::kernels::kernels::sampling_topk::sampling_work_words(rows, vocab)
            .ok_or_else(|| {
                crate::Error::Msg("sampling_filtered_gumbel_f32: scratch overflow".into())
            })?;
        let work = allocate(words, DType::U32, "sampling_work")?;
        let scratch_ids = allocate(elements, DType::U32, "sampling_scratch_ids")?;
        let selection_status = allocate(status_elements, DType::U32, "sampling_topk_status")?;
        let weights = allocate(elements, DType::F32, "sampling_weights")?;
        let support = allocate(elements, DType::U32, "sampling_support")?;
        let ids = allocate(rows, DType::U32, "sampling_private_ids")?;
        let status = allocate(status_elements, DType::U32, "sampling_private_status")?;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            // SAFETY: complete spans/devices/geometry checked before any
            // allocation. Fresh disjoint outputs, scratch and inputs retain
            // existing pool custody through completion/capture/replay; no
            // public ID is constructed.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_select(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (self.buffer(), score_offset),
                    rows as u32,
                    vocab as u32,
                    k.min(vocab) as u32,
                    u32::from(p != 1.0),
                    work.buffer(),
                    scratch_ids.buffer(),
                    selection_status.buffer(),
                )
                .and_then(|()| {
                    crate::kernels::kernels::sampling_topk::call_sampling_filtered_gumbel_f32(
                        device.metal_device(),
                        &encoder,
                        device.kernels(),
                        (self.buffer(), score_offset),
                        rows as u32,
                        vocab as u32,
                        inv_t,
                        p as f32,
                        u32::from(p != 1.0),
                        (variates.buffer(), variate_offset),
                        work.buffer(),
                        scratch_ids.buffer(),
                        weights.buffer(),
                        support.buffer(),
                        ids.buffer(),
                        status.buffer(),
                    )
                })
            }
            .map_err(MetalError::from)?;
        }
        Ok(PrivateMultinomial {
            ids,
            support,
            status,
            rows,
            vocab,
        })
    }
}

pub(super) mod guard;
mod host_scratch;
pub(super) mod prepared;
pub(super) mod prepared_slots;
pub(super) mod readback;
mod result;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gumbel_tests;

#[cfg(test)]
mod timing_tests;
