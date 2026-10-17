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
        // The selector remains the single score-layout validator and K stage.
        // Preflight row/uniform agreement before it allocates or encodes.
        let vocab = *layout.dims().last().ok_or_else(|| {
            crate::Error::Msg("sampling_multinomial: no vocabulary dimension".into())
        })?;
        let elements = layout.dims().iter().rev().try_fold(1usize, |n, &d| {
            n.checked_mul(d).ok_or_else(|| {
                crate::Error::Msg("sampling_multinomial: score count overflow".into())
            })
        })?;
        if vocab == 0 || uniform_count != elements / vocab {
            crate::bail!("sampling_multinomial: invalid row/uniform geometry");
        }
        let (kept, selection_status) = self.sampling_topk_mask(layout, k)?;
        let rows = uniform_count;
        let device = self.device();
        let allocate = |count: usize, dtype, label| {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(count.max(1), dtype)
                .with_label(label)
                .build()?;
            Self::new_allocated(buffer, device.clone(), count, dtype)
        };
        let weights = allocate(elements, DType::F32, "sampling_weights")?;
        let support = allocate(elements, DType::U32, "sampling_support")?;
        let ids = allocate(rows, DType::U32, "sampling_private_ids")?;
        let status = allocate(bytes(rows)?, DType::U32, "sampling_private_status")?;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            // SAFETY: the selector validated the full score span/geometry;
            // uniform spans and device match were checked before encoding.
            // Fresh outputs are disjoint, storage/pool retains reuse authority
            // through completion; the input K/status stay live across the bind.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (self.buffer(), bytes(layout.start_offset())?),
                    rows as u32,
                    vocab as u32,
                    inv_t,
                    p as f32,
                    u32::from(p != 1.0),
                    (uniforms.buffer(), bytes(uniform_start)?),
                    kept.buffer(),
                    selection_status.buffer(),
                    weights.buffer(),
                    support.buffer(),
                    ids.buffer(),
                    status.buffer(),
                )
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
        let (kept, selection_status) = self.sampling_topk_mask(layout, k)?;
        let device = self.device();
        let allocate = |count: usize, dtype, label| {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(count.max(1), dtype)
                .with_label(label)
                .build()?;
            Self::new_allocated(buffer, device.clone(), count, dtype)
        };
        let weights = allocate(elements, DType::F32, "sampling_weights")?;
        let support = allocate(elements, DType::U32, "sampling_support")?;
        let ids = allocate(rows, DType::U32, "sampling_private_ids")?;
        let status = allocate(status_elements, DType::U32, "sampling_private_status")?;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            // SAFETY: complete spans/devices/geometry checked before any
            // allocation; K/status are these unchanged rows' selector output.
            // Fresh disjoint outputs and inputs retain existing pool custody
            // through completion/capture/replay; no public ID is constructed.
            unsafe {
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
                    kept.buffer(),
                    selection_status.buffer(),
                    weights.buffer(),
                    support.buffer(),
                    ids.buffer(),
                    status.buffer(),
                )
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

mod result;
mod readback;
mod guard;
mod prepared;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gumbel_tests;
