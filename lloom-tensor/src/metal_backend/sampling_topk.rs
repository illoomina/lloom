use super::{MetalError, MetalStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, Result};

impl MetalStorage {
    // The sampler's other inputs must be checked before the selector allocates.
    pub(super) fn sampling_topk_geometry(
        &self,
        layout: &Layout,
    ) -> Result<(usize, usize, usize, usize, usize)> {
        if self.dtype != DType::F32 {
            crate::bail!("sampling_topk_mask requires processed f32 scores");
        }
        let vocab = *layout.dims().last().ok_or_else(|| {
            crate::Error::Msg("sampling_topk_mask requires a vocabulary dimension".into())
        })?;
        if vocab == 0 || vocab > u32::MAX as usize {
            crate::bail!("sampling_topk_mask: vocabulary must be in 1..=u32::MAX");
        }
        let elements = layout.dims().iter().rev().try_fold(1usize, |n, &d| {
            n.checked_mul(d).ok_or_else(|| {
                crate::Error::Msg("sampling_topk_mask: element count overflow".into())
            })
        })?;
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| crate::Error::Msg("sampling_topk_mask: input span overflow".into()))?;
        if end > self.count {
            crate::bail!("sampling_topk_mask: input range exceeds its storage");
        }
        if !layout.is_contiguous() {
            crate::bail!("sampling_topk_mask requires contiguous rows");
        }
        let rows = elements / vocab;
        if rows > u32::MAX as usize || elements % vocab != 0 {
            crate::bail!("sampling_topk_mask: invalid row geometry");
        }
        let bytes = |n: usize| {
            n.checked_mul(4)
                .ok_or_else(|| crate::Error::Msg("sampling_topk_mask: byte count overflow".into()))
        };
        let offset = bytes(start)?;
        let status_elements = bytes(rows)?;
        bytes(elements)?;
        let end_bytes = bytes(end)?;
        if end_bytes > self.buffer().length()
            || !self.buffer().base_offset().is_multiple_of(4)
            || self.buffer().base_offset().checked_add(end_bytes).is_none()
        {
            crate::bail!("sampling_topk_mask: invalid physical input span or alignment");
        }
        bytes(status_elements)?;
        Ok((vocab, elements, rows, offset, status_elements))
    }

    /// Private selection-only arm: a dense u32 mask plus four diagnostic words
    /// per row, with no readback or mutation of the processed f32 input.
    /// This is not a sampler and its status is not the ring's error protocol.
    #[allow(
        dead_code,
        reason = "private candidate is intentionally unwired until native acceptance and policy review"
    )]
    pub(crate) fn sampling_topk_mask(&self, layout: &Layout, k: usize) -> Result<(Self, Self)> {
        let (vocab, elements, rows, offset, status_elements) =
            self.sampling_topk_geometry(layout)?;
        let device = self.device();
        let mask = device
            .new_buffer_builder()
            .with_size_for(elements.max(1), DType::U32)
            .with_label("sampling_topk_mask")
            .build()?;
        let status = device
            .new_buffer_builder()
            .with_size_for(status_elements.max(1), DType::U32)
            .with_label("sampling_topk_status")
            .build()?;
        if rows != 0 {
            let encoder = device.command_encoder()?;
            // SAFETY: validated storage span; fresh, disjoint outputs; the
            // storage/pool owns reuse until GPU completion (module binding authority).
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_topk_mask(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    (self.buffer(), offset),
                    rows as u32,
                    vocab as u32,
                    k.min(vocab) as u32,
                    &mask,
                    &status,
                )
            }
            .map_err(MetalError::from)?;
        }
        Ok((
            Self::new_allocated(mask, device.clone(), elements, DType::U32)?,
            Self::new_allocated(status, device.clone(), status_elements, DType::U32)?,
        ))
    }
}

#[cfg(test)]
mod tests;
