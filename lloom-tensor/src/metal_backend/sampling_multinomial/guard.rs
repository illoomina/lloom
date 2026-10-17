//! Private guarded consumers; no raw sampled-id Tensor escape. Prepared outputs
//! remain owned by their existing request/window allocator, never this adapter.
#![allow(
    dead_code,
    reason = "private consumer candidate; native admission precedes public wiring"
)]

use super::PrivateMultinomial;
use crate::backend::{BackendDevice, BackendStorage};
use crate::kernels::kernels::sampling_guard::{self, Consumer};
use crate::metal_backend::{HostSlot, MetalDevice};
use crate::{DType, MetalError, MetalStorage, Result, Storage};
use std::cell::Cell;

/// Request-exclusive sticky failure, initialized once before encoding. It is
/// never reset while queued consumers can still read it. Not Sync by design.
pub(super) struct SampleGuard {
    slot: HostSlot,
    error: MetalStorage,
    last_use: Cell<Option<crate::retire::Epoch>>,
    poisoned: Cell<bool>,
    words: Vec<u32>,
}

/// The only device-id consumer surface. The sample and error owner remain live
/// until the carrier is consumed; buffer custody still follows the queue epoch.
pub(super) struct GuardedSample<'a> {
    pub(super) sample: PrivateMultinomial,
    pub(super) guard: &'a SampleGuard,
    pub(super) workspace_use: Option<&'a Cell<Option<crate::retire::Epoch>>>,
}

pub(super) fn span(
    s: &MetalStorage,
    dtype: DType,
    start: usize,
    count: usize,
) -> Result<(usize, usize)> {
    let end = start.checked_add(count);
    let end = end
        .filter(|&e| e <= s.count)
        .ok_or_else(|| crate::Error::Msg("guarded consumer: storage span".into()))?;
    let size = dtype.size_in_bytes();
    let lo = start
        .checked_mul(size)
        .and_then(|n| s.buffer().base_offset().checked_add(n));
    let hi = end
        .checked_mul(size)
        .filter(|&n| n <= s.buffer().length())
        .and_then(|n| s.buffer().base_offset().checked_add(n));
    if s.dtype != dtype
        || !s.buffer().base_offset().is_multiple_of(size)
        || lo.is_none()
        || hi.is_none()
    {
        crate::bail!("guarded consumer: dtype, physical span or alignment");
    }
    Ok((lo.expect("checked"), hi.expect("checked")))
}

pub(super) fn disjoint(
    a: &MetalStorage,
    ar: (usize, usize),
    b: &MetalStorage,
    br: (usize, usize),
) -> Result<()> {
    if a.buffer().raw_ptr() == b.buffer().raw_ptr() && ar.0 < br.1 && br.0 < ar.1 {
        crate::bail!("guarded consumer: aliased input/output");
    }
    Ok(())
}

impl SampleGuard {
    pub(super) fn check_device(&self, device: &MetalDevice) -> Result<()> {
        if self.poisoned.get() || !self.error.device().same_device(device) {
            crate::bail!("guarded consumer: poisoned guard or device mismatch");
        }
        self.error
            .device()
            .declared_allocation_origin()
            .validate_reuse(&device.allocation_origin()?)?;
        Ok(())
    }
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let slot = HostSlot::new(device, 4, DType::U32)?;
        slot.fill_u32(0, 4)?;
        let tensor = slot.tensor(4)?;
        let (storage, _) = tensor.storage_and_layout();
        let Storage::Metal(error) = &*storage else {
            unreachable!("Metal slot")
        };
        Ok(Self {
            slot,
            error: error.clone(),
            last_use: Cell::new(None),
            poisoned: Cell::new(false),
            words: Vec::with_capacity(4),
        })
    }

    fn encode(&self, consumer: Consumer<'_>) -> Result<()> {
        let device = self.error.device();
        if self.poisoned.get() {
            crate::bail!("guarded consumer: poisoned encoding");
        }
        if !device.commands().executor().is_classical() {
            crate::bail!("guarded consumer: replay requires generation-scoped error ownership");
        }
        device
            .declared_allocation_origin()
            .validate_reuse(&device.allocation_origin()?)?;
        let encoder = device.command_encoder()?;
        self.last_use.set(Some(device.gpu_clock().current_epoch()));
        self.poisoned.set(true);
        // SAFETY: private carrier preflight checks spans, aliases and device;
        // its owned sample and request guard retain allocation authority.
        unsafe {
            sampling_guard::call(device.metal_device(), &encoder, device.kernels(), consumer)
        }
        .map_err(MetalError::from)?;
        self.poisoned.set(false);
        Ok(())
    }

    pub fn bind(&self, sample: PrivateMultinomial) -> Result<GuardedSample<'_>> {
        if sample.rows > u32::MAX as usize || sample.vocab == 0 || sample.vocab > u32::MAX as usize
        {
            crate::bail!("guarded consumer: row/vocabulary geometry");
        }
        let token = GuardedSample {
            sample,
            guard: self,
            workspace_use: None,
        };
        token.check(&token.sample.ids, DType::U32, 0, token.sample.rows)?;
        token.check(
            &token.sample.status,
            DType::U32,
            0,
            token
                .sample
                .rows
                .checked_mul(4)
                .ok_or_else(|| crate::Error::Msg("guarded status overflow".into()))?,
        )?;
        self.encode(Consumer::Validate {
            ids: token.sample.ids.buffer(),
            status: token.sample.status.buffer(),
            error: self.error.buffer(),
            rows: token.sample.rows as u32,
            vocab: token.sample.vocab as u32,
        })?;
        Ok(token)
    }

    /// Existing queue completion and failure channel, no commit, spin or wait.
    /// A consumer must poll this even when a failed publication never stamps.
    pub fn poll_error(&mut self) -> Result<Option<[u32; 4]>> {
        let device = self.error.device();
        device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        if self.poisoned.get() {
            crate::bail!("guarded consumer: poisoned encoding");
        }
        if self
            .last_use
            .get()
            .is_some_and(|e| !device.gpu_clock().is_complete(e))
        {
            return Ok(None);
        }
        device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        // SAFETY: no GPU writer remains before reading a multiword diagnostic.
        self.slot.peek_u32(4, &mut self.words)?;
        Ok(Some([
            self.words[0],
            self.words[1],
            self.words[2],
            self.words[3],
        ]))
    }
}

impl GuardedSample<'_> {
    fn encode(&self, consumer: Consumer<'_>) -> Result<()> {
        let result = self.guard.encode(consumer);
        if let Some(epoch) = self.workspace_use {
            epoch.set(self.guard.last_use.get());
        }
        result
    }

    pub fn snapshot(&self, reader: &mut super::readback::SampleReadback) -> Result<()> {
        let result = reader.enqueue_guarded(self.sample.clone(), self.guard.error.clone());
        if let Some(epoch) = self.workspace_use {
            epoch.set(Some(self.guard.error.device().gpu_clock().current_epoch()));
        }
        result
    }
    fn check(
        &self,
        s: &MetalStorage,
        dtype: DType,
        start: usize,
        count: usize,
    ) -> Result<(usize, usize)> {
        if !s.device().same_device(self.guard.error.device()) {
            crate::bail!("guarded consumer: device mismatch");
        }
        span(s, dtype, start, count)
    }

    fn output(
        &self,
        s: &MetalStorage,
        dtype: DType,
        start: usize,
        count: usize,
    ) -> Result<(usize, usize)> {
        let out = self.check(s, dtype, start, count)?;
        s.device()
            .declared_allocation_origin()
            .validate_reuse(&s.device().allocation_origin()?)?;
        for input in [
            &self.sample.ids,
            &self.sample.status,
            &self.sample.support,
            &self.guard.error,
        ] {
            disjoint(s, out, input, span(input, input.dtype, 0, input.count)?)?;
        }
        Ok(out)
    }

    pub fn copy_into(&self, out: &MetalStorage, start: usize) -> Result<()> {
        self.output(out, DType::U32, start, self.sample.rows)?;
        self.encode(Consumer::Copy {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            out: (out.buffer(), start * 4),
            rows: self.sample.rows as u32,
        })
    }

    pub fn record_counts(&self, counts: &MetalStorage) -> Result<()> {
        self.output(counts, DType::F32, 0, self.sample.vocab)?;
        self.encode(Consumer::Counts {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            counts: counts.buffer(),
            rows: self.sample.rows as u32,
        })
    }

    /// Contiguous [vocab, width] table and prepared [rows, width] output. No
    /// invalid id is read as an embedding index, even in queued lookahead.
    pub fn embedding_into(
        &self,
        table: &MetalStorage,
        table_start: usize,
        width: usize,
        out: &MetalStorage,
        out_start: usize,
    ) -> Result<()> {
        if !matches!(table.dtype, DType::F16 | DType::BF16 | DType::F32) || width == 0 {
            crate::bail!("guarded embedding: dtype/width");
        }
        let table_count = self
            .sample
            .vocab
            .checked_mul(width)
            .ok_or_else(|| crate::Error::Msg("guarded embedding overflow".into()))?;
        let count = self
            .sample
            .rows
            .checked_mul(width)
            .ok_or_else(|| crate::Error::Msg("guarded embedding overflow".into()))?;
        let tr = self.check(table, table.dtype, table_start, table_count)?;
        let or = self.output(out, table.dtype, out_start, count)?;
        disjoint(table, tr, out, or)?;
        let size = table.dtype.size_in_bytes();
        let bytes = count
            .checked_mul(size)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("guarded embedding grid overflow".into()))?;
        let row_bytes = width
            .checked_mul(size)
            .ok_or_else(|| crate::Error::Msg("guarded embedding width overflow".into()))?;
        self.encode(Consumer::Embedding {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            table: (table.buffer(), table_start * size),
            out: (out.buffer(), out_start * size),
            row_bytes: row_bytes as u64,
            bytes: bytes as u64,
        })
    }

    /// One-ring-token budget. All mutable state and outputs are supplied from
    /// prepared request storage. The returned carrier preserves the guard.
    pub fn feed_into(
        &mut self,
        forced: &MetalStorage,
        close: &MetalStorage,
        state: &MetalStorage,
        out: &MetalStorage,
        free: u32,
    ) -> Result<()> {
        if self.sample.rows != 1
            || forced.count > u32::MAX as usize
            || close.count > u32::MAX as usize
        {
            crate::bail!("guarded feed: geometry");
        }
        let fr = self.check(forced, DType::U32, 0, forced.count)?;
        let cr = self.check(close, DType::U32, 0, close.count)?;
        let sr = self.output(state, DType::U32, 0, 2)?;
        let or = self.output(out, DType::U32, 0, 1)?;
        disjoint(state, sr, out, or)?;
        for (input, r) in [(forced, fr), (close, cr)] {
            disjoint(input, r, state, sr)?;
            disjoint(input, r, out, or)?;
        }
        self.encode(Consumer::Feed {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            forced: forced.buffer(),
            close: close.buffer(),
            state: state.buffer(),
            out: out.buffer(),
            free,
            forced_len: forced.count as u32,
            close_len: close.count as u32,
            vocab: self.sample.vocab as u32,
        })?;
        self.sample.ids = out.clone();
        Ok(())
    }

    pub fn eos_flag_into(&self, eos_ids: &MetalStorage, flag: &MetalStorage) -> Result<()> {
        if self.sample.rows != 1 || eos_ids.count > u32::MAX as usize {
            crate::bail!("guarded EOS: one token and u32 EOS count required");
        }
        let er = self.check(eos_ids, DType::U32, 0, eos_ids.count)?;
        let fr = self.output(flag, DType::U8, 0, 1)?;
        disjoint(eos_ids, er, flag, fr)?;
        self.encode(Consumer::Eos {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            eos_ids: eos_ids.buffer(),
            flag: flag.buffer(),
            count: eos_ids.count as u32,
        })
    }

    /// Certify history, EOS and next-input writes through normal encoder hazard
    /// binding. An invalid sample neither writes a token nor advances a marker.
    pub fn publish_marker(
        &self,
        marker: &MetalStorage,
        generation: u32,
        history: &MetalStorage,
        input: &MetalStorage,
        eos: &MetalStorage,
    ) -> Result<()> {
        if generation == 0 {
            crate::bail!("guarded publication: generation zero is unpublished");
        }
        let mr = self.output(marker, DType::U32, 0, 1)?;
        let er = self.check(eos, DType::U8, 0, 1)?;
        disjoint(marker, mr, eos, er)?;
        for s in [history, input] {
            let r = self.check(s, DType::U32, 0, s.count)?;
            disjoint(marker, mr, s, r)?;
        }
        self.encode(Consumer::Marker {
            error: self.guard.error.buffer(),
            marker: marker.buffer(),
            generation,
            history: history.buffer(),
            input: input.buffer(),
            eos: eos.buffer(),
        })
    }
}

#[cfg(test)]
mod tests;
