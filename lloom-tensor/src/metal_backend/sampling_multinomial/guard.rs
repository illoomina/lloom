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
use std::{cell::Cell, rc::Rc};

/// Request-exclusive sticky failure, initialized once before encoding. It is
/// never reset while queued consumers can still read it. Not Sync by design.
pub struct SampleGuard {
    slot: super::super::host_slot::HostReadView,
    error: MetalStorage,
    progress: Rc<ErrorProgress>,
    words: [u32; 4],
}

// One cold allocation per request. Both owners are deliberately !Send/!Sync:
// polling and encoding cannot race on another host thread. Nothing is added to
// RingReader, whose independent stream-reader thread retains its old surface.
#[derive(Default)]
struct ErrorProgress {
    last_use: Cell<Option<crate::retire::Epoch>>,
    poisoned: Cell<bool>,
}

/// Stack-only fallible-entry scope. Success never clears an older failure;
/// any incomplete entry (including unwind) monotonically poisons the request.
pub(super) struct NativeEntry<'a> {
    progress: &'a ErrorProgress,
    complete: bool,
}

impl NativeEntry<'_> {
    pub(super) fn note_epoch(&self, epoch: crate::retire::Epoch) {
        self.progress.last_use.set(Some(epoch));
    }

    pub(super) fn finish(mut self) {
        self.complete = true;
    }
}

impl Drop for NativeEntry<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.progress.poisoned.set(true);
        }
    }
}

/// Request-retained completion/error and deferred marker capability. This is
/// not a sampled-id escape, reset authority, or native generation enrollment.
/// Its private Rc enforces producer-thread polling and encoding. Create once
/// per request, not per token; every later writer updates the shared progress.
pub struct SamplePublisher {
    slot: super::super::host_slot::HostReadView,
    error: MetalStorage,
    progress: Rc<ErrorProgress>,
    words: [u32; 4],
}

/// The only device-id consumer surface. The sample and error owner remain live
/// until the carrier is consumed; buffer custody still follows the queue epoch.
pub struct GuardedSample<'a> {
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
        if self.progress.poisoned.get() || !self.error.device().same_device(device) {
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
            slot: slot.read_view(),
            error: error.clone(),
            progress: Rc::default(),
            words: [0; 4],
        })
    }

    pub(super) fn native_entry(&self) -> NativeEntry<'_> {
        NativeEntry {
            progress: &self.progress,
            complete: false,
        }
    }

    fn encode(&self, consumer: Consumer<'_>) -> Result<()> {
        encode(&self.error, &self.progress, consumer)
    }

    /// Retain after borrowed samples drop. No per-token clone is needed.
    pub fn publisher(&self) -> SamplePublisher {
        SamplePublisher {
            slot: self.slot.clone(),
            error: self.error.clone(),
            progress: Rc::clone(&self.progress),
            words: [0; 4],
        }
    }

    pub(super) fn bind(&self, sample: PrivateMultinomial) -> Result<GuardedSample<'_>> {
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
        poll_error(&self.error, &self.progress, &self.slot, &mut self.words)
    }

    /// Completion-checked request error, including failure without publication.
    pub fn check_error(&mut self) -> Result<bool> {
        let Some(words) = self.poll_error()? else {
            return Ok(false);
        };
        super::result::checked_error(&words)
            .map_err(|e| crate::Error::Msg(format!("sampling row {}: {:?}", e.row, e.reason)))?;
        Ok(true)
    }
}

// An installed executor is only the default for the NEXT encoder. Check the
// actual encoder while its CommandsGuard retains recording-state exclusion;
// this is a retained enum read, not another executor snapshot lock.
pub(super) fn require_classical_encoder(
    encoder: &crate::kernels::metal::commands::CommandsGuard<'_>,
) -> Result<()> {
    if !encoder.as_ref().executor.is_classical() {
        crate::bail!("sampling: actual encoder requires replay-generation ownership");
    }
    Ok(())
}

// The same native entry path serves borrowed and retained consumers. Epoch and
// poison publication happen before calling a fallible native lowerer. A failed
// call can never turn an older completed epoch into permission to read.
fn encode(error: &MetalStorage, progress: &ErrorProgress, consumer: Consumer<'_>) -> Result<()> {
    let device = error.device();
    if progress.poisoned.get() {
        crate::bail!("guarded consumer: poisoned encoding");
    }
    if !device.commands().executor().is_classical() {
        crate::bail!("guarded consumer: replay requires generation-scoped error ownership");
    }
    device
        .declared_allocation_origin()
        .validate_reuse(&device.allocation_origin()?)?;
    let encoder = device.command_encoder()?;
    require_classical_encoder(&encoder)?;
    progress
        .last_use
        .set(Some(device.gpu_clock().current_epoch()));
    progress.poisoned.set(true);
    // SAFETY: opaque consumer preflight checks spans, aliases and device;
    // request owners retain the exact-origin error allocation and queue clock.
    unsafe { sampling_guard::call(device.metal_device(), &encoder, device.kernels(), consumer) }
        .map_err(MetalError::from)?;
    progress.poisoned.set(false);
    Ok(())
}

fn poll_error(
    error: &MetalStorage,
    progress: &ErrorProgress,
    slot: &super::super::host_slot::HostReadView,
    words: &mut [u32; 4],
) -> Result<Option<[u32; 4]>> {
    let device = error.device();
    device
        .commands()
        .check_failed_atomic()
        .map_err(MetalError::from)?;
    if progress.poisoned.get() {
        crate::bail!("guarded consumer: poisoned encoding");
    }
    if progress
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
    // No concurrent host encoding is possible: shared progress is !Send/!Sync.
    // This exact queue has completed every possible writer of this request.
    slot.peek_u32_into(words)?;
    Ok(Some([words[0], words[1], words[2], words[3]]))
}

fn publish_marker(
    error: &MetalStorage,
    progress: &ErrorProgress,
    marker: &MetalStorage,
    generation: u32,
    history: &MetalStorage,
    input: &MetalStorage,
    eos: &MetalStorage,
) -> Result<()> {
    if generation == 0 {
        crate::bail!("guarded publication: generation zero is unpublished");
    }
    let mr = span(marker, DType::U32, 0, 1)?;
    marker
        .device()
        .declared_allocation_origin()
        .validate_reuse(&marker.device().allocation_origin()?)?;
    disjoint(marker, mr, error, span(error, DType::U32, 0, 4)?)?;
    for (s, dtype, count) in [
        (history, DType::U32, history.count),
        (input, DType::U32, input.count),
        (eos, DType::U8, 1),
    ] {
        if !s.device().same_device(error.device()) || !marker.device().same_device(error.device()) {
            crate::bail!("guarded publication: device mismatch");
        }
        disjoint(marker, mr, s, span(s, dtype, 0, count)?)?;
    }
    encode(
        error,
        progress,
        Consumer::Marker {
            error: error.buffer(),
            marker: marker.buffer(),
            generation,
            history: history.buffer(),
            input: input.buffer(),
            eos: eos.buffer(),
        },
    )
}

impl SamplePublisher {
    /// Deferred/final publication over already guarded writes, after the last
    /// sampled carrier drops. Error ownership and writer progress stay live.
    pub fn publish_marker(
        &self,
        marker: &MetalStorage,
        generation: u32,
        history: &MetalStorage,
        input: &MetalStorage,
        eos: &MetalStorage,
    ) -> Result<()> {
        publish_marker(
            &self.error,
            &self.progress,
            marker,
            generation,
            history,
            input,
            eos,
        )
    }

    /// Poll independently of token-marker progress; never wait or commit.
    pub fn check_error(&mut self) -> Result<bool> {
        let Some(words) = poll_error(&self.error, &self.progress, &self.slot, &mut self.words)?
        else {
            return Ok(false);
        };
        super::result::checked_error(&words)
            .map_err(|e| crate::Error::Msg(format!("sampling row {}: {:?}", e.row, e.reason)))?;
        Ok(true)
    }
}

impl GuardedSample<'_> {
    pub fn rows(&self) -> usize {
        self.sample.rows
    }

    pub fn vocab(&self) -> usize {
        self.sample.vocab
    }

    /// Dense embedding boundary, without exposing a sampled-id tensor. Output
    /// storage is prepared by the caller and remains under its normal claim.
    pub fn embedding_into_tensor(&self, table: &crate::Tensor, out: &crate::Tensor) -> Result<()> {
        let (vocab, width) = table.dims2()?;
        if vocab != self.sample.vocab
            || self.sample.rows.checked_mul(width) != Some(out.elem_count())
        {
            crate::bail!("guarded embedding: table/output geometry");
        }
        let (table_storage, table_layout) = table.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        if !table_layout.is_contiguous() || !out_layout.is_contiguous() {
            crate::bail!("guarded embedding: contiguous table/output required");
        }
        let (Storage::Metal(table), Storage::Metal(out)) = (&*table_storage, &*out_storage) else {
            crate::bail!("guarded embedding: Metal table/output required");
        };
        self.embedding_into(
            table,
            table_layout.start_offset(),
            width,
            out,
            out_layout.start_offset(),
        )
    }

    pub fn copy_into_tensor(&self, out: &crate::Tensor, start: usize) -> Result<()> {
        let (storage, layout) = out.storage_and_layout();
        if !layout.is_contiguous()
            || start
                .checked_add(self.sample.rows)
                .is_none_or(|n| n > out.elem_count())
        {
            crate::bail!("guarded copy: contiguous output span required");
        }
        let Storage::Metal(storage) = &*storage else {
            crate::bail!("guarded copy: Metal output required")
        };
        let start = layout
            .start_offset()
            .checked_add(start)
            .ok_or_else(|| crate::Error::Msg("guarded copy offset overflow".into()))?;
        self.copy_into(storage, start)
    }

    fn encode(&self, consumer: Consumer<'_>) -> Result<()> {
        let result = self.guard.encode(consumer);
        if let Some(epoch) = self.workspace_use {
            epoch.set(self.guard.progress.last_use.get());
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

    /// Packed GGUF embedding through the same get_rows/dequantization core as
    /// the incumbent route; request failure is tested before loading any id.
    pub(crate) fn quantized_embedding_into(
        &self,
        table: &crate::quantized::metal::QMetalStorage,
        vocab: usize,
        hidden: usize,
        out: &crate::Tensor,
    ) -> Result<()> {
        let dtype = table.dtype();
        let kind = crate::kernels::kernels::quantized::GgmlDType::from(dtype).kind();
        let count =
            self.sample.rows.checked_mul(hidden).ok_or_else(|| {
                crate::Error::Msg("guarded packed embedding output overflow".into())
            })?;
        let row_stride = hidden
            .checked_div(kind.block_size)
            .and_then(|n| n.checked_mul(kind.type_size))
            .filter(|_| hidden > 0 && hidden.is_multiple_of(kind.block_size))
            .ok_or_else(|| crate::Error::Msg("guarded packed embedding block geometry".into()))?;
        let bytes = vocab
            .checked_mul(row_stride)
            .filter(|&n| n <= table.buffer().length())
            .ok_or_else(|| crate::Error::Msg("guarded packed embedding physical span".into()))?;
        if vocab != self.sample.vocab
            || kind.guarded_get_rows.is_none()
            || !table.device().same_device(self.guard.error.device())
            || out.elem_count() != count
        {
            crate::bail!("guarded packed embedding: unsupported dtype/shape/device");
        }
        // Read-only weights may belong to another payer; active-origin reuse
        // applies to writable outputs/guard. This check does not establish
        // early-drop custody for raw native bindings.
        let (storage, layout) = out.storage_and_layout();
        let Storage::Metal(out) = &*storage else {
            crate::bail!("guarded packed embedding: Metal output required");
        };
        if !layout.is_contiguous() {
            crate::bail!("guarded packed embedding: prepared contiguous output required");
        }
        let range = self.output(out, DType::F32, layout.start_offset(), count)?;
        let base = table.buffer().base_offset();
        let end = base
            .checked_add(bytes)
            .ok_or_else(|| crate::Error::Msg("guarded packed embedding base overflow".into()))?;
        if out.buffer().raw_ptr() == table.buffer().raw_ptr() && range.0 < end && base < range.1 {
            crate::bail!("guarded packed embedding: aliased weight/output");
        }
        self.encode(Consumer::QuantizedEmbedding {
            ids: self.sample.ids.buffer(),
            error: self.guard.error.buffer(),
            table: table.buffer(),
            out: (out.buffer(), layout.start_offset() * 4),
            dtype: dtype.into(),
            hidden,
            row_stride,
            rows: self.sample.rows,
        })
    }

    /// One-ring-token budget. All mutable state and outputs are supplied from
    /// prepared request storage. The returned carrier preserves the guard.
    /// # Safety
    /// The caller exclusively owns `state` and `out`, including every Tensor
    /// or storage alias. No host or device mutation may change `out` after
    /// this feed through the final consumer encoding and completion of its
    /// queued reads. Retain all allocations under their existing queue/epoch
    /// custody. Safe callers use a request-owned, lifetime-bound prepared
    /// budget; raw interoperability must establish the same obligations.
    pub unsafe fn feed_into(
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
        self.output(marker, DType::U32, 0, 1)?;
        publish_marker(
            &self.guard.error,
            &self.guard.progress,
            marker,
            generation,
            history,
            input,
            eos,
        )
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod entry_tests {
    use super::*;

    #[test]
    fn native_entry_failure_and_unwind_are_monotonic_not_a_temporary_reset() {
        let progress = ErrorProgress::default();
        {
            let entry = NativeEntry {
                progress: &progress,
                complete: false,
            };
            entry.finish();
        }
        assert!(!progress.poisoned.get());
        drop(NativeEntry {
            progress: &progress,
            complete: false,
        });
        assert!(progress.poisoned.get());
        NativeEntry {
            progress: &progress,
            complete: false,
        }
        .finish();
        assert!(
            progress.poisoned.get(),
            "successful scope cannot clear an earlier failed request"
        );
        let progress = ErrorProgress::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _entry = NativeEntry {
                progress: &progress,
                complete: false,
            };
            panic!("native-entry unwind falsifier");
        }));
        assert!(panic.is_err());
        assert!(progress.poisoned.get());
    }
}
