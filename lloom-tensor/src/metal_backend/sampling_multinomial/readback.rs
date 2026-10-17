//! Completion-checked host/window transfer. This is not a device-token API:
//! ring lookahead needs guarded device consumers before native admission.

use super::{result::checked_ids, PrivateMultinomial};
use crate::backend::{BackendDevice, BackendStorage};
use crate::kernels::metal::GpuClock;
use crate::metal_backend::{HostSlot, MetalDevice};
use crate::{DType, MetalError, MetalStorage, Result, Storage};
use std::sync::Arc;

/// Prepared outside the decode loop. A pending snapshot owns its source and
/// cannot be overwritten, including when the caller abandons a poll.
#[allow(dead_code, reason = "private bridge; native admission is separate")]
pub(super) struct SampleReadback {
    slot: HostSlot,
    destination: MetalStorage,
    device: MetalDevice,
    clock: Arc<GpuClock>,
    capacity: usize,
    words: Vec<u32>,
    pending: Option<(crate::retire::Epoch, PrivateMultinomial)>,
    poisoned: bool,
}

/// Only constructed after the entire completed result has passed validation.
/// No unchecked Tensor/storage escape exists, even for an invalid later row.
#[allow(dead_code, reason = "private bridge; native admission is separate")]
pub(super) struct CheckedSample<'a> {
    ids: &'a [u32],
}

#[allow(dead_code, reason = "private bridge; native admission is separate")]
impl CheckedSample<'_> {
    pub fn ids(&self) -> &[u32] {
        self.ids
    }
}

#[allow(dead_code, reason = "private bridge; native admission is separate")]
impl SampleReadback {
    pub fn new(device: &MetalDevice, capacity: usize) -> Result<Self> {
        let count = capacity
            .checked_mul(5)
            .filter(|&n| {
                n.checked_mul(4)
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| crate::Error::Msg("sampling readback capacity overflow".into()))?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(count)
            .map_err(|e| crate::Error::Msg(format!("sampling readback capacity: {e}")))?;
        let slot = HostSlot::new(device, count.max(1), DType::U32)?;
        let tensor = slot.tensor(count.max(1))?;
        let destination = {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(m) = &*storage else {
                unreachable!("HostSlot is Metal")
            };
            m.clone()
        };
        Ok(Self {
            slot,
            destination,
            device: device.clone(),
            clock: device.gpu_clock(),
            capacity,
            words,
            pending: None,
            poisoned: false,
        })
    }

    /// Two ordered copies ride the current encoder; no flush, wait, native
    /// completion subscriber or global failure injection is added. Prepared
    /// readback capacity is reused; encoder/pipeline costs remain their own.
    pub fn enqueue(&mut self, sample: PrivateMultinomial) -> Result<()> {
        if self.poisoned || self.pending.is_some() {
            crate::bail!("sampling readback is pending or poisoned");
        }
        // Replay can write this same snapshot again after its first epoch.
        // A direct-queue completion is not a replay-generation lease.
        if !self.device.commands().executor().is_classical() {
            crate::bail!(
                "sampling readback: captured/replayed execution needs generation-scoped snapshots"
            );
        }
        if sample.rows > self.capacity
            || !sample.ids.device().same_device(&self.device)
            || !sample.status.device().same_device(&self.device)
        {
            crate::bail!("sampling readback capacity/device mismatch");
        }
        if sample.rows == 0 {
            self.pending = Some((self.clock.current_epoch(), sample));
            return Ok(());
        }
        self.poisoned = true;
        let encoder = self.device.command_encoder()?;
        // CommandsGuard holds the recording-state exclusion: neither a cut
        // nor a concurrent commit can change this epoch between these copies.
        let epoch = self.clock.current_epoch();
        self.pending = Some((epoch, sample));
        let sample = &self.pending.as_ref().expect("stored above").1;
        for (source, count, offset) in [
            (&sample.status, sample.rows * 4, 0),
            (&sample.ids, sample.rows, sample.rows * 16),
        ] {
            // SAFETY: the private constructors own contiguous U32 outputs of
            // these exact extents. Setup checked 5*capacity*4 and owns the
            // disjoint Shared destination; pending retains sources. Both
            // copies use the same canonical encoder and its hazard tracking.
            unsafe {
                crate::kernels::call_copy2d(
                    self.device.metal_device(),
                    &encoder,
                    self.device.kernels(),
                    crate::kernels::copy2d::U32,
                    source.buffer(),
                    self.destination.buffer(),
                    1,
                    count,
                    count,
                    count,
                    0,
                    offset,
                )
            }
            .map_err(MetalError::from)?;
        }
        self.poisoned = false;
        Ok(())
    }

    /// Pending returns immediately. Completion/error admission uses the
    /// existing queue clock and failure channel; bytes are never polled early.
    /// The caller retains its existing commit cadence and wait policy.
    pub fn poll(&mut self) -> Result<Option<CheckedSample<'_>>> {
        self.device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        if self.poisoned {
            crate::bail!("sampling readback is poisoned");
        }
        let Some((epoch, sample)) = self.pending.as_ref() else {
            return Ok(None);
        };
        if sample.rows != 0 && !self.clock.is_complete(*epoch) {
            return Ok(None);
        }
        self.device
            .commands()
            .check_failed_atomic()
            .map_err(MetalError::from)?;
        let rows = sample.rows;
        let vocab = sample.vocab;
        self.slot.peek_u32(rows * 5, &mut self.words)?;
        // Validation precedes the success type, not merely host publication.
        let result = checked_ids(&self.words, rows, vocab)
            .map_err(|e| crate::Error::Msg(format!("sampling row {}: {:?}", e.row, e.reason)));
        self.pending = None;
        Ok(Some(CheckedSample { ids: result? }))
    }
}
