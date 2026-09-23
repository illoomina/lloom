//! A ring of host-writable staging buffers for a streamed weight tier: the
//! host fills one, a compute-encoder copy lands it in its slot, and the
//! buffer is reused once the command buffer holding that copy has completed
//! — the `depth+1` discipline of `design/docs/expert-streaming.md` §20.4a,
//! gated on the GPU clock (§6.3b) with a relaxed atomic load and never a
//! wait: a ring with nothing complete hands out a fresh buffer instead.
//!
//! Outside every pool on purpose: a pooled host upload is the tenant class
//! whose reuse gate lloom #748 found leaking at long context, and a streamed
//! tier writes hundreds of them per token.
//!
//! Bounded: the ring keeps at most `cap_bytes` of buffers. Past the cap a
//! fresh buffer is a one-shot, held in `overflow` until its epoch completes
//! and freed then — never dropped while a command buffer may still read it,
//! so the liveness holds under `LLOOM_CB_UNRETAINED=1` as well (the
//! unretained form rests on OUR discipline, see `commands::cb_unretained`).
//! A prefill burst thus costs its transient bytes once; steady-state decode
//! allocates nothing.

use super::buffer_pool::{GpuClock, PooledBuffer};
use crate::{MetalDevice, Result};
use std::sync::Arc;

/// An entry acquired and not yet stamped: no epoch completes it.
const IN_USE: u64 = u64::MAX;

struct Entry {
    buf: Arc<PooledBuffer>,
    /// The epoch of the command buffer the last copy from this entry was
    /// encoded into; reusable once the clock shows it complete.
    epoch: u64,
}

pub struct StagingRing {
    entries: Vec<Entry>,
    /// One-shots past the cap, freed once their epoch completes.
    overflow: Vec<Entry>,
    next: usize,
    cap_bytes: usize,
    clock: Arc<GpuClock>,
    allocations: u64,
    reuses: u64,
    overflows: u64,
}

/// A ring shared by every layer of a streamed tier: one bound, one clock.
pub type SharedStagingRing = Arc<parking_lot::Mutex<StagingRing>>;

/// Bytes a ring keeps by default: a decode step's loads several times over
/// (a miss is three ~0.6 MB planes on Qwen4-exp; ~30 per layer at 40 % hit
/// over 48 layers ≈ 0.5 GB per token).
pub const DEFAULT_CAP_BYTES: usize = 1 << 30;

impl StagingRing {
    pub fn new(device: &MetalDevice) -> Self {
        Self::with_cap_bytes(device, DEFAULT_CAP_BYTES)
    }

    pub fn with_cap_bytes(device: &MetalDevice, cap_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            overflow: Vec::new(),
            next: 0,
            cap_bytes,
            clock: device.gpu_clock(),
            allocations: 0,
            reuses: 0,
            overflows: 0,
        }
    }

    pub fn shared(device: &MetalDevice) -> SharedStagingRing {
        Arc::new(parking_lot::Mutex::new(Self::new(device)))
    }

    /// [`Self::shared`] at `cap_bytes` (a tier with a prefetcher wants room
    /// for what its reader threads stage ahead, `nn::expert_prefetch`).
    pub fn shared_with_cap(device: &MetalDevice, cap_bytes: usize) -> SharedStagingRing {
        Arc::new(parking_lot::Mutex::new(Self::with_cap_bytes(
            device, cap_bytes,
        )))
    }

    /// The cap the ring was built with.
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// A Shared buffer of at least `len` bytes whose last copy has completed,
    /// or a fresh one, as `(entry, buffer)`. The entry is in use — never
    /// reusable — until [`Self::stamp`] records the epoch of the command
    /// buffer its copy was encoded into. At the cap with nothing complete the
    /// buffer is a one-shot, kept until its epoch completes and freed then;
    /// nobody waits.
    pub fn acquire(
        &mut self,
        device: &MetalDevice,
        len: usize,
    ) -> Result<(Option<usize>, Arc<PooledBuffer>)> {
        self.overflow.retain(|e| !self.clock.is_complete(e.epoch));
        let n = self.entries.len();
        for i in 0..n {
            let idx = (self.next + i) % n;
            let e = &self.entries[idx];
            if e.buf.size() >= len && self.clock.is_complete(e.epoch) {
                self.entries[idx].epoch = IN_USE;
                self.next = (idx + 1) % n;
                self.reuses += 1;
                return Ok((Some(idx), Arc::clone(&self.entries[idx].buf)));
            }
        }
        let buf = device.new_unpooled_shared(len)?;
        if self.bytes() + len > self.cap_bytes && n > 0 {
            self.overflows += 1;
            self.overflow.push(Entry {
                buf: Arc::clone(&buf),
                epoch: IN_USE,
            });
            return Ok((None, buf));
        }
        self.entries.push(Entry {
            buf: Arc::clone(&buf),
            epoch: IN_USE,
        });
        self.next = 0;
        self.allocations += 1;
        Ok((Some(self.entries.len() - 1), buf))
    }

    /// Record the epoch the copy from `buf` was encoded into — a ring entry
    /// by index, a one-shot by its buffer. Read the clock AFTER taking the
    /// compute encoder for the copy: a command-buffer cut between the
    /// acquisition and the encode would otherwise leave the entry stamped one
    /// epoch early, and it would be reused while its copy is still pending
    /// (the defect the first ring shipped with: `-inf` logits at 382 slots).
    pub fn stamp(&mut self, entry: Option<usize>, buf: &Arc<PooledBuffer>, epoch: u64) {
        match entry {
            Some(i) => self.entries[i].epoch = epoch,
            None => {
                if let Some(e) = self
                    .overflow
                    .iter_mut()
                    .rev()
                    .find(|e| Arc::ptr_eq(&e.buf, buf))
                {
                    e.epoch = epoch;
                }
            }
        }
    }

    /// Hand an acquired entry back UNUSED — no copy was encoded from it (a
    /// prefetch the layer never needed). A ring entry is reusable at once
    /// (epoch 0: nothing reads it); a one-shot is freed now.
    pub fn release(&mut self, entry: Option<usize>, buf: &Arc<PooledBuffer>) {
        match entry {
            Some(i) => self.entries[i].epoch = 0,
            None => self.overflow.retain(|e| !Arc::ptr_eq(&e.buf, buf)),
        }
    }

    /// The clock the ring gates on.
    pub fn clock(&self) -> &Arc<GpuClock> {
        &self.clock
    }

    /// `(buffers allocated, acquisitions served by reuse, one-shots past the cap)`.
    pub fn counters(&self) -> (u64, u64, u64) {
        (self.allocations, self.reuses, self.overflows)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes the ring keeps.
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.buf.size()).sum()
    }

    /// One-shots still held for a pending epoch.
    pub fn overflow_pending(&self) -> usize {
        self.overflow.len()
    }
}

impl std::fmt::Debug for StagingRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StagingRing {{ buffers: {}, allocations: {}, reuses: {}, overflows: {} }}",
            self.entries.len(),
            self.allocations,
            self.reuses,
            self.overflows
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn wait_gpu(dev: &MetalDevice) {
        // Something must be encoded for the epoch to close and complete.
        let t = crate::Tensor::zeros(16, crate::DType::F32, &Device::Metal(dev.clone())).unwrap();
        let _ = t.to_vec1::<f32>().unwrap(); // commits and waits
    }

    /// Two acquisitions inside one open command buffer allocate twice; once
    /// that buffer has completed, the ring serves the next from the first
    /// entry without allocating; an unstamped entry is never reused.
    #[test]
    fn reuse_waits_for_the_epoch_and_never_blocks() {
        let dev = match Device::new_metal(0) {
            Ok(Device::Metal(m)) => m,
            _ => return,
        };
        let mut ring = StagingRing::with_cap_bytes(&dev, 4096 * 2 + (1 << 20));
        let epoch = ring.clock().current_epoch();
        let (ia, a) = ring.acquire(&dev, 4096).unwrap();
        let (ib, b) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(
            ring.counters(),
            (2, 0, 0),
            "an open epoch serves nothing by reuse"
        );
        assert!(!Arc::ptr_eq(&a, &b));
        ring.stamp(ia, &a, epoch);
        wait_gpu(&dev); // `epoch` is complete; b is still unstamped
        let (ic, c) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(
            ring.counters(),
            (2, 1, 0),
            "the stamped, completed entry is reused"
        );
        assert!(Arc::ptr_eq(&a, &c) && ic == ia);
        ring.stamp(ib, &b, epoch);
        ring.stamp(ic, &c, ring.clock().current_epoch());
        let (_, d) = ring.acquire(&dev, 4096).unwrap();
        assert!(
            Arc::ptr_eq(&b, &d),
            "b, stamped complete, is the next reuse"
        );
        // a larger request never fits an old entry: the last that fits the cap
        let (ie, e) = ring.acquire(&dev, 1 << 20).unwrap();
        assert_eq!(ring.counters().0, 3);
        assert!(ie.is_some());
        ring.stamp(ie, &e, ring.clock().current_epoch());
        // at the cap with nothing complete: a one-shot, held for its epoch
        let (i_over, f) = ring.acquire(&dev, 4096).unwrap();
        assert!(i_over.is_none());
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.counters().2, 1);
        ring.stamp(i_over, &f, ring.clock().current_epoch());
        assert_eq!(ring.overflow_pending(), 1);
        wait_gpu(&dev);
        // its epoch complete, the one-shot is released on the next acquire
        let (ig, _g) = ring.acquire(&dev, 4096).unwrap();
        assert!(ig.is_some(), "a completed entry serves it, not a one-shot");
        assert_eq!(ring.overflow_pending(), 0);
        assert_eq!(ring.len(), 3);
    }
}
