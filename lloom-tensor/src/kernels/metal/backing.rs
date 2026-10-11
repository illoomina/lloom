//! L0 backings (`profiles-and-memory-design.md` §4.1): where a container's
//! bytes come from. A container holds handles (slot ids, page indices,
//! frame indices, offsets) and asks a [`Backing`] for the buffers behind
//! them, so the backing is swappable and composable without touching
//! container code — [`Fallback`] takes from one backing first and another
//! when the first refuses.
//!
//! Four backings, one decision each:
//! - [`DeviceBuffer`]: one `MTLBuffer` per request (`newBufferWithLength`),
//!   resident in the device's residency set — committed now ([`Commit::Now`],
//!   a persistent buffer) or at the next command-buffer cut
//!   ([`Commit::Deferred`], a pool miss, which batches a cold prefill's
//!   hundred misses into one commit, §6.3b);
//! - [`Heap`]: a sub-range of a placement `MTLHeap` chunk
//!   ([`HeapChunks`]), resident with its chunk — the range goes back to the
//!   chunk on [`Backing::reclaim`];
//! - [`Mapped`]: a Shared [`DeviceBuffer`] with its pages wired (`mlock`),
//!   so the kernel cannot compress them under pressure (#749 P27);
//! - [`Fallback`]: the first backing, else the second.
//!
//! What stays the holder's: the residency *removal* of a whole-buffer
//! allocation (`remove` now, `remove_deferred`, or `retire_batch` — the
//! cadence #166 measured, chosen per site). A backing hands out resident
//! bytes; it does not decide when the set forgets them.
//!
//! **The ledger (§4.2) sees every byte here.** A whole-buffer backing with a
//! ledger takes a [`Reservation`] for each buffer it hands out and returns it
//! with the buffer ([`Allocation`]); the holder keeps it for the buffer's
//! life — a pool keeps it on the buffer's node through every free / acquire
//! cycle and drops it when the node is destroyed — so the ledger's
//! `committed` is what the process holds through its backings, pools' free
//! and pending lists included, and a pass admission is arithmetic on it
//! rather than a read of `currentAllocatedSize`. A heap's chunks reserve at
//! chunk granularity ([`HeapChunks::new`]); its sub-buffers carry none.

use super::heap::{HeapChunks, HeapStats};
use super::ledger::{Ledger, Reservation};
use super::{Buffer, Device, ResidencySet};
pub use crate::allocation_admission::PredictionBasis;
use crate::allocation_admission::{self, Prediction};
use crate::kernels::MetalKernelError;
use crate::memory_admission::AllocationContext;
pub type BackingAdmissionError = crate::allocation_admission::AllocationError<MetalKernelError>;
use objc2_metal::{MTLDevice as _, MTLResource as _, MTLResourceOptions};
use std::sync::Arc;

/// What a backing holds, in the heap's terms (a whole-buffer backing
/// reports zeros: nothing is sub-allocated).
pub type BackingStats = HeapStats;

/// A buffer and, when the backing reserves per buffer, its reservation.
#[derive(Debug)]
pub struct Allocation {
    pub buffer: Buffer,
    pub reservation: Option<Reservation>,
}

/// A bare buffer, reserved by nobody (a test's, a probe's).
impl From<Buffer> for Allocation {
    fn from(buffer: Buffer) -> Self {
        Self {
            buffer,
            reservation: None,
        }
    }
}

/// Where bytes come from.
pub trait Backing: Send + Sync + std::fmt::Debug {
    /// `bytes` resident bytes of this backing's storage mode.
    fn alloc(&self, bytes: usize) -> Result<Allocation, MetalKernelError>;
    /// Explicit admitted route. Unsupported implementations MUST NOT silently
    /// fall back to legacy unadmitted allocation. Heap enrollment remains pending.
    fn alloc_in(
        &self,
        _bytes: usize,
        _context: &AllocationContext,
    ) -> Result<Allocation, BackingAdmissionError> {
        Err(BackingAdmissionError::Unsupported(
            "backing has no admitted allocation route",
        ))
    }
    /// Take `buffer`'s range back, if it is this backing's sub-allocation;
    /// `false` for a buffer that is not (or is a whole allocation, which
    /// frees when its last handle drops — its residency entry is the
    /// holder's to remove).
    fn reclaim(&self, buffer: &Buffer) -> bool;
    fn stats(&self) -> BackingStats;
}

/// When a [`DeviceBuffer`]'s residency entry is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Commit {
    /// At the next command-buffer cut (`ResidencySet::add`): the allocation
    /// is staged (one `addAllocation`) and `commit_pending`, which
    /// `Commands::commit_swap_locked` runs before it commits, publishes the
    /// cut's staged buffers once — before any dispatch that binds them
    /// executes (`DESIGN.md` §3.7), so the residency invariant holds and the
    /// bytes are the same. The pools miss this way: a cold-pool prefill
    /// misses on the order of a hundred buffers and §6.3b prices a residency
    /// commit at ~0.087 ms, so batching them to one commit is the whole of
    /// the first-forward residency cost.
    Deferred,
    /// Before `alloc` returns (`ResidencySet::insert`): a persistent buffer,
    /// allocated off the hot path and bound by a command buffer nothing has
    /// opened yet.
    Now,
}

/// One `MTLBuffer` per request, resident.
pub struct DeviceBuffer {
    device: Device,
    residency: Arc<ResidencySet>,
    options: MTLResourceOptions,
    commit: Commit,
    /// Where each buffer's bytes are committed (`None` in a bare test).
    ledger: Option<Arc<Ledger>>,
}

impl DeviceBuffer {
    pub fn new(
        device: &Device,
        residency: Arc<ResidencySet>,
        options: MTLResourceOptions,
        commit: Commit,
        ledger: Option<Arc<Ledger>>,
    ) -> Self {
        Self {
            device: device.clone(),
            residency,
            options,
            commit,
            ledger,
        }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("options", &self.options.0)
            .field("commit", &self.commit)
            .finish()
    }
}

impl Backing for DeviceBuffer {
    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, BackingAdmissionError> {
        let requested = bytes.max(1);
        if requested > self.device.as_ref().maxBufferLength() {
            return Err(BackingAdmissionError::Geometry(
                "request exceeds device maxBufferLength",
            ));
        }
        let estimate = self
            .device
            .as_ref()
            .heapBufferSizeAndAlignWithLength_options(requested, self.options);
        let prediction = Prediction::standalone(requested, estimate.size, estimate.align)?;
        let (raw, owner) = allocation_admission::allocate(
            context,
            prediction,
            |length| {
                self.device
                    .as_ref()
                    .newBufferWithLength_options(length, self.options)
                    .ok_or_else(|| MetalKernelError::FailedToCreateResource("Buffer".into()))
            },
            |raw| raw.allocatedSize(),
        )?;
        let buffer = Buffer::new_owned(raw, owner);
        match self.commit {
            Commit::Deferred => self.residency.add(&buffer),
            Commit::Now => self.residency.insert(&buffer),
        };
        // Parallel legacy grant/pacer diagnostics, not another process charge.
        let reservation = self.ledger.as_ref().map(|l| l.reserve(requested));
        Ok(Allocation {
            buffer,
            reservation,
        })
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, MetalKernelError> {
        let buffer = self.device.new_buffer(bytes.max(1), self.options)?;
        match self.commit {
            Commit::Deferred => self.residency.add(&buffer),
            Commit::Now => self.residency.insert(&buffer),
        };
        let reservation = self.ledger.as_ref().map(|l| l.reserve(bytes.max(1)));
        Ok(Allocation {
            buffer,
            reservation,
        })
    }

    fn reclaim(&self, _buffer: &Buffer) -> bool {
        false
    }

    fn stats(&self) -> BackingStats {
        BackingStats::default()
    }
}

/// A sub-range of a placement heap chunk.
pub type Heap = HeapChunks;

impl Backing for HeapChunks {
    /// The chunk holds the reservation; the sub-buffer carries none.
    fn alloc(&self, bytes: usize) -> Result<Allocation, MetalKernelError> {
        Ok(Allocation {
            buffer: HeapChunks::alloc(self, bytes)?,
            reservation: None,
        })
    }

    fn reclaim(&self, buffer: &Buffer) -> bool {
        self.release(buffer)
    }

    fn stats(&self) -> BackingStats {
        HeapChunks::stats(self)
    }
}

/// A Shared buffer with its pages wired.
pub struct Mapped {
    inner: DeviceBuffer,
}

impl Mapped {
    /// `options` must name a CPU-mapped storage mode: a Private buffer has
    /// no pages to wire (`contents()` is null) and is handed out unwired.
    pub fn new(
        device: &Device,
        residency: Arc<ResidencySet>,
        options: MTLResourceOptions,
        commit: Commit,
        ledger: Option<Arc<Ledger>>,
    ) -> Self {
        Self {
            inner: DeviceBuffer::new(device, residency, options, commit, ledger),
        }
    }
}

impl std::fmt::Debug for Mapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapped")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Backing for Mapped {
    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, BackingAdmissionError> {
        let a = self.inner.alloc_in(bytes, context)?;
        wire_pages(a.buffer.contents(), bytes);
        Ok(a)
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, MetalKernelError> {
        let a = self.inner.alloc(bytes)?;
        wire_pages(a.buffer.contents(), bytes);
        Ok(a)
    }

    fn reclaim(&self, buffer: &Buffer) -> bool {
        self.inner.reclaim(buffer)
    }

    fn stats(&self) -> BackingStats {
        self.inner.stats()
    }
}

/// `first`, else `then`.
#[derive(Debug)]
pub struct Fallback<A, B> {
    pub first: A,
    pub then: B,
}

impl<A: Backing, B: Backing> Backing for Fallback<A, B> {
    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, BackingAdmissionError> {
        match self.first.alloc_in(bytes, context) {
            Err(error) if error.permits_fallback(native_allocation_refusal) => {
                self.then.alloc_in(bytes, context)
            }
            result => result,
        }
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, MetalKernelError> {
        match self.first.alloc(bytes) {
            Ok(b) => Ok(b),
            Err(_) => self.then.alloc(bytes),
        }
    }

    fn reclaim(&self, buffer: &Buffer) -> bool {
        self.first.reclaim(buffer) || self.then.reclaim(buffer)
    }

    fn stats(&self) -> BackingStats {
        let (a, b) = (self.first.stats(), self.then.stats());
        BackingStats {
            chunks: a.chunks + b.chunks,
            size: a.size + b.size,
            used: a.used + b.used,
            live: a.live + b.live,
        }
    }
}

/// `mlock` `size` bytes at `ptr` (a Shared buffer's own mapping); a failure
/// is reported, not fatal — the buffer works unwired, only slower under
/// pressure. A null `ptr` (a Private buffer) wires nothing.
pub(crate) fn wire_pages(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr..ptr+size` is the buffer's own mapping, alive for as
    // long as the buffer; `munlock` is implicit at unmap.
    let rc = unsafe { libc::mlock(ptr as *const libc::c_void, size.max(1)) };
    if rc != 0 {
        let e = std::io::Error::last_os_error();
        eprintln!("mlock of a {size}-byte Shared buffer failed: {e}");
    }
}

/// The device's backings: the transient pair the pools miss into (a
/// [`Heap`] when `transient_heaps_on()`, else a deferred-commit
/// [`DeviceBuffer`]), and the resident ones a persistent allocation takes
/// from.
pub struct Backings {
    pub transient_shared: Arc<dyn Backing>,
    pub transient_private: Arc<dyn Backing>,
    /// Persistent Shared, unwired (an unpooled upload).
    pub resident_shared: DeviceBuffer,
    /// Persistent Private (a persistent buffer, the arena and plan bases).
    pub resident_private: DeviceBuffer,
    /// Persistent Shared, wired (a slot pool, a KV plane).
    pub mapped_shared: Mapped,
}

impl std::fmt::Debug for Backings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backings")
            .field("transient_shared", &self.transient_shared)
            .field("transient_private", &self.transient_private)
            .finish()
    }
}

fn native_allocation_refusal(error: &MetalKernelError) -> bool {
    match error {
        MetalKernelError::FailedToCreateResource(_) => true,
        MetalKernelError::WithBacktrace { inner, .. } => native_allocation_refusal(inner),
        _ => false,
    }
}
