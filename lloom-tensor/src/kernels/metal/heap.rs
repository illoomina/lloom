//! Transient allocations from PLACEMENT `MTLHeap` chunks that are in the
//! residency set ONCE, at offsets this allocator chooses (`DESIGN.md` §6.3k,
//! §11.5z-ix item 1).
//!
//! A pool miss used to be an `MTLBuffer` of its own plus a staged
//! `addAllocation`, and every command-buffer cut then committed the residency
//! set — ~0.7 ms a commit over a ~4 000-allocation set, 1.1 s of a 5.2 s
//! streamed prefill (#749 P26). A heap's sub-allocations are resident with
//! the heap, so a miss served from a chunk makes no residency call at all.
//! The first form let Metal place the sub-buffers (`MTLHeapTypeAutomatic`)
//! and its allocator fragmented under the pool's churn: 2.87 GB used of 4.29
//! allocated. This form places them itself (`MTLHeapTypePlacement`,
//! `newBufferWithLength:options:offset:`): one free-list of ranges per chunk,
//! first-fit with coalescing, the range returned when the pool evicts the
//! buffer (`release`) — so the chunks hold what the pool holds and no more.
//!
//! What this does NOT change: the persistent allocations (weights, slot
//! pools, KV, the session region) keep the set as before; the pool's own
//! bookkeeping (free budget, pending epochs, eviction order) is untouched —
//! only where a miss's bytes come from and where an eviction's go.
//! `currentAllocatedSize` counts a chunk at its creation, so the admission
//! gauge (§9.5) reads a conservative figure; the pass admission credits the
//! chunks' free bytes back (`plan_pass`).
//!
//! Aliasing: a placement heap does not track it. A range is handed out once
//! and comes back only from the pool's eviction of a buffer whose GPU epoch
//! has completed (the pool evicts from its FREE list), so no two live
//! buffers overlap and no returned range is still read.

use super::backing::BackingAdmissionError;
use super::{Buffer, ResidencySet};
use crate::allocation_admission::{self, Prediction};
use crate::allocation_owner::AllocationOwner;
use crate::heap_ranges::{aligned, reusable_for, Ranges};
use crate::kernels::MetalKernelError;
use crate::memory_admission::{AllocationContext, DomainKey};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLAllocation as _, MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLStorageMode,
};
use std::collections::HashMap;
use std::sync::Arc;

/// The default chunk: 512 MiB. A 512-row prefill tile's transients are
/// ~1.3 GB with two tiles in flight (P18), so a warmed prefill sits in three
/// to six; the air is the tail of the last chunk of each list.
/// `LLOOM_HEAP_CHUNK_MB` overrides.
pub const DEFAULT_HEAP_CHUNK_BYTES: usize = 512 << 20;

/// The chunk size in force (`LLOOM_HEAP_CHUNK_MB`, default 512), read once.
pub fn heap_chunk_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| crate::tuning::tuning().metal.heap_chunk_mb.max(16) << 20)
}

/// Whether pool misses are served from heap chunks (`LLOOM_TRANSIENT_HEAPS=1`;
/// off by default — the per-buffer arm is what every recorded digest ran
/// on). Measured under the paired prefill (#749 P26): the seam's residency
/// commit 1 101 → 1.5 ms of host time, the wall −3–5 %, and the device
/// footprint **+0.9 GB at the peak, +1.85 GB retained after it** — the
/// chunks hold the pass's peak transient set and the holes the churn leaves
/// in it (2.89 GB used of 4.83 at 512 MiB chunks, best fit; Metal's own
/// automatic placement packed 2.87 of 4.29), where the per-buffer arm frees
/// an evicted buffer to the OS. On this machine's budget the peak is the
/// dearer side; the arm is opt-in, and the fix for the air is structural —
/// a per-pass bump region for the prefill's transients (§9.2's arena, at
/// prefill), which these chunks are the place for. Read once.
pub fn transient_heaps_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().metal.transient_heaps)
}

pub const DEFAULT_TRANSIENT_HEAPS: bool = false;

/// Native heap anchor shared by chunk, sub-buffer and residency owners. It
/// deliberately contains no Device/Commands/PhysicalContext backedge.
struct HeapAnchor {
    raw: Retained<ProtocolObject<dyn MTLHeap>>,
}
// SAFETY: same thread-safe Metal heap ownership as HeapChunks. Host allocator
// metadata remains separately protected by HeapChunks::state.
unsafe impl Send for HeapAnchor {}
unsafe impl Sync for HeapAnchor {}

enum NewHeap {
    Admitted(allocation_admission::StagedAllocation<Retained<ProtocolObject<dyn MTLHeap>>>),
    Legacy(Retained<ProtocolObject<dyn MTLHeap>>),
}
impl NewHeap {
    fn capacity(&self) -> usize {
        match self {
            Self::Admitted(staged) => staged.raw().size(),
            Self::Legacy(raw) => raw.size(),
        }
    }
    fn commit(
        self,
    ) -> Result<
        (
            Retained<ProtocolObject<dyn MTLHeap>>,
            Option<AllocationOwner>,
        ),
        BackingAdmissionError,
    > {
        match self {
            Self::Admitted(staged) => staged.commit().map(|(raw, owner)| (raw, Some(owner))),
            Self::Legacy(raw) => Ok((raw, None)),
        }
    }
}

struct Chunk {
    // Drop native heap reference before the chunk's accounting owner.
    heap: Arc<HeapAnchor>,
    owner: Option<AllocationOwner>,
    ranges: Ranges,
    ready: bool,
    #[allow(dead_code)]
    reservation: Option<super::ledger::Reservation>,
}
impl Chunk {
    fn available_to(&self, domain: Option<&DomainKey>) -> bool {
        let owner_domain = self.owner.as_ref().map(AllocationOwner::domain_key);
        reusable_for(self.ready, owner_domain.as_ref(), domain)
    }
}

/// Where a live sub-buffer sits: its chunk and its range.
#[derive(Clone, Copy)]
struct Placement {
    chunk: usize,
    offset: usize,
    size: usize,
}

struct PendingPlacement {
    placement: Placement,
    heap: Arc<HeapAnchor>,
    owner: Option<AllocationOwner>,
}

struct State {
    chunks: Vec<Chunk>,
    /// Keyed by the `MTLBuffer` pointer (`Buffer::raw_ptr`).
    live: HashMap<usize, Placement>,
}

/// Chunks of one storage mode, and the residency set they are added to.
pub struct HeapChunks {
    device: super::Device,
    residency_set: Arc<ResidencySet>,
    /// Where a chunk's bytes are committed (`None` in a bare test).
    ledger: Option<Arc<super::ledger::Ledger>>,
    options: MTLResourceOptions,
    storage: MTLStorageMode,
    chunk_bytes: usize,
    state: parking_lot::Mutex<State>,
}

// SAFETY: an `MTLHeap` is a thread-safe Metal object; the state is behind a
// mutex.
unsafe impl Send for HeapChunks {}
unsafe impl Sync for HeapChunks {}

/// What the chunks hold: their count, their total size, the bytes their live
/// sub-buffers occupy, and the live sub-buffer count.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub chunks: usize,
    pub size: usize,
    pub used: usize,
    pub live: usize,
}

impl HeapChunks {
    /// Chunks for buffers of `options` (their storage mode is the heaps'),
    /// each `chunk_bytes` unless a request is larger.
    pub fn new(
        device: &super::Device,
        residency_set: Arc<ResidencySet>,
        options: MTLResourceOptions,
        chunk_bytes: usize,
        ledger: Option<Arc<super::ledger::Ledger>>,
    ) -> Self {
        // The storage mode is bits 4..8 of the resource options.
        let storage = MTLStorageMode((options.0 >> 4) & 0xF);
        Self {
            device: device.clone(),
            residency_set,
            ledger,
            options,
            storage,
            chunk_bytes: chunk_bytes.max(1 << 20),
            state: parking_lot::Mutex::new(State {
                chunks: Vec::new(),
                live: HashMap::new(),
            }),
        }
    }

    /// Legacy unadmitted placement route, sharing the checked allocator with
    /// alloc_in but never borrowing an admitted domain's chunks.
    pub fn alloc(&self, length: usize) -> Result<Buffer, MetalKernelError> {
        self.alloc_impl(length, None).map_err(|error| match error {
            BackingAdmissionError::Native(error) => error,
            other => MetalKernelError::InvalidInput(other.to_string()),
        })
    }

    pub(crate) fn alloc_in(
        &self,
        length: usize,
        context: &AllocationContext,
    ) -> Result<Buffer, BackingAdmissionError> {
        self.alloc_impl(length, Some(context))
    }

    fn select(
        st: &mut State,
        size: usize,
        align: usize,
        domain: Option<&DomainKey>,
    ) -> Result<Option<PendingPlacement>, BackingAdmissionError> {
        let mut best: Option<(usize, usize, usize)> = None;
        for (ci, chunk) in st.chunks.iter().enumerate() {
            if !chunk.available_to(domain) {
                continue;
            }
            if let Some((ri, waste)) = chunk.ranges.fit(size, align) {
                if best.is_none_or(|(_, _, old)| waste < old) {
                    best = Some((ci, ri, waste));
                }
            }
        }
        let Some((ci, ri, _)) = best else {
            return Ok(None);
        };
        let chunk = &mut st.chunks[ci];
        let offset =
            chunk
                .ranges
                .take_at(ri, size, align)
                .ok_or(BackingAdmissionError::Geometry(
                    "selected heap range became invalid",
                ))?;
        Ok(Some(PendingPlacement {
            placement: Placement {
                chunk: ci,
                offset,
                size,
            },
            heap: chunk.heap.clone(),
            owner: chunk.owner.clone(),
        }))
    }

    fn stage_chunk(
        &self,
        size: usize,
        align: usize,
        context: Option<&AllocationContext>,
    ) -> Result<NewHeap, BackingAdmissionError> {
        let rounded = aligned(size, align).ok_or(BackingAdmissionError::Geometry(
            "heap chunk rounding overflow",
        ))?;
        let chunk_size = self.chunk_bytes.max(rounded);
        let create = |bytes| {
            let desc = MTLHeapDescriptor::new();
            desc.setSize(bytes);
            desc.setStorageMode(self.storage);
            desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
            desc.setType(MTLHeapType::Placement);
            self.device
                .as_ref()
                .newHeapWithDescriptor(&desc)
                .ok_or_else(|| {
                    MetalKernelError::FailedToCreateResource(format!(
                        "placement heap chunk of {bytes} bytes"
                    ))
                })
        };
        if let Some(context) = context {
            allocation_admission::stage(context, Prediction::heap(chunk_size), create, |raw| {
                raw.allocatedSize()
            })
            .map(NewHeap::Admitted)
        } else {
            create(chunk_size)
                .map(NewHeap::Legacy)
                .map_err(BackingAdmissionError::Native)
        }
    }

    fn commit_chunk(
        &self,
        staged: NewHeap,
        ranges: Ranges,
    ) -> Result<Chunk, BackingAdmissionError> {
        let (raw, owner) = staged.commit()?;
        let capacity = ranges.capacity();
        let heap = Arc::new(HeapAnchor { raw });
        let owner = owner.map(|owner| owner.with_anchor(heap.clone()));
        let reservation = self.ledger.as_ref().map(|ledger| ledger.reserve(capacity));
        Ok(Chunk {
            heap,
            owner,
            ranges,
            ready: false,
            reservation,
        })
    }

    fn publish_chunk(&self, heap: &HeapAnchor, owner: Option<&AllocationOwner>) {
        // SAFETY: MTLHeap implements MTLAllocation; borrowed protocol upcast.
        let allocation = unsafe {
            &*(&*heap.raw as *const ProtocolObject<dyn MTLHeap>
                as *const ProtocolObject<dyn objc2_metal::MTLAllocation>)
        };
        let key = Retained::as_ptr(&heap.raw) as usize;
        if let Some(owner) = owner {
            self.residency_set
                .insert_allocation_owned(allocation, key, owner.clone());
        } else {
            self.residency_set.insert_allocation(allocation, key);
        }
    }

    fn alloc_impl(
        &self,
        length: usize,
        context: Option<&AllocationContext>,
    ) -> Result<Buffer, BackingAdmissionError> {
        let length = length.max(1);
        if length > self.device.as_ref().maxBufferLength() {
            return Err(BackingAdmissionError::Geometry(
                "heap buffer exceeds device maxBufferLength",
            ));
        }
        let sa = self
            .device
            .as_ref()
            .heapBufferSizeAndAlignWithLength_options(length, self.options);
        let (size, align) = (sa.size.max(length), sa.align);
        if align == 0 {
            return Err(BackingAdmissionError::Geometry(
                "zero heap buffer alignment",
            ));
        }
        let domain = context.map(AllocationContext::domain_key);
        // A context must be open even when reusing an already charged chunk.
        // The zero ticket validates cancellation without charging reused bytes.
        let reuse_ticket = context
            .map(|context| context.take(0))
            .transpose()
            .map_err(BackingAdmissionError::Admission)?;
        let selected = {
            let mut st = self.state.lock();
            Self::select(&mut st, size, align, domain.as_ref())?
        };
        let pending = if let Some(selected) = selected {
            selected
        } else {
            // Cold demand, native creation and bookkeeping owner allocation are
            // outside state. A competing grower may make this chunk surplus.
            let staged = self.stage_chunk(size, align, context)?;
            let selected = {
                let mut st = self.state.lock();
                Self::select(&mut st, size, align, domain.as_ref())?
            };
            if let Some(selected) = selected {
                drop(staged); // Raw before uncommitted ticket: full entitlement returns.
                selected
            } else {
                // Selected growth commits outside state. Another grower can
                // create spare capacity after our finite recheck; its separate
                // ticket also bounds that growth. No claim of perfect packing.
                let mut ranges = Ranges::new(staged.capacity());
                let offset = match ranges.take(size, align) {
                    Some(offset) => offset,
                    None => {
                        drop(ranges);
                        drop(staged);
                        return Err(BackingAdmissionError::Geometry(
                            "fresh heap capacity cannot fit requested range",
                        ));
                    }
                };
                let chunk = self.commit_chunk(staged, ranges)?;
                let mut st = self.state.lock();
                let ci = st.chunks.len();
                let pending = PendingPlacement {
                    placement: Placement {
                        chunk: ci,
                        offset,
                        size,
                    },
                    heap: chunk.heap.clone(),
                    owner: chunk.owner.clone(),
                };
                st.chunks.push(chunk); // Not ready: concurrent scans cannot use it.
                drop(st);
                self.publish_chunk(&pending.heap, pending.owner.as_ref());
                self.state.lock().chunks[ci].ready = true;
                pending
            }
        };
        drop(reuse_ticket);
        let placement = pending.placement;
        // SAFETY: checked range reserved under state, unavailable to other
        // allocators; native parent is independently anchored across this call.
        let raw = unsafe {
            pending.heap.raw.newBufferWithLength_options_offset(
                size,
                self.options,
                placement.offset,
            )
        };
        let Some(raw) = raw else {
            let restored = self.state.lock().chunks[placement.chunk]
                .ranges
                .give_back(placement.offset, size);
            if !restored {
                return Err(BackingAdmissionError::Geometry(
                    "failed buffer range restoration",
                ));
            }
            return Err(BackingAdmissionError::Native(
                MetalKernelError::FailedToCreateResource(format!(
                    "{size} bytes at offset {} of placement heap",
                    placement.offset
                )),
            ));
        };
        let buffer = match pending.owner {
            Some(owner) => Buffer::new_owned(raw, owner),
            None => Buffer::new(raw),
        };
        self.state
            .lock()
            .live
            .insert(buffer.raw_ptr() as usize, placement);
        Ok(buffer)
    }

    /// Give a sub-buffer's range back (the pool evicting it, or a trim).
    /// Returns whether `buf` was one of ours; a buffer that is not is left
    /// to its own release.
    pub fn release(&self, buf: &Buffer) -> bool {
        let mut st = self.state.lock();
        let key = buf.raw_ptr() as usize;
        let Some(p) = st.live.get(&key).copied() else {
            return false;
        };
        if !st.chunks[p.chunk].ranges.give_back(p.offset, p.size) {
            return false;
        }
        st.live.remove(&key);
        true
    }

    pub fn collect_reusable_allocations(
        &self,
        domain: Option<&DomainKey>,
        credit: &mut super::backing::ReuseCredit,
    ) {
        let st = self.state.lock();
        for chunk in st.chunks.iter().filter(|chunk| chunk.available_to(domain)) {
            // State never removes chunks; the borrowed HeapChunks owns each
            // Arc<HeapAnchor> throughout collection, so no address reuse occurs.
            credit.observe(
                Retained::as_ptr(&chunk.heap.raw) as usize,
                chunk.ranges.available(),
            );
        }
    }

    pub fn stats(&self) -> HeapStats {
        let st = self.state.lock();
        HeapStats {
            chunks: st.chunks.len(),
            size: st.chunks.iter().map(|c| c.ranges.capacity()).sum(),
            used: st.chunks.iter().map(|c| c.ranges.used()).sum(),
            live: st.live.len(),
        }
    }
}

impl std::fmt::Debug for HeapChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HeapChunks {{ {:?}, chunk_bytes: {} }}",
            self.stats(),
            self.chunk_bytes
        )
    }
}

// Checked placement tests live in crate::heap_ranges and exercise the exact
// production implementation without allocating a native heap.
