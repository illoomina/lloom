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

use super::{Buffer, ResidencySet};
use crate::allocation_admission::{self, Prediction};
use crate::allocation_owner::AllocationOwner;
use crate::heap_ranges::{aligned, reusable_for, Ranges};
use crate::memory_admission::AllocationError;
use crate::memory_admission::{AllocationContext, DomainKey, ModelDomain};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLAllocation as _, MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLStorageMode,
};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
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
        AllocationError,
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
    membership: super::membership::Membership,
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
    membership: super::membership::Membership,
}

enum LivePlacement {
    Manual(Placement),
    // Exact never-reused receipt, NOT native pointer identity. A managed
    // placement remains reserved until its sole root-owned custody drops.
    Rooted { receipt: u64 },
}

/// Cold placement handoff. Field order releases the native Buffer before the
/// sole range-return owner. This is not a submit capability or an alias pin.
/// Only a trusted physical root may accept it; Buffer aliases alone do not
/// retain this custody. No Clone/raw accessors are provided.
pub(crate) struct PlacementAllocation {
    pub(crate) buffer: Buffer,
    pub(crate) custody: PlacementCustody,
}

/// Sole exact range-return owner, moved into independently service-rooted
/// physical A. It retains allocation-local state/heap/chunk charges only,
/// never Backings/Device/Commands/namespace/pool or the physical header.
/// Final Drop must be cold rejection or host service after all rich/native/IO
/// aliases and emission capabilities close. It is NOT producer-warm Drop.
pub(crate) struct PlacementCustody {
    // Retain state on any failed restoration or unwind. State itself contains
    // no custody owner, so this quarantine cannot form a reference cycle.
    state: ManuallyDrop<Arc<parking_lot::Mutex<State>>>,
    placement: Placement,
    receipt: u64,
    native_key: usize,
}
impl Drop for PlacementCustody {
    fn drop(&mut self) {
        let restored = {
            let mut st = self.state.lock();
            let p = self.placement;
            // A corrupt census quarantines; do not credit a range then panic.
            if st.rooted_live == 0 {
                return;
            }
            let Some(chunk) = st.chunks.get_mut(p.chunk) else {
                return;
            };
            if !chunk.ranges.give_back(p.offset, p.size) {
                return;
            }
            // Native Buffer address may be reused after its raw release but
            // before this Drop. Never remove a successor's live entry.
            if matches!(st.live.get(&self.native_key),
                Some(LivePlacement::Rooted { receipt }) if *receipt == self.receipt)
            {
                st.live.remove(&self.native_key);
            }
            // Prechecked before claim; only this non-Clone owner decrements.
            st.rooted_live -= 1;
            true
        };
        if restored {
            // SAFETY: exactly one successful range return; no guard remains.
            // Release heap/native/accounting owners outside allocator state.
            unsafe { ManuallyDrop::drop(&mut self.state) };
        }
    }
}

struct State {
    chunks: Vec<Chunk>,
    /// Keyed by the `MTLBuffer` pointer (`Buffer::raw_ptr`).
    live: HashMap<usize, LivePlacement>,
    next_receipt: u64,
    rooted_live: usize,
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
    state: Arc<parking_lot::Mutex<State>>,
    membership: Arc<super::membership::MembershipDomain>,
    default_domain: Option<ModelDomain>,
    requires_context: bool,
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
        let membership = super::membership::MembershipDomain::new(
            residency_set.clone(),
            crate::metal_backend::retain_evicted_allocations(),
        );
        Self {
            device: device.clone(),
            residency_set,
            membership,
            default_domain: None,
            requires_context: false,
            ledger,
            options,
            storage,
            chunk_bytes: chunk_bytes.max(1 << 20),
            state: Arc::new(parking_lot::Mutex::new(State {
                chunks: Vec::new(),
                live: HashMap::new(),
                next_receipt: 0,
                rooted_live: 0,
            })),
        }
    }

    /// Immutable engine-native payer, installed cold before chunks exist.
    pub(crate) fn with_native_payer(mut self, domain: ModelDomain, requires_context: bool) -> Self {
        self.default_domain = Some(domain);
        self.requires_context = requires_context;
        self
    }

    pub(crate) fn with_membership(
        mut self,
        membership: Arc<super::membership::MembershipDomain>,
    ) -> Self {
        self.membership = membership;
        self
    }

    /// Legacy unadmitted placement route, sharing the checked allocator with
    /// alloc_in but never borrowing an admitted domain's chunks.
    pub fn alloc(&self, length: usize) -> Result<Buffer, AllocationError> {
        self.alloc_impl(length, None, false)
            .map(|(buffer, _)| buffer)
    }

    pub(crate) fn alloc_in(
        &self,
        length: usize,
        context: &AllocationContext,
    ) -> Result<Buffer, AllocationError> {
        self.alloc_impl(length, Some(context), false)
            .map(|(buffer, _)| buffer)
    }

    /// Same checked heap allocator for default and supplied physical origins.
    /// Does not activate rich pool/Buffer callers by itself.
    ///
    /// # Safety
    /// Caller moves returned custody inseparably into the same physical A as
    /// buffer baseline before any usable raw alias escapes. Every future
    /// clone/view/encoder/IO owner retains that physical root's NativeUse until
    /// actual terminality and emission closure. Observer callbacks receive a
    /// counted External role, never this tokenless baseline. Final custody Drop
    /// is cold error handling or actual host-service cleanup, never warm final
    /// producer Drop. Existing legacy alloc/release remains a separate route.
    pub(crate) unsafe fn alloc_custodied(
        &self,
        length: usize,
        context: Option<&AllocationContext>,
    ) -> Result<PlacementAllocation, AllocationError> {
        let (buffer, custody) = self.alloc_impl(length, context, true)?;
        Ok(PlacementAllocation {
            buffer,
            custody: custody.expect("managed route always installs range custody"),
        })
    }

    fn select(
        st: &mut State,
        size: usize,
        align: usize,
        domain: Option<&DomainKey>,
    ) -> Result<Option<PendingPlacement>, AllocationError> {
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
        let offset = chunk
            .ranges
            .take_at(ri, size, align)
            .ok_or(AllocationError::Geometry(
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
            membership: chunk.membership.clone(),
        }))
    }

    fn stage_chunk(
        &self,
        size: usize,
        align: usize,
        context: Option<&AllocationContext>,
    ) -> Result<NewHeap, AllocationError> {
        let rounded = aligned(size, align)
            .ok_or(AllocationError::Geometry("heap chunk rounding overflow"))?;
        let chunk_size = self.chunk_bytes.max(rounded);
        let create = |bytes| {
            let desc = MTLHeapDescriptor::new();
            desc.setSize(bytes);
            desc.setStorageMode(self.storage);
            desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
            desc.setType(MTLHeapType::Placement);
            self.device.as_ref().newHeapWithDescriptor(&desc)
        };
        if let Some(context) = context {
            allocation_admission::stage_owned(
                context,
                Prediction::heap(chunk_size),
                create,
                |raw| raw.allocatedSize(),
            )
            .map(NewHeap::Admitted)
        } else if let Some(domain) = &self.default_domain {
            // Only cold growth reserves. Charged warm free-range reuse below
            // checks the immutable default key and does not take a promise.
            let quote = Prediction::heap(chunk_size).requested_owned_bytes()?;
            let pass = domain.reserve(quote)?;
            allocation_admission::stage_owned(
                &pass.context(),
                Prediction::heap(chunk_size),
                create,
                |raw| raw.allocatedSize(),
            )
            .map(NewHeap::Admitted)
        } else {
            create(chunk_size)
                .map(NewHeap::Legacy)
                .ok_or(AllocationError::DeviceOom {
                    requested: chunk_size,
                })
        }
    }

    fn commit_chunk(&self, staged: NewHeap, ranges: Ranges) -> Result<Chunk, AllocationError> {
        let (raw, owner) = staged.commit()?;
        let capacity = ranges.capacity();
        let heap = Arc::new(HeapAnchor { raw });
        let owner = owner.map(|owner| owner.with_anchor(heap.clone()));
        // The heap token, not each placed buffer, owns this membership. Every
        // placement/view carries it so dropping HeapChunks cannot unregister a
        // still-live sub-buffer. The token's raw owner has no Chunk backedge.
        let allocation = unsafe {
            &*(&*heap.raw as *const ProtocolObject<dyn MTLHeap>
                as *const ProtocolObject<dyn objc2_metal::MTLAllocation>)
        };
        let membership = self
            .membership
            .for_allocation(allocation, owner.clone())
            .map_err(AllocationError::from)?;
        let reservation = self.ledger.as_ref().map(|ledger| ledger.reserve(capacity));
        Ok(Chunk {
            heap,
            owner,
            membership,
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
        rooted: bool,
    ) -> Result<(Buffer, Option<PlacementCustody>), AllocationError> {
        if let Some(context) = context {
            if self
                .default_domain
                .as_ref()
                .is_some_and(|d| d.key() != context.domain_key())
            {
                return Err(AllocationError::DifferentDomain);
            }
            context.check_open()?;
        } else if self.requires_context {
            return Err(AllocationError::Unsupported(
                "model heap requires supplied allocation context",
            ));
        }
        let length = length.max(1);
        if length > self.device.as_ref().maxBufferLength() {
            return Err(AllocationError::Geometry(
                "heap buffer exceeds device maxBufferLength",
            ));
        }
        let sa = self
            .device
            .as_ref()
            .heapBufferSizeAndAlignWithLength_options(length, self.options);
        let (size, align) = (sa.size.max(length), sa.align);
        if align == 0 {
            return Err(AllocationError::Geometry("zero heap buffer alignment"));
        }
        let domain = context
            .map(AllocationContext::domain_key)
            .or_else(|| self.default_domain.as_ref().map(ModelDomain::key));
        // A context must be open even when reusing an already charged chunk.
        // The zero ticket validates cancellation without charging reused bytes.
        let reuse_ticket = context.map(|context| context.take(0)).transpose()?;
        // Cold receipt allocation is bounded numeric state, not a raw-address
        // generation guess. Refusal occurs before any range claim. Gaps after
        // failed attempts are intentional; receipt values are never recycled.
        let receipt = if rooted {
            let mut st = self.state.lock();
            let receipt = st
                .next_receipt
                .checked_add(1)
                .ok_or(AllocationError::Geometry("placement receipt exhausted"))?;
            st.rooted_live
                .checked_add(1)
                .ok_or(AllocationError::Geometry("placement live census exhausted"))?;
            st.next_receipt = receipt;
            Some(receipt)
        } else {
            None
        };
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
                        return Err(AllocationError::Geometry(
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
                    membership: chunk.membership.clone(),
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
                return Err(AllocationError::Geometry("failed buffer range restoration"));
            }
            return Err(AllocationError::DeviceOom { requested: size });
        };
        let buffer = match pending.owner {
            Some(owner) => Buffer::new_owned(raw, owner),
            None => Buffer::new(raw),
        }
        .with_membership(pending.membership);
        let native_key = buffer.raw_ptr() as usize;
        let custody = if let Some(receipt) = receipt {
            let custody = PlacementCustody {
                state: ManuallyDrop::new(self.state.clone()),
                placement,
                receipt,
                native_key,
            };
            let mut st = self.state.lock();
            // Other growers may have increased the count since precheck.
            // If exhausted, release guard then native raw before exact custody
            // rollback; no entry is installed and no observable credit escapes.
            let Some(next_live) = st.rooted_live.checked_add(1) else {
                drop(st);
                drop(buffer);
                // This rejected owner has not joined the live census yet.
                // Restore its exact range directly, retaining state on failure.
                let custody = ManuallyDrop::new(custody);
                let restored = self.state.lock().chunks[placement.chunk]
                    .ranges
                    .give_back(placement.offset, placement.size);
                if restored {
                    // SAFETY: rejected exclusive owner; read state once,
                    // no live census decrement or ordinary custody Drop.
                    let state = unsafe { std::ptr::read(&*custody.state) };
                    drop(state);
                }
                return Err(AllocationError::Geometry("placement live census exhausted"));
            };
            st.rooted_live = next_live;
            st.live
                .insert(native_key, LivePlacement::Rooted { receipt });
            drop(st);
            Some(custody)
        } else {
            self.state
                .lock()
                .live
                .insert(native_key, LivePlacement::Manual(placement));
            None
        };
        Ok((buffer, custody))
    }

    /// Give a sub-buffer's range back (the pool evicting it, or a trim).
    /// Returns whether `buf` was one of ours; a buffer that is not is left
    /// to its own release.
    pub fn release(&self, buf: &Buffer) -> bool {
        let mut st = self.state.lock();
        let key = buf.raw_ptr() as usize;
        let Some(entry) = st.live.get(&key) else {
            return false;
        };
        let p = match entry {
            // Eviction acknowledges membership in this heap; actual range
            // eligibility belongs to the exact physical custody, not callback.
            LivePlacement::Rooted { .. } => return true,
            LivePlacement::Manual(p) => *p,
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
            live: st.rooted_live
                + st.live
                    .values()
                    .filter(|p| matches!(p, LivePlacement::Manual(_)))
                    .count(),
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
