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
pub use crate::heap_ranges::ReuseCredit;
use crate::memory_admission::{AllocationContext, AllocationError, DomainKey, ModelDomain};
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

/// Custom backing's exact range/eligibility return guard, finalized ONLY by
/// cold rejection or physical host service, after native baseline release.
/// It must not clone/own a pool/namespace/Commands/physical root backedge.
///
/// # Safety
/// Drop never panics and returns eligibility exactly once. The guard retains
/// every allocator/native-parent/account owner needed through that return.
/// `Backing::reclaim` must acknowledge this managed allocation without returning
/// its range early. All native/view/IO/emission aliases must be counted by the
/// enclosing physical root before any safe native use; raw interoperability
/// retains equivalent explicit custody. Merely retaining an SDK object is not
/// sufficient. Any failed return quarantines required owners, not false credit.
/// Custom implementations are an explicit unsafe interoperability boundary.
pub unsafe trait ExternalPlacementCustody: Send + Sync {}

pub(crate) enum BackingRangeCustody {
    Heap(super::heap::PlacementCustody),
    External(Box<dyn ExternalPlacementCustody>),
}

/// Cold backing handoff for actual independently rooted physical storage.
/// Public only because Backing is a public trait; fields/accessors remain
/// engine-private. No native or range-return capability escapes publicly.
pub struct CustodiedAllocation {
    allocation: Allocation,
    placement: Option<BackingRangeCustody>,
}
impl CustodiedAllocation {
    /// Custom whole-allocation backing construction, with no allocator range
    /// or other independent eligibility lease to return. No hidden allocation.
    /// # Safety
    /// allocation carries its complete raw/native-parent/account/ledger custody;
    /// no range-reuse obligation exists outside those owners. The same complete
    /// physical alias/emission/service-Drop contract as alloc_custodied applies.
    /// This does not forge admitted provenance: physical insertion validates
    /// owner/domain independently; legacy origins require explicit policy.
    pub unsafe fn from_whole_allocation(allocation: Allocation) -> Self {
        Self {
            allocation,
            placement: None,
        }
    }
    /// Custom ranged backing construction with an already cold-prepared guard.
    /// # Safety
    /// guard is the sole exact eligibility authority for THIS allocation,
    /// obeys ExternalPlacementCustody's contract, and cannot return eligibility
    /// through any other clone/reclaim path. It is moved intact into physical A.
    /// Guard/Box/control/native demand is included in the caller's actual quote
    /// before preparation; accepting this object performs no hidden allocation.
    /// Native/future-submit/raw obligations remain as alloc_custodied documents.
    pub unsafe fn from_external_placement(
        allocation: Allocation,
        guard: Box<dyn ExternalPlacementCustody>,
    ) -> Self {
        Self {
            allocation,
            placement: Some(BackingRangeCustody::External(guard)),
        }
    }
    pub(crate) fn into_parts(self) -> (Allocation, Option<BackingRangeCustody>) {
        (self.allocation, self.placement)
    }
    pub(crate) fn from_parts(
        allocation: Allocation,
        placement: Option<BackingRangeCustody>,
    ) -> Self {
        Self {
            allocation,
            placement,
        }
    }
}

/// Where bytes come from.
pub trait Backing: Send + Sync + std::fmt::Debug {
    /// Cold handoff preserving actual allocator range eligibility, for BOTH
    /// default and supplied origins. Unsupported backings refuse rather than
    /// silently supplying tokenless placement storage.
    ///
    /// # Safety
    /// The caller moves the complete returned object into independently
    /// service-rooted physical C/A before exposing usable aliases. All native,
    /// IO, view and emission roles retain counted physical use through terminal
    /// closure; range custody Drop is service-only or cold rejection, never a
    /// final producer Drop. Observer callbacks receive counted External roles.
    unsafe fn alloc_custodied(
        &self,
        _bytes: usize,
        _context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        Err(AllocationError::Unsupported(
            "backing has no physical range-custody route",
        ))
    }

    /// `bytes` resident bytes of this backing's storage mode.
    fn alloc(&self, bytes: usize) -> Result<Allocation, AllocationError>;
    /// Explicit admitted route. Unsupported implementations MUST NOT silently
    /// fall back to legacy unadmitted allocation. Each implementation opts in.
    fn alloc_in(
        &self,
        _bytes: usize,
        _context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        Err(AllocationError::Unsupported(
            "backing has no admitted allocation route",
        ))
    }
    /// Take `buffer`'s range back, if it is this backing's sub-allocation;
    /// `false` for a buffer that is not (or is a whole allocation, which
    /// frees when its last handle drops — its residency entry is the
    /// holder's to remove).
    fn reclaim(&self, buffer: &Buffer) -> bool;
    fn stats(&self) -> BackingStats;
    /// Enumerate ready free capacity already charged to this domain. Source
    /// identity must remain retained for the whole collection; delegating
    /// wrappers preserve identities so aliases are counted once. None means
    /// legacy/unadmitted capacity. This is advisory capacity, not fit proof.
    fn collect_reusable_allocations(&self, _domain: Option<&DomainKey>, _credit: &mut ReuseCredit) {
    }
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
    membership: Arc<super::membership::MembershipDomain>,
    // Cold immutable default native payer; None only for explicit standalone legacy.
    default_domain: Option<ModelDomain>,
    requires_context: bool,
}

type RawBuffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// A cold candidate not yet published into residency or planner state. The
/// admitted arm retains its uncommitted ticket, raw-first, through selection.
pub(crate) struct StagedDeviceBuffer<'a> {
    backing: &'a DeviceBuffer,
    native: StagedNativeBuffer,
    requested: usize,
}
#[expect(
    clippy::large_enum_variant,
    reason = "Keep raw-first speculative ticket custody inline; Box storage is absent from the admitted quote."
)]
enum StagedNativeBuffer {
    Legacy(Buffer),
    Accounted(allocation_admission::StagedAllocation<RawBuffer>),
}
impl StagedDeviceBuffer<'_> {
    pub(crate) fn commit(self) -> Result<Allocation, AllocationError> {
        let buffer = match self.native {
            StagedNativeBuffer::Legacy(buffer) => buffer,
            StagedNativeBuffer::Accounted(staged) => {
                let (raw, owner) = staged.commit()?;
                Buffer::new_owned(raw, owner)
            }
        };
        self.backing.publish(buffer, self.requested)
    }
}

impl DeviceBuffer {
    pub fn new(
        device: &Device,
        residency: Arc<ResidencySet>,
        options: MTLResourceOptions,
        commit: Commit,
        ledger: Option<Arc<Ledger>>,
    ) -> Self {
        let membership = super::membership::MembershipDomain::new(
            residency.clone(),
            crate::metal_backend::retain_evicted_allocations(),
        );
        Self {
            device: device.clone(),
            residency,
            options,
            commit,
            ledger,
            membership,
            default_domain: None,
            requires_context: false,
        }
    }
    pub(crate) fn with_membership(
        mut self,
        membership: Arc<super::membership::MembershipDomain>,
    ) -> Self {
        self.membership = membership;
        self
    }
    /// Engine construction only, before any allocation. This is actual native
    /// numerical custody, not the device metadata domain or machine admission.
    pub(crate) fn with_default_domain(mut self, domain: ModelDomain) -> Self {
        self.default_domain = Some(domain);
        self.requires_context = false;
        self
    }
    pub(crate) fn with_native_payer(mut self, domain: ModelDomain, requires_context: bool) -> Self {
        self.default_domain = Some(domain);
        self.requires_context = requires_context;
        self
    }
    fn validate_context(&self, context: &AllocationContext) -> Result<(), AllocationError> {
        if let Some(domain) = &self.default_domain {
            if domain.key() != context.domain_key() {
                return Err(AllocationError::DifferentDomain);
            }
        }
        context.check_open()
    }

    /// Same native synchronous copy for default and supplied cold uploads.
    /// Empty input remains the caller's pooled-minimum route.
    pub(crate) fn alloc_data<T>(
        &self,
        data: &[T],
        context: Option<&AllocationContext>,
    ) -> Result<Allocation, AllocationError> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Err(AllocationError::Geometry(
                "empty upload uses pooled minimum allocation",
            ));
        }
        let pointer = std::ptr::NonNull::new(data.as_ptr().cast_mut().cast::<std::ffi::c_void>())
            .expect("nonempty slice pointer");
        let create = |length| {
            debug_assert_eq!(length, bytes);
            // SAFETY: synchronous copy of exactly the borrowed slice; no padding read.
            unsafe {
                self.device.as_ref().newBufferWithBytes_length_options(
                    pointer,
                    length,
                    self.options,
                )
            }
        };
        match context {
            Some(context) => self.alloc_native_in(bytes, context, create),
            None if self.requires_context => Err(AllocationError::Unsupported(
                "model backing requires supplied allocation context",
            )),
            None if self.default_domain.is_some() => {
                self.stage_native_default(bytes, create)?.commit()
            }
            None => self.publish(
                Buffer::new(create(bytes).ok_or(AllocationError::DeviceOom { requested: bytes })?),
                bytes,
            ),
        }
    }

    /// Cold whole-upload physical handoff, using the same synchronous copy.
    /// # Safety
    /// Move directly into the same-domain independently rooted PhysicalPool
    /// before a raw/native/view alias escapes. DeviceBuffer has no range lease;
    /// actual native+IO/foreign/emission terminal carriers are still required.
    pub(crate) unsafe fn alloc_data_custodied<T>(
        &self,
        data: &[T],
        context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        self.alloc_data(data, context)
            .map(|allocation| CustodiedAllocation::from_parts(allocation, None))
    }

    /// The admitted upload preserves native newBufferWithBytes; only the
    /// requested input length reaches the native creator, never padded demand.
    pub(crate) fn alloc_data_in<T>(
        &self,
        data: &[T],
        context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        self.alloc_data(data, Some(context))
    }

    fn alloc_native_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
        create: impl FnOnce(usize) -> Option<RawBuffer>,
    ) -> Result<Allocation, AllocationError> {
        self.stage_native_in(bytes, context, create)?.commit()
    }

    /// Selection-capable cold allocation. No residency entry exists until commit.
    pub(crate) fn stage(
        &self,
        bytes: usize,
        context: Option<&AllocationContext>,
    ) -> Result<StagedDeviceBuffer<'_>, AllocationError> {
        match context {
            Some(context) => self.stage_native_in(bytes, context, |length| {
                self.device
                    .as_ref()
                    .newBufferWithLength_options(length, self.options)
            }),
            None if self.requires_context => Err(AllocationError::Unsupported(
                "model backing requires supplied allocation context",
            )),
            None if self.default_domain.is_some() => self.stage_native_default(bytes, |length| {
                self.device
                    .as_ref()
                    .newBufferWithLength_options(length, self.options)
            }),
            None => Ok(StagedDeviceBuffer {
                backing: self,
                native: StagedNativeBuffer::Legacy(
                    self.device.new_buffer(bytes.max(1), self.options)?,
                ),
                requested: bytes.max(1),
            }),
        }
    }

    fn stage_native_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
        create: impl FnOnce(usize) -> Option<RawBuffer>,
    ) -> Result<StagedDeviceBuffer<'_>, AllocationError> {
        self.validate_context(context)?;
        let prediction = self.prediction(bytes)?;
        let requested = prediction.requested;
        let native = allocation_admission::stage_owned(context, prediction, create, |raw| {
            raw.allocatedSize()
        })?;
        Ok(StagedDeviceBuffer {
            backing: self,
            native: StagedNativeBuffer::Accounted(native),
            requested,
        })
    }

    fn stage_native_default(
        &self,
        bytes: usize,
        create: impl FnOnce(usize) -> Option<RawBuffer>,
    ) -> Result<StagedDeviceBuffer<'_>, AllocationError> {
        let prediction = self.prediction(bytes)?;
        let requested = prediction.requested;
        let domain = self
            .default_domain
            .as_ref()
            .ok_or(AllocationError::Unsupported("default native payer absent"))?;
        // Cold-only: promise BEFORE native creation; no second supplied reserve.
        let quote = prediction.requested_owned_bytes()?;
        let pass = domain.reserve(quote)?;
        let native =
            allocation_admission::stage_owned(&pass.context(), prediction, create, |raw| {
                raw.allocatedSize()
            })?;
        // Ticket already issued; parent cancellation closes future issuance only.
        // Candidate retains raw before ticket through speculative selection.
        Ok(StagedDeviceBuffer {
            backing: self,
            native: StagedNativeBuffer::Accounted(native),
            requested,
        })
    }

    fn prediction(&self, bytes: usize) -> Result<Prediction, AllocationError> {
        let requested = bytes.max(1);
        if requested > self.device.as_ref().maxBufferLength() {
            return Err(AllocationError::Geometry(
                "request exceeds device maxBufferLength",
            ));
        }
        let estimate = self
            .device
            .as_ref()
            .heapBufferSizeAndAlignWithLength_options(requested, self.options);
        // A standalone buffer's `allocatedSize` is whole VM pages, past what
        // the heap estimate says (26880 B allocates 32768 on a 16 KiB page).
        let unit = estimate
            .align
            .max(allocation_admission::native_page_bytes());
        Prediction::standalone(requested, estimate.size, unit)
    }

    /// A selected candidate invalidated after commit. Remove its published
    /// residency ownership while its raw Buffer still exists, outside planner
    /// locks; then raw release precedes physical owner/legacy reservation Drop.
    pub(crate) fn discard_selected(&self, allocation: Allocation) {
        self.residency.remove(&allocation.buffer);
        drop(allocation);
    }

    fn publish(&self, buffer: Buffer, requested: usize) -> Result<Allocation, AllocationError> {
        // Mint custody before membership publication. Native submission leases
        // and alias eligibility remain required before family enablement.
        let token = self
            .membership
            .for_buffer(&buffer)
            .map_err(AllocationError::from)?;
        let buffer = buffer.with_membership(token);
        match self.commit {
            Commit::Deferred => self.residency.add(&buffer),
            Commit::Now => self.residency.insert(&buffer),
        };
        // Parallel legacy diagnostics; physical owner remains on every Buffer alias.
        let reservation = self.ledger.as_ref().map(|l| l.reserve(requested));
        Ok(Allocation {
            buffer,
            reservation,
        })
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
    unsafe fn alloc_custodied(
        &self,
        bytes: usize,
        context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        let allocation = self.stage(bytes, context)?.commit()?;
        // Whole allocation has no placement range. Native/charge/ledger
        // ownership still moves intact into physical C/A under caller contract.
        Ok(CustodiedAllocation::from_parts(allocation, None))
    }

    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        self.alloc_native_in(bytes, context, |length| {
            self.device
                .as_ref()
                .newBufferWithLength_options(length, self.options)
        })
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, AllocationError> {
        self.stage(bytes, None).and_then(StagedDeviceBuffer::commit)
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
    unsafe fn alloc_custodied(
        &self,
        bytes: usize,
        context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        // SAFETY: same complete physical/alias/host-custody obligation above.
        let placement = unsafe { HeapChunks::alloc_custodied(self, bytes, context)? };
        Ok(CustodiedAllocation::from_parts(
            Allocation {
                buffer: placement.buffer,
                reservation: None,
            },
            Some(BackingRangeCustody::Heap(placement.custody)),
        ))
    }

    fn collect_reusable_allocations(&self, domain: Option<&DomainKey>, credit: &mut ReuseCredit) {
        HeapChunks::collect_reusable_allocations(self, domain, credit)
    }
    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        Ok(Allocation {
            buffer: HeapChunks::alloc_in(self, bytes, context)?,
            reservation: None,
        })
    }
    /// The chunk holds the reservation; the sub-buffer carries none.
    fn alloc(&self, bytes: usize) -> Result<Allocation, AllocationError> {
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

impl Mapped {
    pub(crate) fn with_native_payer(mut self, domain: ModelDomain, requires_context: bool) -> Self {
        self.inner = self.inner.with_native_payer(domain, requires_context);
        self
    }
    pub(crate) fn with_membership(
        mut self,
        membership: Arc<super::membership::MembershipDomain>,
    ) -> Self {
        self.inner = self.inner.with_membership(membership);
        self
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
    unsafe fn alloc_custodied(
        &self,
        bytes: usize,
        context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        // SAFETY: preserve same complete physical custody through wrapper.
        let allocation = unsafe { self.inner.alloc_custodied(bytes, context)? };
        wire_pages(allocation.allocation.buffer.contents(), bytes);
        Ok(allocation)
    }

    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        let a = self.inner.alloc_in(bytes, context)?;
        wire_pages(unsafe { a.buffer.contents() }, bytes);
        Ok(a)
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, AllocationError> {
        let a = self.inner.alloc(bytes)?;
        wire_pages(unsafe { a.buffer.contents() }, bytes);
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
    unsafe fn alloc_custodied(
        &self,
        bytes: usize,
        context: Option<&AllocationContext>,
    ) -> Result<CustodiedAllocation, AllocationError> {
        // SAFETY: delegation preserves the exact returned custody object.
        match unsafe { self.first.alloc_custodied(bytes, context) } {
            Err(error) if error.permits_fallback() => unsafe {
                self.then.alloc_custodied(bytes, context)
            },
            result => result,
        }
    }

    fn collect_reusable_allocations(&self, domain: Option<&DomainKey>, credit: &mut ReuseCredit) {
        self.first.collect_reusable_allocations(domain, credit);
        self.then.collect_reusable_allocations(domain, credit);
    }
    fn alloc_in(
        &self,
        bytes: usize,
        context: &AllocationContext,
    ) -> Result<Allocation, AllocationError> {
        match self.first.alloc_in(bytes, context) {
            Err(error) if error.permits_fallback() => self.then.alloc_in(bytes, context),
            result => result,
        }
    }

    fn alloc(&self, bytes: usize) -> Result<Allocation, AllocationError> {
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

#[cfg(test)]
mod prediction_tests {
    use super::*;
    use objc2_metal::MTLResourceOptions as O;

    /// The prediction bounds what Metal actually allocates for a standalone
    /// buffer, at sizes on, under and over page and heap-alignment edges and
    /// at the sizes that were refused before pages were counted.
    #[test]
    fn the_prediction_bounds_the_native_allocated_size() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let residency = Arc::new(ResidencySet::new(&device));
        let mut checked = 0;
        for options in [O::StorageModeShared, O::StorageModePrivate] {
            let backing = DeviceBuffer::new(&device, residency.clone(), options, Commit::Now, None);
            for size in [
                1,
                100,
                4095,
                4096,
                4097,
                16383,
                16384,
                16385,
                26880,
                68608,
                512000,
                525312,
                (1 << 20) + 1,
                (7 << 20) + 123,
                (64 << 20) + 1,
            ] {
                let predicted = backing.prediction(size).unwrap().bytes;
                let raw = device
                    .as_ref()
                    .newBufferWithLength_options(size, options)
                    .expect("a standalone buffer");
                let actual = raw.allocatedSize();
                assert!(
                    actual <= predicted,
                    "{options:?} {size} B: Metal allocated {actual} > predicted {predicted}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 30);
    }

    /// The admitted default route, which refused those sizes, now serves them.
    #[test]
    fn the_admitted_default_route_serves_sizes_past_the_heap_estimate() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let residency = Arc::new(ResidencySet::new(&device));
        let domain = crate::memory_admission::process_account()
            .unadmitted_domain(None)
            .unwrap();
        let backing =
            DeviceBuffer::new(&device, residency, O::StorageModeShared, Commit::Now, None)
                .with_default_domain(domain);
        for size in [26880, 68608, 512000, 525312] {
            let allocation = backing
                .alloc(size)
                .unwrap_or_else(|e| panic!("{size} B: {e}"));
            assert!(allocation.buffer.length() >= size);
        }
    }
}
