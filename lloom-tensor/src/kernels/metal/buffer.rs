use super::physical_roles::{NativeUse, PhysicalStrong, UseWitnessError, PHYSICAL_SLAB_SLOTS};
use crate::allocation_owner::{AllocationLease, AllocationOwner};
use crate::memory_admission::AllocationError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource};
use std::{collections::HashMap, sync::Arc};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

enum BufferRole {
    // Actual raw/accounting/membership baseline. Never a future-use pin.
    Native {
        raw: AllocationLease<Retained<ProtocolObject<dyn MTLBuffer>>>,
        membership: Option<super::membership::Membership>,
    },
    // Private only in rich HeaderRoot C; its genuine strong borrow spans every
    // returned &Buffer. Clone/view NEVER returns another Primary role.
    Primary(PhysicalStrong<PHYSICAL_SLAB_SLOTS>),
    External(NativeUse<PHYSICAL_SLAB_SLOTS>),
}

pub struct Buffer {
    role: BufferRole,

    /// Byte offset this handle addresses within `raw`.
    ///
    /// Zero for every buffer candle allocates today, which is why this is not a
    /// behaviour change: an ordinary allocation owns its whole `MTLBuffer` and
    /// starts at 0. It is nonzero only for an activation-arena slot, which is a
    /// *region* of one shared allocation rather than an allocation of its own
    /// (`DESIGN.md` §9.2).
    ///
    /// # Why the offset lives here rather than at the call sites
    ///
    /// The arena binds every activation as `base + offset`. Threading that
    /// offset outward would mean touching the ~51 kernel entry points that take
    /// `output: &Buffer` and the 62 `buffer_o` call sites in the Metal backend
    /// -- and under `HazardTrackingModeUntracked` (§3.5) every one of those is
    /// an opportunity for a silent wrong-offset bind, which is precisely the
    /// failure class §9.3 says rests entirely on our offset arithmetic.
    ///
    /// Carrying it in the handle instead means the addition happens in exactly
    /// two places -- `set_input_buffer` and `set_output_buffer`, the choke point
    /// every binding already funnels through. That is the same argument §11.1b
    /// makes for putting the executor seam at dispatch submission rather than at
    /// the kernel entry points: one place to review, not fifty.
    base_offset: usize,

    /// Bytes this handle addresses, when it is a view.
    ///
    /// `None` means "all of `raw`", which is every ordinary allocation. A view
    /// carries its slot size so `length()` cannot report the whole arena.
    length: Option<usize>,

    /// Whether CPU access is permitted by storage mode and the selected
    /// experimental mapping policy, fixed at construction (see `new`).
    cpu_visible: bool,

    /// Whether, and why, the device's residency sets list this allocation
    /// from before it is handed out until its last GPU reader has completed.
    /// A Metal 4 slot set may leave such a binding to those sets
    /// (`kernels.residency.metal4`, `tuning.md` §6.3p).
    device_listed: DeviceListing,
}

/// Why the device's residency sets list an allocation until its last GPU
/// reader completes (`invariants.md` §15.1r), or that nothing says so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceListing {
    /// Not known to be: the Metal 4 slot set lists the binding itself.
    #[default]
    No,
    /// A pool's buffer (it leaves the sets only on eviction from the free
    /// list, after its epoch) or a heap chunk's sub-buffer (the chunk is
    /// listed for the heap's life).
    Pool,
    /// A buffer outside every pool whose removal from the sets, and whose
    /// membership record, wait in the in-flight hold for the epoch open at
    /// its last handle's drop (`kernels.residency.metal4_cover = held`).
    Held,
}

impl DeviceListing {
    /// The device's sets list it until its last GPU reader completes.
    #[inline]
    pub fn covered(self) -> bool {
        self != DeviceListing::No
    }
}

// SAFETY: Native preserves Buffer's existing retained-MTLBuffer thread-safety
// contract and adds only Send+Sync accounting/membership custody. Primary is private rich C only:
// a genuine HeaderStrong borrow pins the initialized payload while its methods
// read immutable physical baseline metadata or mint counted External ownership.
// External owns NativeUse with an atomic external census and a genuine counted
// physical HeaderStrong; last release cannot free raw/charge on this thread.
// Native reference access is explicitly unsafe and imposes host/GPU/IO ordering;
// sharing Buffer itself grants neither mutable bytes nor an untracked SDK retain.
unsafe impl Send for Buffer {}
// SAFETY: the same role-specific invariants hold for shared borrows. Primary
// and External lifetime state is atomic, and their physical payload destruction
// is exclusively service-owned; Native preserves existing native lease Drop.
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Legacy unadmitted native handle. External raw retains are not tracked.
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Self::with_owner(raw, None)
    }

    /// Additive source-only admitted constructor; callers must provide both
    /// preallocation admission and native/residency leases before enabling it.
    pub(crate) fn new_owned(
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        owner: AllocationOwner,
    ) -> Buffer {
        Self::with_owner(raw, Some(owner))
    }

    pub(super) fn is_native_baseline(&self) -> bool {
        matches!(self.role, BufferRole::Native { .. })
    }
    pub(crate) fn allocation_owner(&self) -> Option<&AllocationOwner> {
        self.raw_lease().owner()
    }

    /// Owned native custody. On a rich admitted Primary/External this mints a
    /// counted physical reuse pin; its actual generation/IO/derived owner must
    /// retain it through terminality AND all future emission/raw-reference
    /// closure. SDK retain alone is insufficient.
    /// On legacy Native this preserves raw/accounting/membership ownership but
    /// does NOT enroll the legacy pool's reuse census. Raw interoperability must
    /// separately keep the creator's rich handle/reuse authority live or prove
    /// that the legacy allocation cannot be reused throughout every future use.
    pub fn native_lease(&self) -> Result<NativeBufferLease, AllocationError> {
        Ok(NativeBufferLease {
            buffer: self.try_clone()?,
        })
    }

    /// Cold, engine-private native/accounting baseline custody only. Refuse
    /// Primary/External before any retain: a future-use pin must never become
    /// a tokenless baseline (or a namespace/root -> allocation -> root cycle).
    /// This preserves the original AllocationLease native-before-charge Drop
    /// order. It neither carries membership nor grants encoding/reuse authority;
    /// the caller retains membership separately and final release stays in
    /// independently rooted host cleanup. Never use as a generation/IO pin or
    /// as proof that the originating legacy pool cannot reuse the allocation.
    pub(crate) fn native_baseline_lease(
        &self,
    ) -> Result<AllocationLease<Retained<ProtocolObject<dyn MTLBuffer>>>, AllocationError> {
        match &self.role {
            BufferRole::Native { raw, .. } => Ok(raw.clone()),
            BufferRole::Primary(_) | BufferRole::External(_) => {
                Err(AllocationError::InvalidCustody)
            }
        }
    }

    /// Genuine counted binding mint. Refuse legacy Native BEFORE any native
    /// retain/member clone, so a refused legacy lease cannot be destroyed under
    /// a packed capture/encoder guard. Existing native_lease remains available
    /// for explicitly unsafe legacy interoperability with creator custody.
    pub(crate) fn counted_native_lease(&self) -> Result<NativeBufferLease, AllocationError> {
        match &self.role {
            BufferRole::Native { .. } => Err(AllocationError::InvalidCustody),
            BufferRole::Primary(_) | BufferRole::External(_) => self.native_lease(),
        }
    }

    // Tokenless baseline read is private. Public raw interoperability is unsafe.
    fn raw_lease(&self) -> &AllocationLease<Retained<ProtocolObject<dyn MTLBuffer>>> {
        match &self.role {
            BufferRole::Native { raw, .. } => raw,
            // SAFETY: cold physical baseline contains only Native, never this
            // Primary/External role or a rich HeaderRoot/namespace backedge.
            BufferRole::Primary(physical) => unsafe { physical.baseline_unchecked() }.raw_lease(),
            BufferRole::External(pin) => unsafe { pin.native() }.0.raw_lease(),
        }
    }
    fn native_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.raw_lease().raw()
    }
    fn membership(&self) -> Option<&super::membership::Membership> {
        match &self.role {
            BufferRole::Native { membership, .. } => membership.as_ref(),
            BufferRole::Primary(physical) => unsafe { physical.baseline_unchecked() }.membership(),
            BufferRole::External(pin) => unsafe { pin.native() }.0.membership(),
        }
    }
    /// Private rich C constructor only; cannot be returned independently from
    /// its actual HeaderStrong. Existing producer/header Drop is atoms only.
    /// # Safety
    /// Move only into this allocation's unpublished rich C, then expose solely
    /// through genuine HeaderStrong borrows after publish-last. Never move this
    /// role out of C or mint an alias during service cleanup/after DETACHING.
    pub(super) unsafe fn primary(physical: PhysicalStrong<PHYSICAL_SLAB_SLOTS>) -> Self {
        let baseline = unsafe { physical.baseline_unchecked() };
        let cpu_visible = baseline.cpu_visible;
        let device_listed = baseline.device_listed;
        Self {
            role: BufferRole::Primary(physical),
            base_offset: 0,
            length: None,
            cpu_visible,
            device_listed,
        }
    }
    /// # Safety
    /// Trusted rich C cleanup only; caller holds actual DETACHING authority or
    /// never-published initialization custody. Cannot mint/export native use.
    pub(super) unsafe fn physical_primary_unchecked(&self) -> &PhysicalStrong<64> {
        match &self.role {
            BufferRole::Primary(physical) => physical,
            _ => unreachable!("private rich C contains only Primary"),
        }
    }
    pub fn try_clone(&self) -> Result<Self, AllocationError> {
        let role = match &self.role {
            BufferRole::Native { raw, membership } => BufferRole::Native {
                raw: raw.clone(),
                membership: membership.clone(),
            },
            // This role is reachable only through a genuine rich strong borrow;
            // no service/descriptor/raw baseline can publicly mint a Primary.
            BufferRole::Primary(physical) => BufferRole::External(
                unsafe { NativeUse::from_buffer_primary(physical) }
                    .map_err(AllocationError::from)?,
            ),
            BufferRole::External(pin) => {
                BufferRole::External(pin.try_clone().map_err(AllocationError::from)?)
            }
        };
        Ok(Self {
            role,
            base_offset: self.base_offset,
            length: self.length,
            cpu_visible: self.cpu_visible,
            device_listed: self.device_listed,
        })
    }
    /// # Safety
    /// Every SDK retain, texture/view/IO derivation and future encoding access
    /// must retain native_lease() through actual terminality AND emission/raw
    /// reference closure. For legacy Native the lease is raw/charge/member
    /// custody only: additionally prevent creator-pool reuse (e.g. retain its
    /// BufferHandle) or prove this allocation is never pooled. A retained SDK
    /// object alone does not preserve Rust charge or admitted reuse census.
    /// No raw reference/derived capability may overlap physical reuse or account
    /// teardown. Shared residency-set-derived native handles obey the same rule.
    pub unsafe fn raw_unchecked(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.native_ref()
    }

    fn with_owner(
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        owner: Option<AllocationOwner>,
    ) -> Buffer {
        // Private storage has no documented CPU mapping. The experimental
        // opt-in retains the owned-buffer probe, never the heap-offset value
        // (which can be non-null without being a CPU address).
        let cpu_visible = raw.storageMode() != objc2_metal::MTLStorageMode::Private
            || (crate::tuning::tuning().metal.private_cpu_mapping && raw.heap().is_none());
        super::arena::note_new_buffer(Retained::as_ptr(&raw) as usize);
        Buffer {
            role: BufferRole::Native {
                raw: AllocationLease::new(raw, owner),
                membership: None,
            },
            base_offset: 0,
            length: None,
            cpu_visible,
            device_listed: DeviceListing::No,
        }
    }

    /// Marks this allocation as listed by the device's residency sets until
    /// its last GPU reader completes, for `why` ([`DeviceListing`]): the
    /// device's pool adoption of a fresh allocation from its own backings,
    /// or an unpooled buffer whose last drop goes to the in-flight hold.
    pub(crate) fn with_device_listing(mut self, why: DeviceListing) -> Self {
        self.device_listed = why;
        self
    }

    /// Whether a membership record lists this allocation in the device's
    /// sets: every allocation a device backing published
    /// (`DeviceBuffer::publish`), and a heap sub-buffer through its chunk's.
    #[inline]
    pub(crate) fn listed_by_record(&self) -> bool {
        self.membership().is_some()
    }

    /// Why the device's sets list this allocation, if they do.
    #[inline]
    pub(crate) fn device_listing(&self) -> DeviceListing {
        self.device_listed
    }

    pub(crate) fn with_membership(mut self, membership: super::membership::Membership) -> Self {
        let BufferRole::Native {
            membership: member, ..
        } = &mut self.role
        else {
            panic!("membership must precede physical/rich enrollment");
        };
        assert!(member.is_none(), "membership identity is minted once");
        *member = Some(membership);
        self
    }

    /// A handle addressing `base_offset..base_offset + length` of this buffer.
    ///
    /// Shares the underlying `MTLBuffer` -- this is a view, not a copy, and
    /// nothing is allocated. Used by the activation arena to hand out slots.
    ///
    /// The shared `MTLBuffer` identity is deliberate on two counts. It is what
    /// makes buffer identity *stable* at a dispatch position (§11.1a.1's 674),
    /// since every slot presents the same allocation. And it makes the
    /// encoder's hazard tracking, which keys on the raw pointer, treat the whole
    /// arena as one resource -- so a read-after-write on any slot orders against
    /// a write to any other. That is conservative rather than unsound: Metal
    /// barriers are resource-granular anyway (§3.5), so a finer key could not
    /// have expressed a narrower edge.
    ///
    /// # Panics
    /// Invalid or overflowing ranges are rejected in every build before native
    /// handle cloning. The return type is preserved; this is not quota refusal.
    pub fn view(&self, base_offset: usize, length: usize) -> Buffer {
        let absolute = crate::allocation_geometry::checked_view_offset(
            self.base_offset,
            self.length(),
            base_offset,
            length,
        )
        .expect("invalid Buffer::view range");
        let role = match &self.role {
            BufferRole::Native { raw, membership } => BufferRole::Native {
                raw: raw.clone(),
                membership: membership.clone(),
            },
            BufferRole::Primary(physical) => {
                let pin = unsafe { NativeUse::from_buffer_primary(physical) }
                    .expect("valid primary count");
                BufferRole::External(
                    pin.into_view(base_offset, length)
                        .expect("checked primary view range"),
                )
            }
            BufferRole::External(pin) => {
                BufferRole::External(pin.view(base_offset, length).expect("checked view count"))
            }
        };
        Buffer {
            role,
            base_offset: absolute,
            length: Some(length),
            cpu_visible: self.cpu_visible,
            device_listed: self.device_listed,
        }
    }

    /// Byte offset this handle addresses within the underlying `MTLBuffer`.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Whether this handle addresses a **sub-range** of an `MTLBuffer` it does
    /// not own, rather than a whole allocation.
    ///
    /// # Why the pool asks (lloom issue #507)
    ///
    /// `BufferPool::acquire` keys on **size alone** and knows nothing of
    /// provenance -- it hands back whatever is in the bucket. That is correct
    /// for allocations, which own their whole `MTLBuffer` and are
    /// interchangeable at a given size, and it is silent corruption for views,
    /// which are N handles onto **one** allocation: measured on the shipping
    /// arena, five "separate" activation buffers at one base address and five
    /// offsets, under `HazardTrackingModeUntracked` where §9.3 says there is no
    /// safety net (§9.2m, issue #489).
    ///
    /// So the pool's invariant is that a view never enters it, and `adopt` is
    /// where that is checked -- see [`super::BufferPool::adopt`] for why there
    /// rather than at `acquire`.
    ///
    /// **`base_offset != 0` is NOT the test**, which is why this exists as its
    /// own predicate: an arena's first slot is a view at offset 0 (the measured
    /// trace has one), so an offset check would pass exactly the aliasing handle
    /// that shares its address with the base. What makes a handle a view is that
    /// it carries its own extent.
    pub fn is_view(&self) -> bool {
        self.length.is_some()
    }

    /// Whether querying a CPU pointer is permitted by the selected policy.
    /// An experimental Private probe can still return null.
    pub fn may_be_cpu_mapped(&self) -> bool {
        self.cpu_visible
    }

    /// # Safety
    /// As data(): retain ownership-tracked custody and prove host/GPU/IO
    /// ordering through every use of this unbounded raw pointer.
    pub unsafe fn contents(&self) -> *mut u8 {
        unsafe { self.data() }
    }

    /// CPU pointer to the first byte *this handle addresses*.
    ///
    /// Offset by `base_offset`, so a view reads and writes its own region
    /// rather than the parent's start. Keeping the whole `Buffer` API
    /// consistent about what "this buffer" means is what stops a view from
    /// being a silent-corruption hazard: `read_to_vec` and every other CPU-side
    /// consumer takes this pointer without knowing whether it holds a view.
    ///
    /// Null for Private storage by default, without invoking Metal's CPU
    /// getter. With `private_cpu_mapping` enabled, non-heap Private buffers
    /// retain the implementation-dependent mapping probe; heap placements
    /// remain inaccessible. Shared storage retains the direct-access path.
    ///
    /// # This function's own null check folded away (#366, repaired #454)
    ///
    /// §11.5d records the early return below being eliminated in the release
    /// binary: `base_offset` is 0 for every non-arena allocation, so the
    /// optimiser proves `null + 0 == null`, concludes the branch is redundant
    /// and drops it -- and with it the only thing standing between a null
    /// mapping and the `memmove` in a caller. The two production callers in
    /// `metal_backend` were repaired with `black_box` by #366 and #445; this
    /// one is the function *they call*, and it was not.
    ///
    /// Measured for #454 (`measurements/issue-454-raw/guard-probe/`, with the
    /// disassembly in `measurements/issue-454-raw/shape-disassembly.txt`): in
    /// an unguarded function the pointer `objc_msgSend` returns reaches `ldrb`
    /// with **no `cbz` on it at all**, at `opt-level=3` with LTO off as well as
    /// on; with `black_box` the same function stores it, reloads it and
    /// branches on `cbz`. Guarding here means every consumer of `data()`
    /// inherits a pointer whose nullity was actually tested, rather than each
    /// having to defend itself.
    /// # Safety
    /// Keep native_lease() through every pointer use and exclude racing host/
    /// GPU/IO accesses. Engine-only exclusive service cleanup authority after
    /// actual alias/native closure may instead borrow its retained baseline;
    /// this exception licenses no future encoding/derivation or exported pointer.
    /// Respect this view's length and retain no pointer past custody/reuse.
    pub unsafe fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        // `black_box` is load-bearing, not a micro-optimisation guard: it makes
        // the pointer opaque so the `is_null` below cannot be folded against
        // what the optimiser can prove about `contents()`.
        if !self.cpu_visible {
            return std::ptr::null_mut();
        }
        let base = std::hint::black_box(self.native_ref().contents().as_ptr()) as *mut u8;
        if base.is_null() {
            return base;
        }
        // SAFETY: `base_offset` is within the parent allocation by construction
        // -- public `view` checks relative and absolute ranges in every build
        // before constructing metadata; nested views preserve that invariant.
        unsafe { base.add(self.base_offset) }
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(super) fn managed_late_bind(&self) -> Option<&super::late_bind::ManagedRelease> {
        match &self.role {
            BufferRole::Native { .. } => None,
            BufferRole::Primary(physical) => physical.managed_late_bind(),
            BufferRole::External(pin) => pin.managed_late_bind(),
        }
    }
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(self.raw_lease().raw())
    }

    /// Address of the underlying `MTLBuffer`, as the encoder sees it.
    ///
    /// The same value the hazard tracking and the dispatch trace key on, so an
    /// arena recording can attribute a bind back to the allocation that produced
    /// the buffer. It identifies the *buffer*, which is deliberately not the
    /// same thing as the value occupying it (§9.2c).
    pub fn raw_addr(&self) -> usize {
        self.raw_ptr() as usize
    }

    /// Bytes this handle addresses.
    ///
    /// A view reports its slot's extent, not the arena's. `fill_buffer` and
    /// `copy_from_buffer` both take `buffer.length()` as a range, so reporting
    /// the parent's length would let a blit over one slot run across every slot
    /// after it -- silent corruption of the exact kind §9.3 says rests on our
    /// offset arithmetic.
    pub fn length(&self) -> usize {
        match self.length {
            Some(len) => len,
            None => match &self.role {
                BufferRole::Native { raw, .. } => raw.raw().length(),
                BufferRole::Primary(physical) => physical.bytes(),
                BufferRole::External(pin) => pin.length(),
            },
        }
    }

    /// Immutable allocation extent, independent of a logical view. Does not
    /// return SDK ownership or authorize encoding, memory access, or reuse.
    pub(crate) fn native_allocation_length(&self) -> usize {
        self.native_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.native_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.native_ref().setLabel(Some(&NSString::from_str(label)))
    }
}

// Ownership metadata is not buffer identity. Preserve the pre-token fields.
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.raw_lease() == other.raw_lease()
            && self.base_offset == other.base_offset
            && self.length == other.length
            && self.cpu_visible == other.cpu_visible
    }
}
impl std::hash::Hash for Buffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.raw_lease(), state);
        std::hash::Hash::hash(&self.base_offset, state);
        std::hash::Hash::hash(&self.length, state);
        std::hash::Hash::hash(&self.cpu_visible, state);
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        // Existing infallible trait stays source-compatible. Native preserves
        // exact native/accounting/member Clone. Counted roles use checked atomics;
        // refusal leaves the original valid and no partial count increment.
        // This panic is count/invariant-only, NEVER quota/capacity/allocation refusal.
        // New carrier callers use fallible try_clone/native_lease instead.
        self.try_clone()
            .expect("Buffer::clone counted custody invariant or count refusal")
    }
}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("raw", self.raw_lease())
            .field("base_offset", &self.base_offset)
            .field("length", &self.length)
            .field("cpu_visible", &self.cpu_visible)
            .finish_non_exhaustive()
    }
}
/// Owned custody with no safe Deref/AsRef/Into native conversion. Rich admitted
/// leases pin physical reuse; legacy leases retain raw/accounting/member owners
/// but do not close legacy pool reuse. The raw boundary carries that additional
/// legacy creator-custody obligation. No terminality is inferred merely from
/// this wrapper existing or its underlying SDK object's retained count.
pub struct NativeBufferLease {
    buffer: Buffer,
}
impl NativeBufferLease {
    /// Borrow the already-owned counted External Buffer for existing hazard/
    /// Executor binding hooks. The lease spans the borrow; cloning/viewing
    /// this Buffer preserves counted External custody. No Primary baseline,
    /// raw pointer, independent producer root or new native retain is exposed.
    /// Legacy raw leases refuse here: unsafe legacy interop must separately
    /// retain its verified original creator/reuse authority.
    pub(crate) fn buffer_for_binding(&self) -> Result<&Buffer, UseWitnessError> {
        match &self.buffer.role {
            BufferRole::External(_) => Ok(&self.buffer),
            BufferRole::Native { .. } => Err(UseWitnessError::UntrackedLegacy),
            BufferRole::Primary(_) => Err(UseWitnessError::InvalidCustody),
        }
    }
    /// Exact native generation host-cleanup observation, not an open-clock sample.
    /// # Safety
    /// Same terminal/closed-generation contract as NativeUse::observe_submitted.
    /// Never observe a merely staged epoch on a known no-handoff cancellation.
    /// On error retain the initialized owning slot/root; no log-and-Drop path.
    pub(crate) unsafe fn observe_submitted(
        &self,
        clock: &Arc<super::buffer_pool::GpuClock>,
        epoch: crate::retire::Epoch,
    ) -> Result<(), UseWitnessError> {
        match &self.buffer.role {
            BufferRole::External(pin) => unsafe { pin.observe_submitted(clock, epoch) },
            BufferRole::Native { .. } => Err(UseWitnessError::UntrackedLegacy),
            BufferRole::Primary(_) => Err(UseWitnessError::InvalidCustody),
        }
    }
    pub fn try_clone(&self) -> Result<Self, AllocationError> {
        self.buffer.native_lease()
    }
    pub(crate) fn is_counted_physical_use(&self) -> bool {
        matches!(self.buffer.role, BufferRole::External(_))
    }
    /// # Safety
    /// As Buffer::raw_unchecked; this lease outlives every native/future use.
    pub unsafe fn raw_unchecked(&self) -> &ProtocolObject<dyn MTLBuffer> {
        unsafe { self.buffer.raw_unchecked() }
    }
    /// Native object identity while both owned leases remain live; not physical
    /// generation/namespace identity and not independent terminality proof.
    pub fn same_allocation(&self, other: &Self) -> bool {
        self.buffer.raw_ptr() == other.buffer.raw_ptr()
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;

#[cfg(test)]
mod issue_454_guard_shape {
    //! The null guards on `contents()` are load-bearing and were eliminated
    //! twice (#366, #445). This pins their SHAPE in source.
    //!
    //! It cannot be a runtime test: the defect is that a branch is absent from
    //! the emitted code, and a test that runs is executing the build it is
    //! trying to indict. #366 and #445 both verified by reading the
    //! disassembly, and #454's own probe does the same
    //! (`measurements/issue-454-raw/shape-disassembly.txt`). What a test CAN
    //! do cheaply is fail when a site loses its `black_box` in a refactor,
    //! which is how both regressions arrived -- a correct-looking source diff.
    //!
    //! Scoped to the sites where a null is REACHABLE, i.e. buffers that can be
    //! Private storage. Shared-storage buffers (`RESOURCE_OPTIONS`,
    //! `CURSOR_OPTIONS`) are CPU-mapped at every size, so their guards are
    //! unreachable and are deliberately not required to carry one.

    /// Every load-bearing `contents()` site keeps its optimisation barrier.
    ///
    /// Mutation-tested: removing the `black_box` from `Buffer::data` makes this
    /// fail. A first version sliced the function body loosely and passed under
    /// that mutation, because `black_box` also appears in the doc comment above
    /// it -- §15.1 #1's vacuous-comparison class, in the guard written to
    /// prevent a regression. It matches the STATEMENT now, not the region.
    #[test]
    fn load_bearing_contents_sites_keep_their_black_box() {
        let src = include_str!("buffer.rs");

        // Only the production half. This module mentions `.contents()` in its
        // own filter literal and in its messages, and a check that reads
        // itself reports on the wrong file -- which is how the first version
        // failed its own clean arm.
        let production = src
            .split("mod issue_454_guard_shape")
            .next()
            .expect("split always yields one element");

        // The line that actually reads the pointer, matched exactly.
        let reads: Vec<&str> = production
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && l.contains(".contents()"))
            .collect();

        assert!(
            !reads.is_empty(),
            "no `contents()` call found in buffer.rs -- was `data` renamed? \
             A guard that finds nothing to check passes for the wrong reason."
        );

        for line in reads {
            assert!(
                line.contains("black_box"),
                "Buffer::data's null test lost its `black_box`:\n    {line}\n\
                 Without it the optimiser proves the pointer non-null and drops \
                 the branch -- DESIGN.md §11.5d, §14.5c, and #454's own \
                 disassembly (`measurements/issue-454-raw/shape-disassembly.txt`). \
                 `data()` is what every CPU-side consumer reaches, so this is the \
                 one site whose guard covers the others. If removing it is \
                 deliberate, re-read the disassembly rather than the source."
            );
        }
    }
}
