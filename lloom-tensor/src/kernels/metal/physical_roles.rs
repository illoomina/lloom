//! Genuine allocation-local admitted-role adapter; accompanying Buffer changes
//! remove safe raw coercions and migrate raw pointer access to unsafe boundaries.
//! Not native-enrolled: cold caller and actual submission leases remain required.
use super::{
    backing::{Allocation, CustodiedAllocation},
    buffer_pool::GpuClock,
    ledger::Reservation,
    Buffer,
};
use crate::{
    allocation_alias::AliasPins,
    allocation_header::HeaderError,
    allocation_header_root::{
        HeaderPool, HeaderStrong, PreparedHeader, RootError, RootRejected, ValidatedEmptyHeader,
    },
    allocation_physical_ready::{
        PhysicalReady, PreparedUse, ReadyError, ReadyReservation, UseGeneration,
    },
    memory_admission::{AllocationContext, DomainKey},
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};

pub(crate) const PHYSICAL_SLAB_SLOTS: usize = 64;

struct PhysicalRecord {
    // Tokenless FUTURE-USE role: Buffer still retains raw/accounting/membership
    // baseline, but contains no PhysicalStrong/alias pin/namespace/hold edge.
    baseline: Buffer,
    bytes: usize, // Immutable cold native query; no producer ObjC getter.
    clock: Arc<GpuClock>,
    aliases: AliasPins<()>,
    ready: PhysicalReady,
    // Exact closed submitted creator epochs, encoded epoch+1 (zero=None).
    // Historical max is intentionally not reset during reuse/rollback: older
    // completed uses add no wait and cannot erase a delayed genuine witness.
    // Never sample the creator clock's current/open epoch here.
    submitted: AtomicU64,
    // Optional cold registered diagnostic state, independent of native/reuse
    // authority; Stage slots install it before any primary/use role escapes.
    managed_late_bind: OnceLock<super::late_bind::ManagedRegistration>,
}
impl Drop for PhysicalRecord {
    fn drop(&mut self) {
        // SAFETY: actual PhysicalHeader service ready predicate requires FREE
        // and closed external census. Physical A retains raw/member/charge and
        // ledger through this arbitrary diagnostic cleanup and C panic.
        unsafe { super::poison::unpoison_buffer(&self.baseline) };
        if let Some(state) = self.managed_late_bind.get() {
            super::late_bind::unregister_managed(&self.baseline, state);
        } else {
            super::late_bind::on_leave(&self.baseline);
        }
    }
}
struct PhysicalAnchors {
    // Separate raw/accounting/member baseline across PhysicalRecord C Drop.
    // Current Buffer clones no PhysicalStrong/future pin; never expose this A.
    baseline: Buffer,
    // Actual Backing::Allocation parallel ledger custody persists through free
    // and reused generations; not discarded at cold physical enrollment.
    reservation: Option<Reservation>,
    // Actual placement eligibility outlives both C and A native retains.
    // Sole non-Clone receipt; host service returns range only after all tracked
    // aliases/emission owners are closed and final physical storage dies.
    placement: Option<super::backing::BackingRangeCustody>,
}
pub(crate) struct PhysicalPool<const N: usize> {
    headers: HeaderPool<PhysicalRecord, PhysicalAnchors, N>,
    domain: DomainKey, // Cold affinity only, no warm domain clone/query.
}
/// Private pool descriptor/native accounting baseline, NOT a future-submit pin.
/// No Weak, root Arc, native borrow, or generic public Clone is exported.
pub(crate) struct PhysicalStrong<const N: usize> {
    header: HeaderStrong<PhysicalRecord, PhysicalAnchors, N>,
}
#[derive(Debug)]
pub(crate) enum PhysicalError {
    Header(HeaderError),
    Ready(ReadyError),
    ViewRange,
}
pub(crate) struct PhysicalClaim<'a, const N: usize> {
    backing: &'a PhysicalStrong<N>,
    claim: ReadyReservation<'a>,
}
/// All fallible counted clones occur before actual native sidecar construction.
pub(crate) struct PhysicalPreparation<'a, const N: usize> {
    primary: PhysicalStrong<N>,
    anchor: PhysicalStrong<N>,
    claim: ReadyReservation<'a>,
}
pub(crate) struct RichCaptures<S, const N: usize> {
    // Actual PooledBuffer poison/residency/hold sidecars must join here; arbitrary
    // cleanup first. A independently retains physical custody on C panic.
    sidecars: S,
    physical: PhysicalStrong<N>,
}
pub(crate) struct RecycleAnchor<const N: usize> {
    physical: PhysicalStrong<N>,
    receipt: Option<PreparedUse>,
    generation: UseGeneration,
}
struct ExternalPin<const N: usize> {
    physical: PhysicalStrong<N>,
}
/// Owned native/view/IO/submission role. Native references are borrowed from
/// cold physical custody, never retained/released on this producer path.
/// The terminal owner must end every usable borrow/handoff before pin Drop.
pub(crate) struct NativeUse<const N: usize> {
    offset: usize,
    length: usize,
    pin: ExternalPin<N>,
}
#[derive(Debug)]
pub(crate) enum PhysicalColdError {
    View,
    Unadmitted,
    UseRole,
    DifferentDomain,
    Header(RootError),
}
/// Cold-only rejected custody; None denotes host-rooted quarantine, not loss.
pub(crate) struct PhysicalColdRejected {
    pub(crate) error: PhysicalColdError,
    pub(crate) owners: Option<(CustodiedAllocation, Arc<GpuClock>)>,
}
impl<const N: usize> PhysicalPool<N> {
    /// # Safety
    /// Caller enrolls actual native allocation/domain/clock identity cold, before
    /// any raw usable alias escapes. No native view is a standalone backing.
    /// Every future submit-capable clone/view/IO/native owner uses NativeUse and
    /// retains it through actual terminality, including foreign queues. Legacy
    /// AsRef/raw/SDK-derived escapes are NOT closed by this draft. Physical C/A
    /// contain allocation-local custody only, no pool/namespace/hold/Core edges.
    pub(crate) unsafe fn cold_in(context: &AllocationContext) -> Result<Self, RootError> {
        // SAFETY: above exact baseline/alias contract; only FREE and no external
        // issuer permits final physical C/A destruction. No producer callback.
        let domain = context.domain_key(); // Cold only, before root construction.
        Ok(Self {
            domain,
            headers: unsafe {
                HeaderPool::<PhysicalRecord, PhysicalAnchors, N>::cold_owned_when_in(
                    context,
                    |c, _| c.ready.is_free() && !c.aliases.has_external(),
                    None,
                )?
            },
        })
    }
    /// Cold native allocation must already have its actual accounting lease and
    /// same-domain/clock provenance; future alias closure is constructor-unsafe.
    pub(crate) unsafe fn insert_cold(
        &self,
        allocation: CustodiedAllocation,
        clock: Arc<GpuClock>,
    ) -> Result<PhysicalStrong<N>, PhysicalColdRejected> {
        let (allocation, placement) = allocation.into_parts();
        let buffer = &allocation.buffer;
        let error = if !buffer.is_native_baseline() {
            Some(PhysicalColdError::UseRole)
        } else if buffer.is_view() {
            Some(PhysicalColdError::View)
        } else {
            match buffer.allocation_owner() {
                None => Some(PhysicalColdError::Unadmitted),
                Some(owner) if owner.domain_key() != self.domain => {
                    Some(PhysicalColdError::DifferentDomain)
                }
                Some(_) => None,
            }
        };
        if let Some(error) = error {
            return Err(PhysicalColdRejected {
                error,
                owners: Some((
                    CustodiedAllocation::from_parts(allocation, placement),
                    clock,
                )),
            });
        }
        let bytes = buffer.length(); // Native query at cold enrollment only.
        let Allocation {
            buffer,
            reservation,
        } = allocation;
        let anchors = PhysicalAnchors {
            baseline: buffer.clone(),
            reservation,
            placement,
        };
        let aliases = AliasPins::new();
        assert!(aliases.claim_primary()); // Convert birth external1 to baseline0.
        self.headers
            .insert(
                PhysicalRecord {
                    baseline: buffer,
                    bytes,
                    clock,
                    aliases,
                    ready: PhysicalReady::new(),
                    submitted: AtomicU64::new(0),
                    managed_late_bind: OnceLock::new(),
                },
                anchors,
            )
            .map(|header| PhysicalStrong { header })
            .map_err(|rejected| PhysicalColdRejected {
                error: PhysicalColdError::Header(rejected.error),
                // Rejection is cold: separate A native retain may drop here,
                // while returned C baseline still owns raw/member/accounting.
                owners: rejected.owners.map(|(c, a)| {
                    // Cold rejected C has no physical use/poison; preserve
                    // its complete returned ownership instead of running Drop.
                    let c = std::mem::ManuallyDrop::new(c);
                    // SAFETY: rejected C is exclusively owned, no slot retained
                    // it. Read each owning field once; AliasPins/Ready have no Drop.
                    let buffer = unsafe { std::ptr::read(&c.baseline) };
                    let clock = unsafe { std::ptr::read(&c.clock) };
                    (
                        CustodiedAllocation::from_parts(
                            Allocation {
                                buffer,
                                reservation: a.reservation,
                            },
                            a.placement,
                        ),
                        clock,
                    )
                }),
            })
    }
}
impl<const N: usize> PhysicalStrong<N> {
    /// # Safety
    /// Borrow only for trusted native encoding/cleanup; the baseline must never
    /// escape as a clone/view/raw capability without its counted NativeUse.
    pub(super) unsafe fn baseline_unchecked(&self) -> &Buffer {
        &self.record().baseline
    }
    pub(super) fn bytes(&self) -> usize {
        self.record().bytes
    }
    /// # Safety is supplied by the private PooledBuffer rich service Drop.
    pub(super) fn cleanup_epoch(
        &self,
        generation: UseGeneration,
    ) -> Result<Option<crate::retire::Epoch>, ReadyError> {
        // The only caller owns actual same-root DETACHING/initialized cleanup.
        unsafe { self.record().ready.cleanup_cutoff(generation) }
            .map(|epoch| epoch.map(crate::retire::Epoch))
    }

    pub(super) fn managed_late_bind(&self) -> Option<&super::late_bind::ManagedRelease> {
        self.record()
            .managed_late_bind
            .get()
            .map(super::late_bind::ManagedRegistration::state)
    }
    fn record(&self) -> &PhysicalRecord {
        self.header.captures()
    }
    fn retained_baseline(&self) -> Result<Self, HeaderError> {
        self.header.try_clone().map(|header| Self { header })
    }
    pub(crate) fn reserve(&self) -> Result<PhysicalClaim<'_, N>, ReadyError> {
        // FREE has no primary issuer. Any prior external issuer must own a pin,
        // so zero closes it; descriptor/baseline cannot mint external aliases.
        if self.record().aliases.has_external() {
            return Err(ReadyError::Busy);
        }
        self.record().ready.reserve().map(|claim| PhysicalClaim {
            backing: self,
            claim,
        })
    }
}
impl<'a, const N: usize> PhysicalClaim<'a, N> {
    pub(crate) fn prepare(self) -> Result<PhysicalPreparation<'a, N>, PhysicalError> {
        let primary = self
            .backing
            .retained_baseline()
            .map_err(PhysicalError::Header)?;
        let anchor = self
            .backing
            .retained_baseline()
            .map_err(PhysicalError::Header)?;
        Ok(PhysicalPreparation {
            primary,
            anchor,
            claim: self.claim,
        })
    }
}
impl<'a, const N: usize> PhysicalPreparation<'a, N> {
    /// Sidecars contain no future-use pin for this same backing and cannot
    /// export its baseline. Actual poison/residency/hold adapter remains pending.
    /// Root-bound original occupancy receipt moves only after all fallible
    /// admission/root clones and pool guarded revalidation. No callback/alloc.
    /// Vacant-cell validation preceded C/A construction. The consuming write
    /// has no error/owner-return path; initialized unwind is existing rooted
    /// header cleanup custody, never producer C/A destruction or early-FREE.
    pub(crate) fn initialize<S: Send + Sync + 'static, const H: usize>(
        self,
        empty: ValidatedEmptyHeader<'a, RichCaptures<S, N>, RecycleAnchor<N>, H>,
        sidecars: S,
    ) -> PreparedHeader<'a, RichCaptures<S, N>, RecycleAnchor<N>, H> {
        let generation = self.claim.next_generation();
        let receipt = self.claim.handoff();
        empty.initialize(
            RichCaptures {
                sidecars,
                physical: self.primary,
            },
            RecycleAnchor {
                physical: self.anchor,
                receipt: Some(receipt),
                generation,
            },
        )
    }
}
impl<'a> PhysicalPreparation<'a, 64> {
    /// Genuine PooledBuffer rich C. No sidecar/native destructor occurs between
    /// sole receipt handoff, private Primary construction and validated slot write.
    /// # Safety
    /// Same closed alias/queue/IO/raw contract as PhysicalPool::cold_in. The
    /// Primary is moved directly into this exact unpublished HeaderRoot; it
    /// cannot escape or mint an External until the genuine rich handle publishes.
    pub(crate) unsafe fn initialize_pooled(
        self,
        empty: ValidatedEmptyHeader<'a, super::buffer_pool::PooledBuffer, RecycleAnchor<64>, 64>,
        size: usize,
        pool: Option<&super::buffer_pool::BufferPool>,
        residency: Option<Arc<super::ResidencySet>>,
    ) -> PreparedHeader<'a, super::buffer_pool::PooledBuffer, RecycleAnchor<64>, 64> {
        // Sidecars may execute cold Arc/Weak bookkeeping BEFORE handoff; any
        // panic still owns armed ReadyReservation and empty validated header.
        let sidecars = super::buffer_pool::PooledBuffer::rich_sidecars(pool, residency);
        let generation = self.claim.next_generation();
        let receipt = self.claim.handoff();
        let captures = unsafe {
            super::buffer_pool::PooledBuffer::rich_primary(self.primary, generation, size, sidecars)
        };
        empty.initialize(
            captures,
            RecycleAnchor {
                physical: self.anchor,
                receipt: Some(receipt),
                generation,
            },
        )
    }
}
/// Actual rich PooledBuffer pool predicate; not a namespace/raw closure inference.
pub(crate) fn pooled_ready(c: &super::buffer_pool::PooledBuffer, a: &RecycleAnchor<64>) -> bool {
    // SAFETY: PooledBuffer rich C exists only in this private exact initializer.
    let primary = unsafe { c.as_ref().physical_primary_unchecked() };
    if !HeaderStrong::same(&primary.header, &a.physical.header) {
        a.physical.record().ready.quarantine();
        return false;
    }
    ready_record(a)
}
fn ready_record<const N: usize>(a: &RecycleAnchor<N>) -> bool {
    let record = a.physical.record();
    let receipt = a.receipt.as_ref().expect("sole receipt");
    match record.ready.claim_abandoned(receipt) {
        Ok(true) => true,
        Ok(false) => match unsafe {
            record.ready.poll_closed_exact_use(
                a.generation,
                !record.aliases.has_external(),
                record.submitted.load(Ordering::Acquire),
                record.clock.completed_count().0,
            )
        } {
            Ok(ready) => ready,
            Err(_) => {
                record.ready.quarantine();
                false
            }
        },
        Err(_) => {
            record.ready.quarantine();
            false
        }
    }
}
impl<const N: usize> RecycleAnchor<N> {
    /// Actual PreparedHeader::anchors hook, immediately before publish-last.
    pub(crate) fn commit(&self) -> Result<(), ReadyError> {
        self.physical
            .record()
            .ready
            .commit_prepared(self.receipt.as_ref().expect("sole receipt"))
            .map(|_| ())
    }
}
/// Installed ONLY on the actual rich header pool, whose claimed DETACHING
/// authority excludes all new primary borrows. Not namespace/global closure.
pub(crate) fn rich_ready<S, const N: usize>(c: &RichCaptures<S, N>, a: &RecycleAnchor<N>) -> bool {
    // Correct-root binding produced solely by PhysicalPreparation::initialize;
    // no safe caller can build these private fields with a cross-root receipt.
    let record = a.physical.record();
    debug_assert!(HeaderStrong::same(&c.physical.header, &a.physical.header));
    ready_record(a)
}
impl<const N: usize> Drop for RecycleAnchor<N> {
    fn drop(&mut self) {
        let receipt = self.receipt.take().expect("sole recycle receipt");
        let record = self.physical.record();
        // Normal HeaderRoot service invokes A only AFTER actual C Drop succeeds.
        // Unexpected producer/preinit Drop cannot authorize PREPARING→FREE.
        if record.ready.finish_prepared_cleanup(receipt).is_err() {
            record.ready.quarantine();
        }
    }
}
impl<const N: usize> ExternalPin<N> {
    fn minted(backing: &PhysicalStrong<N>) -> Result<Self, HeaderError> {
        // SAFETY: this statically selected counter is inside the SAME genuine
        // physical HeaderStrong payload; selector reads only immutable custody.
        let header = unsafe { backing.header.try_clone_with_external(|c| &c.aliases) }?;
        Ok(Self {
            physical: PhysicalStrong { header },
        })
    }
}
impl<const N: usize> Drop for ExternalPin<N> {
    fn drop(&mut self) {
        // Counter-only use of existing AliasPins: no park pointer is ever
        // installed and no node/generation can be stolen by a final notice.
        let returned = self.physical.record().aliases.release_external();
        debug_assert!(returned.is_none());
    }
}
#[derive(Debug)]
pub(crate) enum UseWitnessError {
    DifferentClock,
    EpochExhausted,
    UntrackedLegacy,
    InvalidCustody,
}
impl<const N: usize> NativeUse<N> {
    /// Host terminal generation-page cleanup ONLY, before this lease is released.
    /// # Safety
    /// epoch is the exact JUST-CLOSED epoch issued to this same native generation
    /// and submitted to the supplied clock's queue. It is not a merely staged
    /// epoch on a known no-handoff cancellation and not clock.current_epoch().
    /// Native references and all escaped emission capabilities for this binding
    /// are already closed. Foreign queues need independent terminal receipts;
    /// this creator-clock witness cannot manufacture their terminality.
    /// Error leaves the pin unchanged. Caller must retain/quarantine the owning
    /// initialized slot/root, not log then Drop or unwind through ordinary fields.
    pub(crate) unsafe fn observe_submitted(
        &self,
        clock: &Arc<GpuClock>,
        epoch: crate::retire::Epoch,
    ) -> Result<(), UseWitnessError> {
        let record = self.pin.physical.record();
        if !Arc::ptr_eq(&record.clock, clock) {
            return Err(UseWitnessError::DifferentClock);
        }
        let encoded = epoch
            .0
            .checked_add(1)
            .ok_or(UseWitnessError::EpochExhausted)?;
        record.submitted.fetch_max(encoded, Ordering::Release);
        Ok(())
    }

    /// Engine Buffer primary role only: private construction ensures the value
    /// resides solely in live rich C and public borrow spans genuine rich strong.
    /// This is NOT an API to mint use from a pool descriptor or native baseline.
    /// # Safety
    /// Caller borrows private Primary Buffer solely through a genuine live rich
    /// HeaderStrong, never from descriptor/baseline/service DETACHING custody.
    pub(super) unsafe fn from_buffer_primary(
        physical: &PhysicalStrong<N>,
    ) -> Result<Self, HeaderError> {
        let pin = ExternalPin::minted(physical)?;
        Ok(Self {
            offset: 0,
            length: physical.record().bytes,
            pin,
        })
    }
    pub(super) fn length(&self) -> usize {
        self.length
    }
    pub(super) fn managed_late_bind(&self) -> Option<&super::late_bind::ManagedRelease> {
        self.pin.physical.managed_late_bind()
    }
    pub(crate) fn same_backing(&self, other: &Self) -> bool {
        HeaderStrong::same(&self.pin.physical.header, &other.pin.physical.header)
    }
    pub(crate) fn from_primary<S, const H: usize>(
        primary: &HeaderStrong<RichCaptures<S, N>, RecycleAnchor<N>, H>,
    ) -> Result<Self, HeaderError> {
        // Borrowed genuine rich strong blocks DETACHING/new-issuer closure.
        let captures = primary.captures();
        let pin = ExternalPin::minted(&captures.physical)?;
        Ok(Self {
            offset: 0,
            length: captures.physical.record().bytes,
            pin,
        })
    }
    pub(crate) fn try_clone(&self) -> Result<Self, HeaderError> {
        // Existing owned external pin remains nonzero throughout minting.
        let pin = ExternalPin::minted(&self.pin.physical)?;
        Ok(Self {
            offset: self.offset,
            length: self.length,
            pin,
        })
    }
    /// Transfer a newly minted primary pin into its view without another count
    /// increment/decrement. Relative and absolute range checks remain identical
    /// to borrowed view(), including when this owned handle is already a view.
    pub(super) fn into_view(mut self, offset: usize, length: usize) -> Result<Self, PhysicalError> {
        let absolute = crate::allocation_geometry::checked_view_offset_numeric(
            self.offset,
            self.length,
            offset,
            length,
        )
        .map_err(|_| PhysicalError::ViewRange)?;
        self.offset = absolute;
        self.length = length;
        Ok(self)
    }
    pub(crate) fn view(&self, offset: usize, length: usize) -> Result<Self, PhysicalError> {
        let absolute = crate::allocation_geometry::checked_view_offset_numeric(
            self.offset,
            self.length,
            offset,
            length,
        )
        .map_err(|_| PhysicalError::ViewRange)?;
        let pin = ExternalPin::minted(&self.pin.physical).map_err(PhysicalError::Header)?;
        Ok(Self {
            offset: absolute,
            length,
            pin,
        })
    }
    /// # Safety
    /// Trusted encoding/IO only. Do not export/clone/view the returned Buffer,
    /// raw SDK handles, textures or CPU pointers without retaining a genuine
    /// NativeUse through their complete future-use/native terminal interval.
    /// No Deref/AsRef compatibility route silently circumvents this contract.
    pub(crate) unsafe fn native(&self) -> (&Buffer, usize, usize) {
        // Borrow baseline whole Buffer; the actual encoding/IO adapter MUST
        // apply this absolute offset/length. It is not a legacy Buffer view.
        (
            &self.pin.physical.record().baseline,
            self.offset,
            self.length,
        )
    }
}

// ACTIVATION CONTRACT: rich_ready's exact-only branch may be called only on
// closed default/supplied rich creators whose EVERY safe/native/IO/foreign/raw
// future-use path owns NativeUse through actual terminal+emission closure.
// Actual generation host cleanup observes exact creator cutoff BEFORE dropping
// its pin; never-submitted cancellation skips observation, unknown retains pins.
// Some historical creator max never resets during physical reuse. Foreign uses
// need their own positive terminal receipt, not a substituted creator epoch.
// Existing untracked/default/IO callers do not yet meet this contract. This
// source tranche enables type composition, not native or safe-default activation.

// INACTIVE production type composition; applied/partially compiled, not native
// accepted. Actual cold default OR supplied constructors remain private unsafe.
// Existing safe large-upload branch is NOT switched yet: native+IO/raw closure
// and complete quote remain required. The private device method below preserves
// same-account default payer and supplied promise, not a legacy provenance guess.
// Warm pooling, complete Arc/control/service prices and native enrollment remain open.

pub(crate) struct ColdUnpooled {
    physical: PhysicalPool<64>,
    rich: HeaderPool<super::buffer_pool::PooledBuffer, RecycleAnchor<64>, 64>,
}
impl ColdUnpooled {
    /// Both roots' actual requested slab/root/control/registration terms. Add
    /// ONCE before cold birth; shared service, pass/domain, managed diagnostics,
    /// membership, backing/range/native hidden costs remain factory quote terms.
    /// This is not a complete native/RSS price envelope.
    pub(crate) fn requested_owned_bytes() -> Result<usize, RootError> {
        HeaderPool::<PhysicalRecord, PhysicalAnchors, 64>::requested_owned_bytes()?
            .checked_add(HeaderPool::<
                super::buffer_pool::PooledBuffer,
                RecycleAnchor<64>,
                64,
            >::requested_owned_bytes()?)
            .ok_or(RootError::Admission(
                crate::memory_admission::AllocationError::Overflow,
            ))
    }
    /// Real data/output/staging birth also admits independent diagnostic stamp
    /// and its exact map entry; large upload without this instrument uses the
    /// smaller capsule quote. Actual native/shadow/shared terms stay distinct.
    pub(crate) fn requested_pool_owned_bytes() -> Result<usize, RootError> {
        Self::requested_owned_bytes()?
            .checked_add(super::late_bind::requested_managed_bytes().map_err(RootError::Admission)?)
            .ok_or(RootError::Admission(
                crate::memory_admission::AllocationError::Overflow,
            ))
    }
    /// # Safety
    /// Actual backing/native/creator queue provenance and native/raw/IO terminal
    /// carriers are closed before this route is enabled. Same C/A contracts as
    /// PhysicalPool; no global namespace/member proof. Existing full load
    /// prediction includes requested Layout plus separately complete overhead.
    pub(crate) unsafe fn cold_in(context: &AllocationContext) -> Result<Self, RootError> {
        // Both roots take slices of the SAME existing promise before native
        // allocation. Failed second setup leaves first charge service-retained
        // until actual root storage destruction, never numerically erased.
        let physical = unsafe { PhysicalPool::cold_in(context)? };
        let rich = unsafe { HeaderPool::cold_owned_when_in(context, pooled_ready, None)? };
        Ok(Self { physical, rich })
    }
    /// # Safety
    /// allocate returns the complete real alloc_in result before any usable raw
    /// alias escapes. Actual generation/IO owners must retain every future-use
    /// pin through terminality+emission closure. Initial C has no pooled owner.
    /// This consumes no reuse capacity and imposes no global outstanding cap:
    /// each cold birth obtains its own admitted capsule. Amortized shared slabs
    /// are a later allocator transaction, not a hidden permanent limit.
    pub(crate) unsafe fn allocate_with(
        &self,
        context: &AllocationContext,
        allocate: impl FnOnce() -> Result<CustodiedAllocation, crate::Error>,
        clock: Arc<GpuClock>,
        size: usize,
        residency: Option<Arc<super::ResidencySet>>,
        commands: &super::Commands,
    ) -> Result<super::buffer_pool::BufferHandle, crate::Error> {
        // Root credit is validated BEFORE actual native allocation or any C/A.
        // The cold closure executes outside all engine guards and before any
        // rich payload exists. Header refusal cannot allocate a native backing.
        let empty = self
            .rich
            .reserve_validated_header()
            .map_err(|error| crate::Error::Msg(format!("rich header: {error:?}")))?;
        let allocation = allocate()?;
        let physical =
            unsafe { self.physical.insert_cold(allocation, clock) }.map_err(|rejected| {
                crate::Error::Msg(format!("physical birth: {:?}", rejected.error))
            })?;
        let claim = physical
            .reserve()
            .map_err(|error| crate::Error::Msg(format!("physical occupancy: {error:?}")))?;
        let preparation = claim
            .prepare()
            .map_err(|error| crate::Error::Msg(format!("physical preparation: {error:?}")))?;
        // Sole-root nonfallible initialization immediately establishes host
        // C/A custody. Subsequent ordinary errors queue never-published cleanup.
        let prepared = unsafe { preparation.initialize_pooled(empty, size, None, residency) };
        context.check_open()?;
        commands.check_failed().map_err(crate::Error::wrap)?;
        // No fallible/arbitrary/native operation after LIVE commit before
        // publish-last except existing checked metadata publish, whose impossible
        // phase failure retains/quarantines initialized storage as documented.
        unsafe { prepared.anchors() }
            .commit()
            .map_err(|error| crate::Error::Msg(format!("physical commit: {error:?}")))?;
        let header = prepared
            .publish()
            .map_err(|error| crate::Error::Msg(format!("rich publish: {error:?}")))?;
        Ok(super::buffer_pool::BufferHandle::from_rich(header))
    }
}

/// One cold plan-assigned whole physical data/output bucket position. This
/// descriptor is not a usable native alias and never enters legacy FreeNode.
/// A plan holds the actual creator capability and supplies admitted cold spill
/// on Busy/Full; no permanent outstanding-count cap is introduced. Independent
/// physical C/A own member/native/range/account across every checkout and FREE.
/// No output bytes are initialized by checkout: the actual operation must
/// establish write-before-read; this is not a staging/CPU initialization API.
pub(crate) struct PreparedPoolSlot {
    physical: PhysicalStrong<64>,
    capsule: ColdUnpooled,
}
struct PreparedRichCheckout<'a> {
    empty: ValidatedEmptyHeader<'a, super::buffer_pool::PooledBuffer, RecycleAnchor<64>, 64>,
    preparation: PhysicalPreparation<'a, 64>,
    physical: &'a PhysicalStrong<64>,
}
impl ColdUnpooled {
    fn check_checkout_context(
        &self,
        physical: &PhysicalStrong<64>,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<(), StagingError> {
        if context.domain_key() != self.physical.domain {
            return Err(StagingError::DifferentDomain);
        }
        context.check_open().map_err(StagingError::Admission)?;
        commands
            .check_failed_atomic()
            .map_err(StagingError::Native)?;
        self.physical
            .headers
            .check_failed()
            .map_err(StagingError::Root)?;
        self.rich.check_failed().map_err(StagingError::Root)?;
        physical
            .record()
            .ready
            .check_failed()
            .map_err(StagingError::Ready)
    }
    /// Both actual CPU staging and data/output checkout use this SAME genuine
    /// header/physical reservation. No native operation or user callback runs.
    fn prepare_checkout<'a>(
        &'a self,
        physical: &'a PhysicalStrong<64>,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<PreparedRichCheckout<'a>, StagingError> {
        self.check_checkout_context(physical, context, commands)?;
        let empty = self
            .rich
            .reserve_validated_header()
            .map_err(StagingError::Root)?;
        let claim = physical.reserve().map_err(StagingError::Ready)?;
        let preparation = claim.prepare().map_err(StagingError::Physical)?;
        self.check_checkout_context(physical, context, commands)?;
        Ok(PreparedRichCheckout {
            empty,
            preparation,
            physical,
        })
    }
}
impl PreparedRichCheckout<'_> {
    /// # Safety
    /// Same closed role/native/IO/foreign/CPU contract as actual cold birth;
    /// unpublished private Primary moves straight into genuine rooted C/A.
    /// CPU staging finishes its exclusive complete initialization BEFORE entry.
    unsafe fn publish(
        self,
        bytes: usize,
    ) -> Result<super::buffer_pool::BufferHandle, StagingError> {
        let prepared = unsafe {
            self.preparation
                .initialize_pooled(self.empty, bytes, None, None)
        };
        unsafe { prepared.anchors() }
            .commit()
            .map_err(StagingError::Ready)?;
        self.physical
            .managed_late_bind()
            .expect("cold plan diagnostic")
            .acquired();
        let header = prepared.publish().map_err(StagingError::Root)?;
        Ok(super::buffer_pool::BufferHandle::from_rich(header))
    }
}
impl PreparedPoolSlot {
    pub(crate) fn capacity(&self) -> usize {
        self.physical.bytes()
    }
    pub(crate) fn check_context(
        &self,
        context: &AllocationContext,
        commands: &super::Commands,
    ) -> Result<(), StagingError> {
        self.check_context_under(&context.operation_context(), commands)
    }
    pub(crate) fn check_context_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<(), StagingError> {
        self.capsule
            .check_checkout_context(&self.physical, context, commands)
    }
    /// Warm direct plan position, not the legacy mutex/free-list lookup.
    /// Ordinary refusal preserves original FREE generation/cutoff, metadata
    /// credit, and diagnostic stamp. Initialized abandonment is service-owned.
    /// # Safety
    /// Actual rich input/output operation must retain creator/External/native
    /// ownership, initialize all bytes it reads, and close every native/IO/raw/
    /// foreign/escaped emission capability before reuse. Allocation is NOT an
    /// initialized Tensor value. No safe selector is enabled by this method.
    pub(crate) unsafe fn try_checkout(
        &self,
        context: &AllocationContext,
        commands: &super::Commands,
    ) -> Result<super::buffer_pool::BufferHandle, StagingError> {
        unsafe { self.try_checkout_under(&context.operation_context(), commands) }
    }
    /// # Safety
    /// Same originating cold-price/closed-use/initialized-range obligations as
    /// the AllocationContext compatibility wrapper; warm permission alone is
    /// neither a native-use receipt nor a write-before-read certificate.
    pub(crate) unsafe fn try_checkout_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<super::buffer_pool::BufferHandle, StagingError> {
        let checkout = self
            .capsule
            .prepare_checkout(&self.physical, context, commands)?;
        unsafe { checkout.publish(self.capacity()) }
    }
}

/// One cold plan-assigned Shared staging position. The descriptor is NOT a
/// primary/native-use alias: previous primary and every generation/IO/ICB pin
/// must retire before FREE authorizes any CPU write. No clock comparison can
/// substitute for this exclusive occupancy. A caller obtains cold spill slots
/// when Busy or rich-header Full; 64 header cells are slab granularity, not a
/// permanent limit on dispatches/old weak handles.
///
/// The capsule's roots are independently registered; final slot Drop only
/// releases counted metadata/capabilities. Native/member/range/charge owners
/// remain in actual physical C/A until host cleanup. No backedge to this slot,
/// its plan, namespace, Commands, or another pool is stored in physical C/A.
pub(crate) struct PreparedStagingSlot {
    physical: PhysicalStrong<64>,
    capsule: ColdUnpooled,
    // Queried exactly once at cold enrollment. Never expose this address or
    // invoke an SDK contents getter on the warm rewrite path.
    mapped: usize,
    bytes: usize,
}
#[derive(Debug)]
pub(crate) enum StagingError {
    Root(RootError),
    Ready(ReadyError),
    Physical(PhysicalError),
    Admission(crate::memory_admission::AllocationError),
    Native(crate::kernels::MetalKernelError),
    DifferentDomain,
    Length,
}
impl PreparedStagingSlot {
    pub(crate) fn capacity(&self) -> usize {
        self.bytes
    }
    /// Validate authentic plan memo affinity and failure before either unchanged
    /// upgrade or a rewrite. This observation takes no credit and does not
    /// require FREE: an unchanged live handle is allowed, but cannot authorize
    /// a CPU write. It is not an execution lease; operation EDIT remains needed.
    pub(crate) fn check_context(
        &self,
        context: &AllocationContext,
        commands: &super::Commands,
    ) -> Result<(), StagingError> {
        self.check_context_under(&context.operation_context(), commands)
    }
    pub(crate) fn check_context_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<(), StagingError> {
        self.capsule
            .check_checkout_context(&self.physical, context, commands)
    }

    /// Warm plan transaction BEFORE native generation's exact EDIT. Only
    /// checked atomics, bounded memcpy and zero-padding run here; no allocation/admission
    /// mutex, native retain/getter, service Wake or arbitrary callback runs.
    /// Root/occupancy/count refusal precedes copy. Context/failure are checked
    /// again after reservation, preserving the original eligibility/cutoff on
    /// ordinary refusal. No global success counter needs rollback.
    ///
    /// The returned rich Handle is actual creator authority, not a reconstructed
    /// native pointer. Caller then mints the matching generation PreparedBuffer
    /// and keeps the original handle/External custody across tuple refusal.
    /// A successful copy/publish may change a host shadow/memo; prior refusals
    /// must not. An unchanged memo hit may clone/upgrade that genuine rich
    /// handle WITHOUT writing; it is a distinct caller transaction.
    ///
    /// Bytes are owned by the caller's preallocated plan shadow. Do not pass an
    /// unsafe-interop slice aliasing this slot's mapped storage. The cold unsafe
    /// constructor establishes that every other usable access owns a pin.
    pub(crate) fn try_write(
        &self,
        bytes: &[u8],
        context: &AllocationContext,
        commands: &super::Commands,
    ) -> Result<super::buffer_pool::BufferHandle, StagingError> {
        self.try_write_under(bytes, &context.operation_context(), commands)
    }
    pub(crate) fn try_write_under(
        &self,
        bytes: &[u8],
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &super::Commands,
    ) -> Result<super::buffer_pool::BufferHandle, StagingError> {
        if bytes.len() > self.bytes {
            return Err(StagingError::Length);
        }
        let checkout = self
            .capsule
            .prepare_checkout(&self.physical, context, commands)?;
        // SAFETY: cold constructor retained this exact full Shared mapping in
        // physical C/A. Sole PREPARING closes every prior primary/external
        // access and prevents another writer. Bounds checked above; input is
        // the independent caller shadow. No raw pointer leaves this scope.
        unsafe {
            let mapped = self.mapped as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len());
            // Published PooledBuffer/Buffer length is the entire slot capacity.
            // Initialize EVERY remaining byte, not merely a caller-assumed
            // shader prefix; shorter packed/array payloads expose only zeros.
            std::ptr::write_bytes(mapped.add(bytes.len()), 0, self.bytes - bytes.len());
        }
        // No fallible/arbitrary/native operation between memcpy and this exact
        // nonfallible validated cell write. Same physical receipt is transferred
        // once; initialized abandonment stays rooted for C cleanup then FREE.
        unsafe { checkout.publish(self.bytes) }
    }
}
impl ColdUnpooled {
    /// One real reusable whole data/output allocation, retaining exact custody.
    /// # Safety
    /// Same closed default/supplied native/IO/foreign/raw/CPU use contract as
    /// cold_in and allocate_with. allocate produces whole storage, never a view.
    /// No alias escapes before insert_cold. Every escaped ICB/replay/emission
    /// capability retains its counted External pin, including after native
    /// terminality if it can emit again. Complete cold prices precede entry.
    pub(crate) unsafe fn allocate_pool_with(
        self,
        context: &AllocationContext,
        allocate: impl FnOnce() -> Result<CustodiedAllocation, crate::Error>,
        clock: Arc<GpuClock>,
        bytes: usize,
        commands: &super::Commands,
    ) -> Result<PreparedPoolSlot, crate::Error> {
        if bytes == 0 {
            return Err(crate::Error::Msg(
                "staging requires nonempty storage".into(),
            ));
        }
        context.check_open()?;
        commands.check_failed().map_err(crate::Error::wrap)?;
        let physical = unsafe { self.physical.insert_cold(allocate()?, clock) }
            .map_err(|r| crate::Error::Msg(format!("staging physical birth: {:?}", r.error)))?;
        if bytes > physical.bytes() {
            return Err(crate::Error::Msg("staging physical extent mismatch".into()));
        }
        context.check_open()?;
        commands.check_failed().map_err(crate::Error::wrap)?;
        // Cold diagnostic enrollment is mandatory even when instrument is off,
        // so later instrumentation cannot route a managed acquire to a mutex.
        // Full price includes this Arc/map entry. Only final physical C may
        // remove it, before independent A frees actual native storage.
        let (state, node) =
            super::late_bind::prepare_managed(context).map_err(crate::Error::wrap)?;
        if let Err(uninstalled) = physical.record().managed_late_bind.set(state) {
            // Cold invariant refusal, BEFORE publishing any map entry. Preserve
            // the already-installed record/map identity and drop only this
            // independent diagnostic allocation offguard; no assertion/unwind
            // may orphan a newly published diagnostic registration.
            drop(node);
            drop(uninstalled);
            return Err(crate::Error::Msg(
                "staging diagnostic already installed".into(),
            ));
        }
        super::late_bind::register_managed(
            unsafe { physical.baseline_unchecked() },
            physical
                .record()
                .managed_late_bind
                .get()
                .expect("just installed"),
            node,
        )
        .map_err(|why| crate::Error::Msg(why.into()))?;
        Ok(PreparedPoolSlot {
            physical,
            capsule: self,
        })
    }
    /// Shared-only CPU staging wrapper over the SAME cold whole physical birth.
    /// # Safety
    /// Same complete closed-use/pricing contract as allocate_pool_with, plus
    /// stable full Shared mapping with no other untracked CPU writer/reader.
    pub(crate) unsafe fn allocate_staging_with(
        self,
        context: &AllocationContext,
        allocate: impl FnOnce() -> Result<CustodiedAllocation, crate::Error>,
        clock: Arc<GpuClock>,
        bytes: usize,
        commands: &super::Commands,
    ) -> Result<PreparedStagingSlot, crate::Error> {
        let slot = unsafe { self.allocate_pool_with(context, allocate, clock, bytes, commands)? };
        // SAFETY: no Primary/use escaped; cold constructor retained full Shared
        // allocation. Refusal releases descriptor roots to service, not native.
        let mapped = unsafe { slot.physical.baseline_unchecked().data() } as usize;
        if mapped == 0 {
            return Err(crate::Error::Msg(
                "staging requires Shared CPU mapping".into(),
            ));
        }
        let bytes = slot.capacity();
        let PreparedPoolSlot { physical, capsule } = slot;
        Ok(PreparedStagingSlot {
            physical,
            capsule,
            mapped,
            bytes,
        })
    }
}
