//! SOURCE DRAFT: actual M4 carrier, not activated or native-accepted.
//! Requires reviewed #952 HeaderRoot relay/validated preparation and counted
//! Buffer lease migration. Cold constructors/append are outside EntryState.
//! Layout quotes below remain partial; default payer and complete price open.
use crate::{
    allocation_header_root::{HeaderPool, HeaderStrong},
    deferred_failure_relay::DeferredFailureRelay,
    kernels::{
        metal::{buffer::NativeBufferLease, buffer_pool::GpuClock, Buffer},
        MetalKernelError,
    },
    memory_admission::AllocationContext,
    retire::Epoch,
    retirement_owners::{Failure, FailureSignal},
};
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
};

// Existing HeaderPool slab granularity, NOT a generation/workload capacity.
const SLAB: usize = 64;
// Inline PAGE credit granularity only. Cold append recovers the SAME generation.
const PAGE: usize = 64;
const EMPTY: u8 = 0;
const RESERVED: u8 = 1;
const LIVE: u8 = 2;
const LINK_WRITING: u8 = 1;
const LINK_LIVE: u8 = 2;

type Head = HeaderStrong<Generation, (), SLAB>;
type PageHead = HeaderStrong<BindingPage, (), SLAB>;
// The witness is independently rooted: pages may outlive Generation C cleanup.
// No witness->generation/page/Core/namespace edge exists.
type WitnessHead = HeaderStrong<UseWitness, (), SLAB>;

fn refused(message: &'static str) -> MetalKernelError {
    MetalKernelError::QueueUnavailable(message)
}
fn metadata<T>(result: Result<T, impl std::fmt::Debug>) -> Result<T, MetalKernelError> {
    result.map_err(|_| refused("native binding metadata refused"))
}

pub(super) struct GenerationBindings {
    head: Head,
}
/// Cold-created pair. Unconsumed preparation owns no native handoff and cancels
/// its witness on Drop; consuming into Queue does not itself mark terminality.
pub(super) struct PreparedGeneration {
    submission: Option<GenerationBindings>,
    slot: Option<GenerationBindings>,
    queue_failure: Arc<FailureSignal>,
}
impl PreparedGeneration {
    pub(super) fn matches_queue(&self, failure: &Arc<FailureSignal>) -> bool {
        Arc::ptr_eq(&self.queue_failure, failure)
    }
    pub(super) fn constructor_head(&self) -> Result<GenerationBindings, MetalKernelError> {
        self.submission.as_ref().expect("prepared head").try_clone()
    }
    pub(super) fn take(mut self) -> (GenerationBindings, GenerationBindings) {
        (
            self.submission.take().expect("prepared submission head"),
            self.slot.take().expect("prepared slot head"),
        )
    }
}
impl Drop for PreparedGeneration {
    fn drop(&mut self) {
        if let Some(head) = self.submission.as_ref().or(self.slot.as_ref()) {
            // SAFETY: this pair was never consumed into a native constructor.
            // No raw object/cached reference was attached to this head.
            if unsafe { head.terminal_cancelled() }.is_err() {
                head.quarantine(Failure::AdapterFailed);
            }
        }
    }
}
struct WitnessPreparation {
    witness: Option<WitnessHead>,
}
impl Drop for WitnessPreparation {
    fn drop(&mut self) {
        if let Some(witness) = &self.witness {
            // No native sidecar was admitted; cold initialized pages stay rooted
            // and become reclaimable instead of stranding an OPEN witness.
            let _ = unsafe { witness.captures().terminal_cancelled() };
        }
    }
}
struct Generation {
    witness: WitnessHead,
    relay: Arc<DeferredFailureRelay>,
    queue_failure: Arc<FailureSignal>,
    first_phase: AtomicU8,
    first: UnsafeCell<MaybeUninit<PageHead>>,
    // Forward-only pointer into the immutable strong-owned page chain. Only a
    // genuine Head borrow accesses it; it is not an acquisition capability.
    cursor: AtomicPtr<BindingPage>,
}
/// Reusable cold control/Queue-owned slab credits. Neither root C contains this
/// inventory, Queue, Commands, namespace or model, so no retention backedge.
pub(crate) struct BindingInventory {
    witnesses: HeaderPool<UseWitness, (), SLAB>,
    heads: HeaderPool<Generation, (), SLAB>,
    // One initialized first page per prepared generation, BEFORE any native
    // constructor. Additional operation pages have their separate payer.
    first_pages: HeaderPool<BindingPage, (), SLAB>,
    clock: Arc<GpuClock>,
    relay: Arc<DeferredFailureRelay>,
    queue_failure: Arc<FailureSignal>,
}
/// Operation/plan-owned page credits. Cold setup may use its supplied context;
/// shared inventory storage and operation tranche have distinct explicit payer.
/// No first-model inference and no per-bind reserve/registry/Vec growth.
pub(crate) struct BindingCapacity {
    pages: HeaderPool<BindingPage, (), SLAB>,
    relay: Arc<DeferredFailureRelay>,
    queue_failure: Arc<FailureSignal>,
}
struct BindingPage {
    witness: WitnessHead,
    relay: Arc<DeferredFailureRelay>,
    next_index: AtomicUsize,
    slots: [BindingSlot; PAGE],
    link_phase: AtomicU8,
    next: UnsafeCell<MaybeUninit<PageHead>>,
    failed: AtomicU8,
}
enum BindingLease {
    Buffer(NativeBufferLease),
    // Genuine catalog root + exclusive SAME-node writer. No operation Arc or
    // parallel registry. Host page cleanup, not method return, ends the writer.
    Plan(crate::metal_backend::CatalogPlanLease),
}
struct BindingSlot {
    state: AtomicU8,
    lease: UnsafeCell<MaybeUninit<BindingLease>>,
}
// SAFETY: native Buffer role's Send/Sync contract is a prerequisite from #952.
// Cursor/link publication writes only under a sole CAS; slots have monotone,
// unique claims. LIVE Release publishes lease, never replaced/recycled. Genuine
// Head/PageHead borrows span producer access. Service alone drops payload after
// strong0; ready reads only LIVE slots. No bare pointer acquisition API escapes.
unsafe impl Send for BindingPage {}
unsafe impl Sync for BindingPage {}
unsafe impl Send for Generation {}
unsafe impl Sync for Generation {}

fn generation_ready(generation: &Generation, _: &()) -> bool {
    generation.witness.captures().cutoff().is_some()
}
fn page_ready(page: &BindingPage, _: &()) -> bool {
    if page.failed.load(Ordering::Acquire) != 0 {
        return false;
    }
    // Exact strong0 service claim excludes every writer/issuer. An empty fresh
    // leaf that lost append owns no native lease or linked successor. It can
    // return metadata credits independently of the recording's OPEN witness.
    // RESERVED is never silently empty and initialized LIVE cannot take this arm.
    if page.link_phase.load(Ordering::Acquire) == EMPTY
        && page
            .slots
            .iter()
            .all(|slot| slot.state.load(Ordering::Acquire) == EMPTY)
    {
        return true;
    }
    let Some(cutoff) = page.witness.captures().cutoff() else {
        return false;
    };
    // HeaderRoot invokes this after exact strong0 claim; there is no remaining
    // slot writer/append borrower. This is service-only and atoms-only.
    for slot in &page.slots {
        match slot.state.load(Ordering::Acquire) {
            EMPTY => {}
            LIVE => {
                let lease = unsafe { (*slot.lease.get()).assume_init_ref() };
                // A Plan has no allocation-use stamp. Its writer/root is kept
                // until this SAME page's exact terminal/emission cutoff, then
                // service C Drop releases it. Unknown cannot authorize either.
                let BindingLease::Buffer(lease) = lease else {
                    continue;
                };
                if let Cutoff::Submitted { clock, epoch } = cutoff {
                    // SAFETY: actual native terminal+returned-handoff fact is
                    // immutable, and this SAME page's future-emission holders
                    // are gone. Creator-clock mismatch is refusal, not Drop.
                    if unsafe { lease.observe_submitted(clock, epoch) }.is_err() {
                        page.failed.store(1, Ordering::Release);
                        page.relay.publish(Failure::AdapterFailed);
                        return false;
                    }
                }
            }
            _ => {
                // RESERVED/WRITING cannot be treated as EMPTY on unwind. Keep
                // initialized native pins/root, signal outside active lock.
                page.failed.store(1, Ordering::Release);
                page.relay.publish(Failure::AdapterFailed);
                return false;
            }
        }
    }
    true
}
impl BindingPage {
    fn new(witness: WitnessHead, relay: Arc<DeferredFailureRelay>) -> Self {
        Self {
            witness,
            relay,
            next_index: AtomicUsize::new(0),
            slots: std::array::from_fn(|_| BindingSlot {
                state: AtomicU8::new(EMPTY),
                lease: UnsafeCell::new(MaybeUninit::uninit()),
            }),
            link_phase: AtomicU8::new(EMPTY),
            next: UnsafeCell::new(MaybeUninit::uninit()),
            failed: AtomicU8::new(0),
        }
    }
    fn next(&self) -> Option<&PageHead> {
        (self.link_phase.load(Ordering::Acquire) == LINK_LIVE)
            .then(|| unsafe { (*self.next.get()).assume_init_ref() })
    }
    /// Cold-only fresh, unlinked leaf. No native/fallible/user operation between
    /// unique WRITING claim, write and LIVE. Never overwrites/creates a cycle.
    fn append(&self, fresh: PageHead) -> Result<(), PageHead> {
        if self
            .link_phase
            .compare_exchange(EMPTY, LINK_WRITING, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err(fresh);
        }
        unsafe {
            (*self.next.get()).write(fresh);
        }
        self.link_phase.store(LINK_LIVE, Ordering::Release);
        Ok(())
    }
}
impl Drop for BindingPage {
    fn drop(&mut self) {
        // Called by HeaderRoot only after ready=true + strong0. Native-use
        // Buffer leases are counted External only, never legacy baseline Drop.
        // Plan Drop releases only exact writer + opaque rooted metadata; actual
        // pipeline/shadow/native C remains independently service-owned.
        // No arbitrary callback/Box/user finalizer is installed in these slots.
        for slot in &mut self.slots {
            if *slot.state.get_mut() == LIVE {
                unsafe {
                    slot.lease.get_mut().assume_init_drop();
                }
            }
        }
        if *self.link_phase.get_mut() == LINK_LIVE {
            unsafe {
                self.next.get_mut().assume_init_drop();
            }
        }
    }
}

impl BindingInventory {
    pub(super) fn requested_bytes() -> Result<usize, MetalKernelError> {
        // Evaluate every fallible owned-layout quote before checked arithmetic:
        // unsupported toolchain/target is not mislabeled as integer overflow.
        let witnesses = metadata(HeaderPool::<UseWitness, (), SLAB>::requested_owned_bytes())?;
        let heads = metadata(HeaderPool::<Generation, (), SLAB>::requested_owned_bytes())?;
        let pages = metadata(HeaderPool::<BindingPage, (), SLAB>::requested_owned_bytes())?;
        witnesses
            .checked_add(heads)
            .and_then(|bytes| bytes.checked_add(pages))
            .ok_or_else(|| refused("binding inventory quote overflow"))
    }
    /// Cold plan/device/refill boundary only, not per CB/bind. Account takes the
    /// owned three-root tranche before allocation; owned root/storage/registration/
    /// charge-control terms included. Shared relay/service/pass/domain/native and
    /// allocator rounding remain separately required, not a total RSS quote.
    pub(super) unsafe fn cold_in(
        context: &AllocationContext,
        clock: Arc<GpuClock>,
        relay: Arc<DeferredFailureRelay>,
        queue_failure: Arc<FailureSignal>,
    ) -> Result<Self, MetalKernelError> {
        let witnesses = metadata(unsafe {
            HeaderPool::cold_owned_when_in(context, |_, _| true, Some(relay.clone()))
        })?;
        let heads = metadata(unsafe {
            HeaderPool::cold_owned_when_in(context, generation_ready, Some(relay.clone()))
        })?;
        let first_pages = metadata(unsafe {
            HeaderPool::cold_owned_when_in(context, page_ready, Some(relay.clone()))
        })?;
        Ok(Self {
            witnesses,
            heads,
            first_pages,
            clock,
            relay,
            queue_failure,
        })
    }
    /// Warm bounded atoms-only metadata preparation, BEFORE native Slot removal.
    /// Full is explicit recovery demand; caller refills inventory outside guards,
    /// never registers another root or grows a container here. 64 is PER SLAB.
    pub(super) fn prepare(&self) -> Result<PreparedGeneration, MetalKernelError> {
        self.try_prepare()
            .map_err(BindingPreparationError::native_error)
    }
    fn try_prepare(&self) -> Result<PreparedGeneration, BindingPreparationError> {
        let vacant_head = self
            .heads
            .reserve_validated_header()
            .map_err(BindingPreparationError::from)?;
        let vacant_witness = self
            .witnesses
            .reserve_validated_header()
            .map_err(BindingPreparationError::from)?;
        let vacant_page = self
            .first_pages
            .reserve_validated_header()
            .map_err(BindingPreparationError::from)?;
        let witness = vacant_witness
            .initialize(UseWitness::new(self.clock.clone()), ())
            .publish()
            .map_err(BindingPreparationError::from)?;
        // Own cancellation BEFORE the first fallible counted clone. A refused
        // clone must not drop the only OPEN witness without a terminal cancel.
        let mut cancellation = WitnessPreparation {
            witness: Some(witness),
        };
        let witness = cancellation
            .witness
            .as_ref()
            .expect("preparation witness")
            .try_clone()
            .map_err(BindingPreparationError::from)?;
        let page_witness = cancellation
            .witness
            .as_ref()
            .expect("preparation witness")
            .try_clone()
            .map_err(BindingPreparationError::from)?;
        let first = vacant_page
            .initialize(BindingPage::new(page_witness, self.relay.clone()), ())
            .publish()
            .map_err(BindingPreparationError::from)?;
        let cursor = std::ptr::from_ref(first.captures()).cast_mut();
        let head = vacant_head
            .initialize(
                Generation {
                    witness,
                    relay: self.relay.clone(),
                    queue_failure: self.queue_failure.clone(),
                    first_phase: AtomicU8::new(LINK_LIVE),
                    first: UnsafeCell::new(MaybeUninit::new(first)),
                    cursor: AtomicPtr::new(cursor),
                },
                (),
            )
            .publish()
            .map_err(BindingPreparationError::from)?;
        let submission = GenerationBindings { head };
        let slot = submission
            .try_clone()
            .map_err(|_| BindingPreparationError::Adapter)?;
        cancellation.witness.take();
        Ok(PreparedGeneration {
            submission: Some(submission),
            slot: Some(slot),
            queue_failure: self.queue_failure.clone(),
        })
    }
}
impl BindingInventory {
    /// Additional preadmitted page storage can belong to a supplied operation
    /// tranche without changing the shared base storage payer or buffer origin.
    pub(super) unsafe fn cold_capacity(
        &self,
        context: &AllocationContext,
    ) -> Result<BindingCapacity, MetalKernelError> {
        unsafe { BindingCapacity::cold_in(context, self.relay.clone(), self.queue_failure.clone()) }
    }
}
impl BindingCapacity {
    pub(super) fn requested_bytes() -> Result<usize, MetalKernelError> {
        metadata(HeaderPool::<BindingPage, (), SLAB>::requested_owned_bytes())
    }
    /// Cold plan/refill, never inside set_param/EntryState. This exact capacity
    /// tranche belongs to its explicit supplied/default metadata promise.
    pub(super) unsafe fn cold_in(
        context: &AllocationContext,
        relay: Arc<DeferredFailureRelay>,
        queue_failure: Arc<FailureSignal>,
    ) -> Result<Self, MetalKernelError> {
        let pages = metadata(unsafe {
            HeaderPool::cold_owned_when_in(context, page_ready, Some(relay.clone()))
        })?;
        Ok(Self {
            pages,
            relay,
            queue_failure,
        })
    }
    fn prepare_page(&self, witness: &WitnessHead) -> Result<PageHead, MetalKernelError> {
        let vacant = metadata(self.pages.reserve_validated_header())?;
        let witness = metadata(witness.try_clone())?;
        metadata(
            vacant
                .initialize(BindingPage::new(witness, self.relay.clone()), ())
                .publish(),
        )
    }
}
impl Generation {
    fn first(&self) -> Option<&PageHead> {
        (self.first_phase.load(Ordering::Acquire) == LINK_LIVE)
            .then(|| unsafe { (*self.first.get()).assume_init_ref() })
    }
    fn append_first(&self, fresh: PageHead) -> Result<(), PageHead> {
        if self
            .first_phase
            .compare_exchange(EMPTY, LINK_WRITING, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err(fresh);
        }
        let cursor = std::ptr::from_ref(fresh.captures()).cast_mut();
        unsafe {
            (*self.first.get()).write(fresh);
        }
        self.first_phase.store(LINK_LIVE, Ordering::Release);
        self.cursor.store(cursor, Ordering::Release);
        Ok(())
    }
    /// Exact original encoder EDIT excludes every other slot writer. Build the
    /// whole immutable page/index plan first: Full changes NO credits/cursor.
    /// Cold append can add forward links but never replaces an existing link.
    /// # Safety
    /// Caller owns that generation's original Ticket EDIT for this entire call;
    /// ALL callers that publish slots must satisfy this same admission contract.
    unsafe fn reserve_slots<const N: usize>(
        &self,
    ) -> Result<[*const BindingSlot; N], MetalKernelError> {
        let start = self.cursor.load(Ordering::Acquire);
        let plan = crate::native_binding_preflight::plan_page_credits::<_, N>(
            start,
            PAGE,
            |page| {
                if page.is_null() {
                    return None;
                }
                Some(unsafe { &*page }.next_index.load(Ordering::Relaxed))
            },
            |page| {
                unsafe { &*page }
                    .next()
                    .map(|head| std::ptr::from_ref(head.captures()).cast_mut())
            },
        )
        .ok_or_else(|| refused("native binding batch needs cold capacity refill"))?;
        // No fallible operation, callback, native effect or allocation below.
        // Stack plan indexes are exact valid PAGE slots; sole EDIT keeps them
        // unchanged. Each slot is unique even when a batch crosses page links.
        let slots = plan.slots.map(|(page, index)| {
            let page = unsafe { &*page };
            page.next_index.store(index + 1, Ordering::Relaxed);
            let slot = &page.slots[index];
            slot.state.store(RESERVED, Ordering::Release);
            std::ptr::from_ref(slot)
        });
        if N != 0 {
            self.cursor.store(plan.last_page, Ordering::Release);
        }
        Ok(slots)
    }
}
impl Drop for Generation {
    fn drop(&mut self) {
        // Host service only after terminal readiness + strong0. Unpublished
        // WRITING cannot be free; publication itself has no panic/fallible gap.
        if *self.first_phase.get_mut() == LINK_LIVE {
            unsafe {
                self.first.get_mut().assume_init_drop();
            }
        }
    }
}
impl GenerationBindings {
    pub(super) fn belongs_to_clock(&self, clock: &Arc<GpuClock>) -> bool {
        Arc::ptr_eq(&self.head.captures().witness.captures().clock, clock)
    }
    pub(super) fn try_clone(&self) -> Result<Self, MetalKernelError> {
        metadata(self.head.try_clone()).map(|head| Self { head })
    }
    pub(super) fn same(&self, other: &Self) -> bool {
        HeaderStrong::same(&self.head, &other.head)
    }
    pub(super) fn publish_failure(&self, reason: Failure) {
        self.head.publish_failure(reason);
    }
    pub(super) fn quarantine(&self, reason: Failure) {
        let _ = self.head.captures().witness.captures().quarantine();
        self.publish_failure(reason);
    }
    pub(super) fn stage_epoch(&self, epoch: Epoch) -> Result<(), MetalKernelError> {
        metadata(self.head.captures().witness.captures().stage_epoch(epoch))
    }
    /// Native entry may only consume the exact epoch staged by its owner.
    /// Missing owner preparation cannot be recovered by taking another epoch.
    pub(super) fn require_staged(&self) -> Result<(), MetalKernelError> {
        let witness = self.head.captures().witness.captures();
        match witness.phase.load(Ordering::Acquire) {
            STAGED => Ok(()),
            _ => Err(MetalKernelError::OwnerSubmissionRequired),
        }
    }
    /// M3 ordered record drain only, after ALL arbitrary callbacks returned.
    /// # Safety
    /// This exact generation's record was selected by the associated prefix
    /// drainer; every preceding registered record's own GPU work is terminal.
    pub(super) unsafe fn ordered_callbacks_finished(&self) {
        self.head
            .captures()
            .witness
            .captures()
            .ordered_callbacks
            .store(true, Ordering::Release);
    }
    pub(super) fn entered(&self) -> Result<(), MetalKernelError> {
        metadata(self.head.captures().witness.captures().entered())
    }
    pub(super) fn returned(&self) -> Result<(), MetalKernelError> {
        metadata(self.head.captures().witness.captures().returned())
    }
    pub(super) unsafe fn terminal_submitted(&self) -> Result<(), MetalKernelError> {
        let witness = self.head.captures().witness.captures();
        metadata(unsafe { witness.terminal_submitted() })?;
        // M4 caller supplies ordered callback-prefix + returned-handoff proof.
        // This also covers direct public commit lacking Commands' clock callback;
        // Commands' earlier mark_completed of the same exact epoch is idempotent.
        witness
            .clock
            .mark_completed(Epoch(witness.closed_epoch.load(Ordering::Relaxed)));
        Ok(())
    }
    pub(super) unsafe fn terminal_cancelled(&self) -> Result<(), MetalKernelError> {
        metadata(unsafe { self.head.captures().witness.captures().terminal_cancelled() })
    }
    /// Consume preadmitted page credit on the SAME generation, atoms only.
    /// Cold allocation/refill is performed separately if capacity has no credit.
    /// Caller calls this at operation preflight BEFORE any executor/native effect.
    pub(super) fn append_capacity(
        &self,
        capacity: &BindingCapacity,
    ) -> Result<(), MetalKernelError> {
        let generation = self.head.captures();
        if !Arc::ptr_eq(&generation.queue_failure, &capacity.queue_failure) {
            return Err(refused("binding capacity belongs to another native queue"));
        }
        let fresh = capacity.prepare_page(&generation.witness)?;
        if let Some(first) = generation.first() {
            // Append is a preflight/refill action, never repeated per bind. The
            // forward reserve cursor below avoids O(pages) lookup in setters.
            let mut tail = first.captures();
            while let Some(next) = tail.next() {
                tail = next.captures();
            }
            tail.append(fresh)
                .map_err(|_| refused("native binding capacity append busy"))
        } else {
            generation
                .append_first(fresh)
                .map_err(|_| refused("native binding first page append busy"))
        }
    }
    /// Genuine counted mint ONLY. Refusal and stale-generation recovery consume
    /// no page credit; caller may prepare every tuple member before admission.
    pub(super) fn prepare_buffer(
        &self,
        buffer: &Buffer,
    ) -> Result<PreparedBuffer, MetalKernelError> {
        let generation = self.try_clone()?;
        let lease = buffer
            .counted_native_lease()
            .map_err(|_| refused("native buffer lease refused"))?;
        if !lease.is_counted_physical_use() {
            return Err(refused(
                "legacy buffer needs unsafe creator-custody interop",
            ));
        }
        Ok(PreparedBuffer { generation, lease })
    }
    /// Install the WHOLE already-preflighted tuple before the first native/
    /// executor/capture effect. Stack arrays impose no workload capacity limit;
    /// cold page append/refill remains on this SAME generation.
    /// # Safety
    /// Exact original Ticket EDIT remains held through install and lowering.
    /// Every slot writer is this method under that same EDIT. All ordinary
    /// geometry/capture/argument checks have completed before installation.
    pub(super) unsafe fn install_prepared<const N: usize>(
        &self,
        prepared: [PreparedBuffer; N],
    ) -> Result<[InstalledBuffer; N], MetalKernelError> {
        for binding in &prepared {
            if !binding.matches(self) {
                return Err(refused("stale native binding generation"));
            }
        }
        let slots = unsafe { self.head.captures().reserve_slots::<N>() }?;
        let mut position = 0usize;
        Ok(prepared.map(|PreparedBuffer { generation, lease }| {
            let slot = unsafe { &*slots[position] };
            position += 1;
            // From reservation through all LIVE stores only validated indexing
            // and nonpanicking ownership moves. Never call user code here.
            unsafe {
                (*slot.lease.get()).write(BindingLease::Buffer(lease));
            }
            slot.state.store(LIVE, Ordering::Release);
            InstalledBuffer {
                generation,
                slot: std::ptr::from_ref(slot),
            }
        }))
    }
}
impl GenerationBindings {
    /// WHOLE N-buffer + one immutable plan install under original-ticket EDIT.
    /// Every clone/refusal and full N+1 credit preview precedes reservation. The
    /// writer/root is moved intact, not unlocked/reacquired between preparation
    /// and effects. No native operation occurs in this method.
    /// # Safety
    /// SAME original EDIT spans this install and all subsequent lowering. Plan
    /// originates from that operation's authentic catalog writer; all buffers,
    /// parameters, shader geometry/schema and capture mode were preflighted.
    pub(super) unsafe fn install_prepared_with_plan<const N: usize>(
        &self,
        prepared: [PreparedBuffer; N],
        plan: crate::metal_backend::CatalogPlanLease,
    ) -> Result<([InstalledBuffer; N], InstalledPlan), MetalKernelError> {
        for binding in &prepared {
            if !binding.matches(self) {
                return Err(refused("stale native binding generation"));
            }
        }
        let plan_generation = self.try_clone()?;
        let generation = self.head.captures();
        let start = generation.cursor.load(Ordering::Acquire);
        let used = |page: *mut BindingPage| {
            if page.is_null() {
                None
            } else {
                Some(unsafe { &*page }.next_index.load(Ordering::Relaxed))
            }
        };
        let next = |page: *mut BindingPage| {
            unsafe { &*page }
                .next()
                .map(|head| std::ptr::from_ref(head.captures()).cast_mut())
        };
        let buffers =
            crate::native_binding_preflight::plan_page_credits::<_, N>(start, PAGE, used, next)
                .ok_or_else(|| refused("native operation needs cold capacity refill"))?;
        // Preview an extra credit with the virtual used count after N. No
        // current cursor/index/state has been written, including cross-page
        // refusal at the plan credit. This avoids unstable generic N+1 arrays.
        let last_used = if let Some((_, index)) = buffers.slots.last() {
            index + 1
        } else {
            used(buffers.last_page).unwrap_or(0)
        };
        let tail = crate::native_binding_preflight::plan_page_credits::<_, 1>(
            buffers.last_page,
            PAGE,
            |page| {
                if page == buffers.last_page && N != 0 {
                    Some(last_used)
                } else {
                    used(page)
                }
            },
            next,
        )
        .ok_or_else(|| refused("native operation plan needs cold capacity refill"))?;
        // All ordinary errors ended. Sole EDIT freezes used indexes; cold append
        // is forward-only. Commit ALL N+1 RESERVED slots before payload moves.
        let slots = buffers.slots.map(|(page, index)| {
            let page = unsafe { &*page };
            page.next_index.store(index + 1, Ordering::Relaxed);
            let slot = &page.slots[index];
            slot.state.store(RESERVED, Ordering::Release);
            std::ptr::from_ref(slot)
        });
        let (page, index) = tail.slots[0];
        let page = unsafe { &*page };
        page.next_index.store(index + 1, Ordering::Relaxed);
        let plan_slot = &page.slots[index];
        plan_slot.state.store(RESERVED, Ordering::Release);
        generation.cursor.store(tail.last_page, Ordering::Release);
        // Only nonpanicking ownership moves/publication below. RESERVED is not
        // empty on unexpected unwind; page_ready retains initialized custody.
        unsafe {
            (*plan_slot.lease.get()).write(BindingLease::Plan(plan));
        }
        plan_slot.state.store(LIVE, Ordering::Release);
        let mut position = 0usize;
        let installed = prepared.map(|PreparedBuffer { generation, lease }| {
            let slot = unsafe { &*slots[position] };
            position += 1;
            unsafe {
                (*slot.lease.get()).write(BindingLease::Buffer(lease));
            }
            slot.state.store(LIVE, Ordering::Release);
            InstalledBuffer {
                generation,
                slot: std::ptr::from_ref(slot),
            }
        });
        Ok((
            installed,
            InstalledPlan {
                generation: plan_generation,
                slot: std::ptr::from_ref(plan_slot),
            },
        ))
    }
}
/// Prepared External custody, not yet installed and consuming no page credit.
/// Both fields have counted atoms-only final producer release contracts.
pub(crate) struct PreparedBuffer {
    generation: GenerationBindings,
    lease: NativeBufferLease,
}
impl PreparedBuffer {
    pub(super) fn matches(&self, generation: &GenerationBindings) -> bool {
        self.generation.same(generation)
    }
    pub(super) fn lease(&self) -> &NativeBufferLease {
        &self.lease
    }
    pub(super) fn try_clone(&self) -> Result<Self, MetalKernelError> {
        let generation = self.generation.try_clone()?;
        let lease = self
            .lease
            .try_clone()
            .map_err(|_| refused("binding pin clone refused"))?;
        Ok(Self { generation, lease })
    }
}
/// Authenticated installed native pin. SAME head roots its immutable LIVE slot.
pub(crate) struct InstalledBuffer {
    generation: GenerationBindings,
    slot: *const BindingSlot,
}
// SAFETY: immutable LIVE slot, authentic head roots its exact chain. Never
// acquired from pointer/geometry/weak identity. External borrow is counted.
unsafe impl Send for InstalledBuffer {}
unsafe impl Sync for InstalledBuffer {}
impl InstalledBuffer {
    pub(super) fn lease(&self) -> &NativeBufferLease {
        match unsafe { (*(*self.slot).lease.get()).assume_init_ref() } {
            BindingLease::Buffer(lease) => lease,
            BindingLease::Plan(_) => unreachable!("authenticated buffer slot changed role"),
        }
    }
}

/// Readonly installed plan. Authentic Head keeps exact LIVE slot and catalog
/// lease alive; never acquired from native pointer/shape/weak/raw Node identity.
pub(crate) struct InstalledPlan {
    generation: GenerationBindings,
    slot: *const BindingSlot,
}
impl InstalledPlan {
    pub(super) fn affine_shader(&self) -> &crate::kernels::kernels::affine::PreparedAffinePipeline {
        match unsafe { (*(*self.slot).lease.get()).assume_init_ref() } {
            BindingLease::Plan(plan) => plan.affine_shader(),
            BindingLease::Buffer(_) => unreachable!("authenticated plan slot changed role"),
        }
    }
}

const OPEN: u8 = 0;
const STAGING: u8 = 1;
const STAGED: u8 = 2;
const ENTERED: u8 = 3;
const RETURNED: u8 = 4;
const TERMINAL_SUBMITTED: u8 = 5;
const TERMINAL_CANCELLED: u8 = 6;
const QUARANTINED: u8 = 7;

struct UseWitness {
    // Clock is the Commands submission clock, NOT a guessed allocation domain.
    // It owns only atomics/identity, no Commands/namespace/page/pool backedge.
    clock: Arc<GpuClock>,
    phase: AtomicU8,
    ordered_callbacks: std::sync::atomic::AtomicBool,
    closed_epoch: std::sync::atomic::AtomicU64,
}
#[derive(Clone, Copy, Debug)]
enum WitnessError {
    WrongPhase,
    EpochExhausted,
    TerminalIrrevocable,
}

impl UseWitness {
    fn new(clock: Arc<GpuClock>) -> Self {
        Self {
            clock,
            phase: AtomicU8::new(OPEN),
            ordered_callbacks: std::sync::atomic::AtomicBool::new(false),
            closed_epoch: std::sync::atomic::AtomicU64::new(0),
        }
    }
    /// CommandsCore::submit_prepared supplies the exact commit_epoch return.
    /// Restage only while actual PREPARED admission excludes all native entry, before on_completion/native commit, with existing native
    /// prepare admission excluding concurrent transactions for this generation.
    fn stage_epoch(&self, epoch: Epoch) -> Result<(), WitnessError> {
        // The PhysicalRecord stores epoch+1. MAX is not silently wrapped to0.
        if epoch.0.checked_add(1).is_none() {
            return Err(WitnessError::EpochExhausted);
        }
        let previous = self.phase.load(Ordering::Acquire);
        if !matches!(previous, OPEN | STAGED) {
            return Err(WitnessError::WrongPhase);
        }
        // Existing unused PREPARED cancellation restores recording. A later
        // retry owns the exact admission and may replace an UNOBSERVED epoch.
        // No stage is emitted/observed before immutable terminal publication.
        self.phase
            .compare_exchange(previous, STAGING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| WitnessError::WrongPhase)?;
        // No callback/allocation/fallible/panicking point from claim to publish.
        self.closed_epoch.store(epoch.0, Ordering::Relaxed);
        // A concurrent quarantine may close STAGING. Never resurrect it by a
        // store; the initialized epoch word is ignored unless this CAS publishes.
        self.phase
            .compare_exchange(STAGING, STAGED, Ordering::Release, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| WitnessError::WrongPhase)
    }
    /// BEFORE first native queue entry, not after native status becomes visible.
    fn entered(&self) -> Result<(), WitnessError> {
        self.phase
            .compare_exchange(STAGED, ENTERED, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| WitnessError::WrongPhase)
    }
    /// AFTER every native segment/queue handoff returns. Not terminality.
    fn returned(&self) -> Result<(), WitnessError> {
        self.phase
            .compare_exchange(ENTERED, RETURNED, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| WitnessError::WrongPhase)
    }
    /// # Safety
    /// Native adapter has independent actual own-terminal proof AND returned
    /// handoff proof, not own status alone, FIFO inference or compute prefix for
    /// IO. Old native cached references are either cleared or still retain the
    /// SAME page head; every escaped recording/emission capability also retains
    /// that SAME head. Reused M4 slots never transfer an old witness to a new gen.
    unsafe fn terminal_submitted(&self) -> Result<(), WitnessError> {
        self.phase
            .compare_exchange(
                RETURNED,
                TERMINAL_SUBMITTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| WitnessError::WrongPhase)
    }
    /// # Safety
    /// No native handoff occurred, including no partial segment/IO entry; all
    /// native references/emission capabilities are cleared OR retain SAME head.
    /// A staged epoch can exist before callback registration fails: cancellation
    /// must NOT observe that epoch, since no callback may ever complete it.
    unsafe fn terminal_cancelled(&self) -> Result<(), WitnessError> {
        let mut previous = self.phase.load(Ordering::Acquire);
        loop {
            if !matches!(previous, OPEN | STAGED) {
                return Err(WitnessError::WrongPhase);
            }
            match self.phase.compare_exchange_weak(
                previous,
                TERMINAL_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(found) => previous = found,
            }
        }
    }
    /// Unknown/error/invariant exit publishes only atoms. No retry/resurrection
    /// and no native/charge release follows this marker. Existing roots retain.
    fn quarantine(&self) -> Result<(), WitnessError> {
        let mut previous = self.phase.load(Ordering::Acquire);
        loop {
            match previous {
                QUARANTINED => return Ok(()),
                // Terminal authorization is irrevocable. A later diagnostic
                // cannot invalidate a proven completed/cancelled native use.
                // Publish later errors elsewhere; no pretend retraction of a
                // cutoff already borrowed by service cleanup is possible.
                TERMINAL_SUBMITTED | TERMINAL_CANCELLED => {
                    return Err(WitnessError::TerminalIrrevocable);
                }
                OPEN | STAGING | STAGED | ENTERED | RETURNED => {}
                _ => return Err(WitnessError::WrongPhase),
            }
            match self.phase.compare_exchange_weak(
                previous,
                QUARANTINED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(found) => previous = found,
            }
        }
    }
    // Terminal phase is never rewritten. Once returned, this cutoff stays valid;
    // independent actual handoff/terminal/no-submit proof is not a revocable
    // optimistic observation. Escaped future emission still retains SAME head
    // and blocks page strong0. Page stamp mismatch is its own quarantine reason,
    // not a native terminality retraction, and blocks that page BEFORE C Drop.
    fn cutoff(&self) -> Option<Cutoff<'_>> {
        // M3 has no recyclable Slot dual-side token. Its actual ordered drain
        // publishes callbacks only after all arbitrary callbacks returned;
        // producer publishes RETURNED only after raw.commit returns. Both facts
        // are immutable, same-generation and atomic. This service-only join
        // cannot turn own data-ready feedback or a staged epoch into terminality.
        if self.phase.load(Ordering::Acquire) == RETURNED
            && self.ordered_callbacks.load(Ordering::Acquire)
            && self
                .phase
                .compare_exchange(
                    RETURNED,
                    TERMINAL_SUBMITTED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            self.clock
                .mark_completed(Epoch(self.closed_epoch.load(Ordering::Relaxed)));
        }
        match self.phase.load(Ordering::Acquire) {
            TERMINAL_SUBMITTED => Some(Cutoff::Submitted {
                clock: &self.clock,
                epoch: Epoch(self.closed_epoch.load(Ordering::Relaxed)),
            }),
            TERMINAL_CANCELLED => Some(Cutoff::NeverSubmitted),
            _ => None,
        }
    }
}
#[derive(Clone, Copy)]
enum Cutoff<'a> {
    Submitted {
        clock: &'a Arc<GpuClock>,
        epoch: Epoch,
    },
    NeverSubmitted,
}

/// Actual constructor guard only: until publish-last returns a usable wrapper,
/// this private generation has no encoder/native emission capability or handoff.
/// On refusal initialized anchor/record owners remain rooted, but no-use witness
/// may be cancelled. Never use this guard around an already-exposed generation.
pub(super) struct ConstructorCancellation(Option<GenerationBindings>);
impl ConstructorCancellation {
    pub(super) fn new(head: GenerationBindings) -> Self {
        Self(Some(head))
    }
    pub(super) fn finish(mut self) {
        self.0.take();
    }
}
impl Drop for ConstructorCancellation {
    fn drop(&mut self) {
        if let Some(head) = &self.0 {
            if unsafe { head.terminal_cancelled() }.is_err() {
                head.quarantine(Failure::AdapterFailed);
            }
        }
    }
}

/// Explicit initial/refill control storage; generations never retain this bank.
/// No native queue/Core/namespace backedge occurs in any header C payload.
pub(crate) struct BindingBank {
    first: InventoryNodeOwner,
    active: AtomicPtr<InventoryNode>,
    domain: crate::memory_admission::ModelDomain,
    queue_identity: usize,
}
struct InventoryNode {
    inventory: BindingInventory,
    // Release-CAS publishes sole owning Box; immutable until final Bank Drop.
    next: AtomicPtr<InventoryNode>,
    charge: Option<crate::memory_admission::PhysicalCharge>,
}
struct InventoryNodeOwner {
    node: Option<Box<InventoryNode>>,
}
impl InventoryNodeOwner {
    fn node(&self) -> &InventoryNode {
        self.node.as_deref().expect("owned inventory node")
    }
    fn publish_owner(mut self) {
        let _owned_by_prefix = Box::into_raw(self.node.take().expect("published node owner"));
    }
}
impl Drop for InventoryNodeOwner {
    fn drop(&mut self) {
        // A genuine Bank borrow excludes its final owner while a publisher or
        // reader is active. No pointer acquisition escapes. Final/cold cleanup
        // detaches immutable next ownership and uses constant stack cardinality.
        let mut node = self.node.take();
        while let Some(mut current) = node {
            let next = *current.next.get_mut();
            let charge = current.charge.take();
            drop(current); // deallocate Box BEFORE refunding its numeric receipt
            drop(charge);
            node = if next.is_null() {
                None
            } else {
                // SAFETY: sole next Box was transferred by successful link CAS;
                // no other owner destroys it, and final Bank excludes borrowers.
                Some(unsafe { Box::from_raw(next) })
            };
        }
    }
}
#[derive(Debug)]
pub(super) enum BindingPreparationError {
    Full,
    Root(crate::allocation_header_root::RootError),
    Adapter,
}
impl From<crate::allocation_header::HeaderError> for BindingPreparationError {
    fn from(error: crate::allocation_header::HeaderError) -> Self {
        Self::Root(crate::allocation_header_root::RootError::Header(error))
    }
}
impl From<crate::allocation_header_root::RootError> for BindingPreparationError {
    fn from(error: crate::allocation_header_root::RootError) -> Self {
        match error {
            crate::allocation_header_root::RootError::Full => Self::Full,
            other => Self::Root(other),
        }
    }
}
impl BindingPreparationError {
    fn native_error(self) -> MetalKernelError {
        match self {
            Self::Full => refused("binding header capacity needs cold admitted refill"),
            Self::Root(_) => refused("binding header root refused"),
            Self::Adapter => refused("binding counted preparation refused"),
        }
    }
}
impl BindingBank {
    /// Owned requested node/root/storage/registration/charge-control terms.
    /// Shared factory/native/allocator overhead remains separate, not RSS bound.
    pub(super) fn requested_bytes() -> Result<usize, MetalKernelError> {
        let roots = BindingInventory::requested_bytes()?;
        let control = crate::memory_admission::AllocationTicket::requested_charge_control_bytes()
            .map_err(|_| refused("binding node charge-control quote refused"))?;
        roots
            .checked_add(std::mem::size_of::<InventoryNode>())
            .and_then(|bytes| bytes.checked_add(control))
            .ok_or_else(|| refused("binding bank quote overflow"))
    }
    /// # Safety
    /// Existing inventory readiness/relay contracts apply. This bank is owned
    /// outside header C by actual Core/control; its final owner is service or
    /// cold rejected custody. Context/domain name the SAME base metadata payer.
    pub(super) unsafe fn cold_in(
        domain: crate::memory_admission::ModelDomain,
        context: &AllocationContext,
        queue_identity: usize,
        make: impl FnOnce(&AllocationContext) -> Result<BindingInventory, MetalKernelError>,
    ) -> Result<Self, MetalKernelError> {
        if context.domain_key() != domain.key() {
            return Err(refused("binding bank payer mismatch"));
        }
        let first = Self::cold_node(context, make)?;
        let active = first.node() as *const InventoryNode as *mut InventoryNode;
        Ok(Self {
            first,
            active: AtomicPtr::new(active),
            domain,
            queue_identity,
        })
    }
    fn cold_node(
        context: &AllocationContext,
        make: impl FnOnce(&AllocationContext) -> Result<BindingInventory, MetalKernelError>,
    ) -> Result<InventoryNodeOwner, MetalKernelError> {
        context
            .check_open()
            .map_err(|_| refused("binding bank context closed"))?;
        let bytes = std::mem::size_of::<InventoryNode>();
        let quote = bytes
            .checked_add(
                crate::memory_admission::AllocationTicket::requested_charge_control_bytes()
                    .map_err(|_| refused("binding node charge-control quote refused"))?,
            )
            .ok_or_else(|| refused("binding node quote overflow"))?;
        let mut ticket = context
            .take(quote)
            .map_err(|_| refused("binding bank node admission refused"))?;
        let inventory = make(context)?;
        // Shared cold exact-Layout allocator returns the COMPLETE initialized
        // input on null. Drop it offguard before its unused promise cancels.
        let mut node = match crate::cold_box::try_new(InventoryNode {
            inventory,
            next: AtomicPtr::new(std::ptr::null_mut()),
            charge: None,
        }) {
            Ok(node) => node,
            Err(rejected) => {
                drop(rejected);
                return Err(refused("binding bank node allocation refused"));
            }
        };
        let charge = match ticket.try_commit_with_control(bytes) {
            Ok(charge) => charge,
            Err(_) => {
                drop(node);
                return Err(refused("binding bank node charge refused"));
            }
        };
        drop(ticket);
        node.charge = Some(charge);
        Ok(InventoryNodeOwner { node: Some(node) })
    }
    fn current(&self) -> &InventoryNode {
        let ptr = self.active.load(Ordering::Acquire);
        // SAFETY: first + immutable next owners retain EVERY selected node.
        // Only this genuine bank borrow grants access; no raw acquisition API.
        // Cold append never unlinks, and active never selects an unpublished node.
        unsafe { &*ptr }
    }
    /// Call ONLY before recording EntryState/native acquisition. Ordinary warm
    /// prepare is atomic-only. Typed Full enters admitted cold refill; poison or
    /// counted refusal never allocates a replacement root or falls back unbound.
    pub(super) fn prepare_before_entry(&self) -> Result<PreparedGeneration, MetalKernelError> {
        self.prepare_with(|context, old| unsafe {
            BindingInventory::cold_in(
                context,
                old.clock.clone(),
                old.relay.clone(),
                old.queue_failure.clone(),
            )
        })
    }
    // Preservative factoring: the production cold factory remains identical;
    // manual scalar controls substitute only service selection for new roots.
    fn prepare_with(
        &self,
        make: impl FnOnce(
            &AllocationContext,
            &BindingInventory,
        ) -> Result<BindingInventory, MetalKernelError>,
    ) -> Result<PreparedGeneration, MetalKernelError> {
        let old = self.current();
        match old.inventory.try_prepare() {
            Ok(prepared) => return Ok(prepared),
            Err(BindingPreparationError::Full) => {}
            Err(error) => return Err(error.native_error()),
        }
        // Full is the explicit cold boundary. Reuse previously admitted slabs
        // before growing: scan only the finite immutable prefix ending at the
        // original active node, not concurrent future appends. This is NOT a
        // per-bind/healthy warm scan. Selecting an old SLAB issues fresh rooted
        // generation identity; it never resets a generation page cursor/header.
        let mut previous = self.first.node();
        while !std::ptr::eq(previous, old) {
            match previous.inventory.try_prepare() {
                Ok(prepared) => {
                    self.active.store(
                        previous as *const InventoryNode as *mut _,
                        Ordering::Release,
                    );
                    return Ok(prepared);
                }
                Err(BindingPreparationError::Full) => {}
                Err(error) => return Err(error.native_error()),
            }
            let next = previous.next.load(Ordering::Acquire);
            assert!(!next.is_null(), "active lies in owned prefix");
            // SAFETY: same Bank-owned immutable prefix as current().
            previous = unsafe { &*next };
        }
        // Cold capacity recovery, not warm claim: same process/base payer reserve
        // precedes node/root allocations. No supplied model is charged twice.
        crate::completion_scope::before_park(self.queue_identity)
            .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
        let pass = self
            .domain
            .reserve(Self::requested_bytes()?)
            .map_err(|_| refused("binding bank refill admission refused"))?;
        let context = pass.context();
        let fresh = Self::cold_node(&context, |context| make(context, &old.inventory))?;
        // Reserve this actor's credit BEFORE publishing its fresh node; no
        // competitor can exhaust it or cause unbounded speculative growth.
        let prepared = fresh
            .node()
            .inventory
            .try_prepare()
            .map_err(|e| e.native_error())?;
        let selected = fresh.node() as *const InventoryNode as *mut InventoryNode;
        let mut tail = old;
        loop {
            match tail.next.compare_exchange(
                std::ptr::null_mut(),
                selected,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Publication cannot unwind between CAS and ownership
                    // transfer; this bank borrow excludes final chain cleanup.
                    fresh.publish_owner();
                    break;
                }
                Err(next) => {
                    // A failed CAS observed a genuine successful node link.
                    // Carry ONE private candidate forward; no lock, wait, new
                    // allocation or second candidate occurs on contention.
                    tail = unsafe { &*next };
                }
            }
        }
        // Every failed CAS witnesses system progress. No starvation-freedom
        // claim under unbounded competing appenders; admission rejects before
        // native effects and current submitted custody stays recoverable.
        self.active.store(selected, Ordering::Release);
        Ok(prepared)
    }
    pub(super) unsafe fn cold_capacity(
        &self,
        context: &AllocationContext,
    ) -> Result<BindingCapacity, MetalKernelError> {
        unsafe { self.current().inventory.cold_capacity(context) }
    }
}

#[cfg(test)]
mod inventory_controls {
    use super::*;
    use crate::{
        completion_wake::Wake,
        memory_admission::{AccountSnapshot, ModelDomain},
        retirement_service::HostRetirementService,
    };

    fn make_inventory(
        context: &AllocationContext,
        service: &Arc<HostRetirementService>,
    ) -> BindingInventory {
        let failure = Arc::new(FailureSignal::default());
        let relay = Arc::new(DeferredFailureRelay::cold(
            failure.clone(),
            Arc::new(Wake::default()),
        ));
        make_same_inventory(context, service, Arc::new(GpuClock::new()), relay, failure)
    }
    fn make_same_inventory(
        context: &AllocationContext,
        service: &Arc<HostRetirementService>,
        clock: Arc<GpuClock>,
        relay: Arc<DeferredFailureRelay>,
        failure: Arc<FailureSignal>,
    ) -> BindingInventory {
        // SAFETY: no native lease or handoff is created in these controls. All
        // generations are cancelled through actual PreparedGeneration Drop;
        // escaped counted heads stay owned through every inspected reference.
        unsafe {
            BindingInventory {
                witnesses: HeaderPool::cold_owned_when_in_service(
                    context,
                    |_, _| true,
                    Some(relay.clone()),
                    service,
                )
                .unwrap(),
                heads: HeaderPool::cold_owned_when_in_service(
                    context,
                    generation_ready,
                    Some(relay.clone()),
                    service,
                )
                .unwrap(),
                first_pages: HeaderPool::cold_owned_when_in_service(
                    context,
                    page_ready,
                    Some(relay.clone()),
                    service,
                )
                .unwrap(),
                clock,
                relay,
                queue_failure: failure,
            }
        }
    }
    fn drain(service: &Arc<HostRetirementService>) {
        // Each pass retires at most one cell per root. At most two SLAB banks
        // (six roots) exist; cascading Gen->Page->Witness adds two passes.
        for _ in 0..(SLAB * 3 + 8) {
            service.manual_pass();
        }
    }
    fn bank(capacity: usize) -> (BindingBank, Arc<HostRetirementService>, ModelDomain) {
        let service = HostRetirementService::manual();
        let quote = BindingBank::requested_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote.checked_mul(capacity).unwrap());
        let pass = domain.reserve(quote).unwrap();
        let context = pass.context();
        // SAFETY: actual bank/owned node and all three rooted inventories, but
        // no native Queue/CB/emission. Explicit zero identity never parks.
        let bank = unsafe {
            BindingBank::cold_in(domain.clone(), &context, 0, |c| {
                Ok(make_inventory(c, &service))
            })
        }
        .unwrap();
        drop(context);
        drop(pass);
        (bank, service, domain)
    }
    fn prepare(
        bank: &BindingBank,
        service: &Arc<HostRetirementService>,
    ) -> Result<PreparedGeneration, MetalKernelError> {
        bank.prepare_with(|context, old| {
            Ok(make_same_inventory(
                context,
                service,
                old.clock.clone(),
                old.relay.clone(),
                old.queue_failure.clone(),
            ))
        })
    }
    #[test]
    fn native_entry_requires_owner_staging_without_taking_an_epoch() {
        let (bank, service, domain) = bank(1);
        let prepared = prepare(&bank, &service).unwrap();
        let head = prepared.constructor_head().unwrap();
        let clock = head.head.captures().witness.captures().clock.clone();
        for _ in 0..2 {
            assert!(matches!(
                head.require_staged(),
                Err(MetalKernelError::OwnerSubmissionRequired)
            ));
            assert_eq!(clock.current_epoch().0, 0);
        }
        let epoch = clock.commit_epoch();
        head.stage_epoch(epoch).unwrap();
        head.require_staged().unwrap();
        head.require_staged().unwrap();
        assert_eq!(clock.current_epoch().0, 1);
        assert!(
            head.stage_epoch(epoch).is_err(),
            "owner staging cannot replay"
        );
        drop(prepared);
        drop(head);
        drain(&service);
        drop(bank);
        drain(&service);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn inventory_first_page_unconsumed_cancel_returns_credit() {
        let (bank, service, domain) = bank(1);
        let prepared = prepare(&bank, &service).unwrap();
        let escaped = prepared.constructor_head().unwrap();
        let generation = escaped.head.captures();
        let first = generation.first().expect("initial prepared page");
        assert_eq!(first.captures().next_index.load(Ordering::Acquire), 0);
        assert!(first
            .captures()
            .slots
            .iter()
            .all(|slot| slot.state.load(Ordering::Acquire) == EMPTY));
        drop(prepared); // Actual cancellation body, never-consumed first page.
        assert!(matches!(
            generation.witness.captures().cutoff(),
            Some(Cutoff::NeverSubmitted)
        ));
        drop(escaped);
        drain(&service);
        let successor = prepare(&bank, &service).expect("cancelled first-page credit returns");
        drop(successor);
        drain(&service);
        drop(bank);
        drain(&service);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn inventory_full_grows_then_reuses_with_escaped_old_heads() {
        let (bank, service, domain) = bank(2);
        let first = bank.active.load(Ordering::Acquire);
        let mut escaped = Vec::with_capacity(SLAB);
        for _ in 0..SLAB {
            let prepared = prepare(&bank, &service).unwrap();
            escaped.push(prepared.constructor_head().unwrap());
            drop(prepared); // Cancelled, but authentic escaped head retains credit.
        }
        let replacement = prepare(&bank, &service).expect("Full admits new slab");
        let second = bank.active.load(Ordering::Acquire);
        assert_ne!(
            first, second,
            "Full must not reuse an escaped live generation"
        );
        assert_eq!(unsafe { (*first).next.load(Ordering::Acquire) }, second);
        drop(replacement);
        drain(&service);
        // Saturate second root while all old heads are retained; no third slab.
        let mut second_heads = Vec::with_capacity(SLAB);
        for _ in 0..SLAB {
            let prepared = prepare(&bank, &service).unwrap();
            second_heads.push(prepared.constructor_head().unwrap());
            drop(prepared);
        }
        escaped.clear();
        drain(&service);
        let reused = prepare(&bank, &service).expect("Full reuses returned earlier root");
        assert_eq!(
            bank.active.load(Ordering::Acquire),
            first,
            "returned old slab is reused before reserve"
        );
        drop(reused);
        drop(second_heads);
        drop(escaped);
        drain(&service);
        drop(bank);
        drain(&service);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn inventory_initial_and_two_lookahead_are_distinct_counted_pairs() {
        let (bank, service, domain) = bank(1);
        let initial = prepare(&bank, &service).unwrap();
        let recovery = prepare(&bank, &service).unwrap();
        let cut = prepare(&bank, &service).unwrap();
        let a = initial.constructor_head().unwrap();
        let b = recovery.constructor_head().unwrap();
        let c = cut.constructor_head().unwrap();
        assert!(
            !a.same(&b) && !b.same(&c) && !a.same(&c),
            "two lookahead credits must not alias initial or each other"
        );
        // These are real preparations, not native replacement execution.
        drop(initial);
        drop(recovery);
        drop(cut);
        for head in [&a, &b, &c] {
            assert!(matches!(
                head.head.captures().witness.captures().cutoff(),
                Some(Cutoff::NeverSubmitted)
            ));
        }
        drop((a, b, c));
        drain(&service);
        drop(bank);
        drain(&service);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn inventory_refill_refusal_retains_escaped_generation_and_active_root() {
        let (bank, service, domain) = bank(1);
        let original = bank.active.load(Ordering::Acquire);
        let mut escaped = Vec::with_capacity(SLAB);
        for _ in 0..SLAB {
            let prepared = prepare(&bank, &service).unwrap();
            escaped.push(prepared.constructor_head().unwrap());
            drop(prepared);
        }
        let before = domain.snapshot();
        let attempts = std::cell::Cell::new(0);
        let refused = bank.prepare_with(|context, old| {
            attempts.set(attempts.get() + 1);
            Ok(make_same_inventory(
                context,
                &service,
                old.clock.clone(),
                old.relay.clone(),
                old.queue_failure.clone(),
            ))
        });
        assert!(refused.is_err(), "tight cap must refuse cold refill");
        assert_eq!(
            attempts.get(),
            0,
            "refill admission refusal precedes inventory/native construction"
        );
        assert_eq!(bank.active.load(Ordering::Acquire), original);
        assert!(
            unsafe { (*original).next.load(Ordering::Acquire) }.is_null(),
            "refusal cannot publish an unbound fallback node"
        );
        assert_eq!(domain.snapshot(), before);
        assert!(escaped.iter().all(|head| matches!(
            head.head.captures().witness.captures().cutoff(),
            Some(Cutoff::NeverSubmitted)
        )));
        drop(escaped);
        drain(&service);
        drop(bank);
        drain(&service);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
}
