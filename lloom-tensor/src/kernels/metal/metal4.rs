//! Native Metal 4 submission lifetime. Native buffers/allocators are recycled;
//! a logical submission keeps its own completion state so old markers cannot
//! start observing a later use of the same native object.
use super::Device;
use crate::completion_wake::Wake;
use crate::kernels::MetalKernelError;
use block2::RcBlock;
use crossbeam_queue::ArrayQueue;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::*;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

const SLOTS: usize = 8;
const PAGE_BYTES: usize = 64 * 1024;

// Feedback initialization alone is not readiness: a reader could observe the
// OnceLock before Error is published and misclassify the failed submission.
// Both native completion writers initialize feedback before this terminal
// Release; all data-wait readers acquire this same publication boundary.
fn publish_terminal_status(status: &AtomicUsize, failed: bool) {
    status.store(
        if failed {
            MTLCommandBufferStatus::Error.0
        } else {
            MTLCommandBufferStatus::Completed.0
        },
        Ordering::Release,
    );
}

fn terminal_data_ready(status: &AtomicUsize) -> bool {
    let observed = MTLCommandBufferStatus(status.load(Ordering::Acquire));
    observed == MTLCommandBufferStatus::Completed || observed == MTLCommandBufferStatus::Error
}

type Callback = Box<dyn Fn(&Feedback) + Send + Sync>;

pub(crate) struct Feedback {
    pub error: Option<String>,
    pub start: f64,
    pub end: f64,
}

pub(crate) struct Queue {
    pub raw: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    device: Device,
    available: ArrayQueue<Slot>,
    allocated: AtomicUsize,
    ready: Arc<Wake>,
    failure: Arc<crate::retirement_owners::FailureSignal>,
    retirement: super::retirement::Registry,
    completions: OnceLock<Arc<super::retirement::Metal4Completions>>,
    commit_order: Mutex<u64>,
    acquire_parks: AtomicUsize,
    ledger: OnceLock<Arc<super::ledger::Ledger>>,
    pub created: AtomicUsize,
    pub reused: AtomicUsize,
    /// The per-slot sets' counts, shared with every slot (`SlotTrace`).
    slot_trace: Arc<SlotTrace>,
    /// What a command buffer declares (`kernels.residency.metal4`).
    declare: super::Metal4Declare,
    /// The device's sets this queue declares with: on the queue (`queue-*`)
    /// or used by every command buffer (`cb-*`). Installed by `Commands`
    /// before the first acquisition; a queue without them lists everything
    /// in its slot sets whatever the arm.
    device_sets: OnceLock<Arc<super::ResidencySet>>,
    /// The host retirement service's Wake, notified when a slot becomes free
    /// while a final owner that found none is waiting for one
    /// (`slot_waiter`): it returned pending and is polled again on this.
    slot_wake: OnceLock<Arc<Wake>>,
    slot_waiter: AtomicBool,
    // LAST: raw/available slots disappear before final capability publication.
    physical: crate::retirement_service::PhysicalAnchor<
        crate::metal_backend::device_retirement::NamespaceOwner,
    >,
}
unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl Queue {
    pub fn new(device: Device) -> Result<Arc<Self>, MetalKernelError> {
        crate::tuning::try_tuning()
            .map_err(|error| MetalKernelError::InvalidInput(error.to_string()))?;
        let raw = device
            .as_ref()
            .newMTL4CommandQueue()
            .ok_or_else(|| MetalKernelError::FailedToCreateResource("MTL4CommandQueue".into()))?;
        let failure = Arc::new(crate::retirement_owners::FailureSignal::default());
        let ready = Arc::new(Wake::default());
        let retirement = super::retirement::registry(failure.clone(), ready.clone())
            .map_err(|_| MetalKernelError::QueueUnavailable("retirement service unavailable"))?;
        let queue = Arc::new(Self {
            raw,
            device,
            available: ArrayQueue::new(SLOTS),
            allocated: AtomicUsize::new(0),
            ready,
            failure: failure.clone(),
            retirement,
            completions: OnceLock::new(),
            commit_order: Mutex::new(0),
            acquire_parks: AtomicUsize::new(0),
            ledger: OnceLock::new(),
            created: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            slot_trace: Arc::new(SlotTrace {
                on: crate::tuning::tuning().metal.residency_slot_trace,
                ..SlotTrace::default()
            }),
            declare: super::Metal4Declare::selected(),
            device_sets: OnceLock::new(),
            slot_wake: OnceLock::new(),
            slot_waiter: AtomicBool::new(false),
            physical: crate::retirement_service::PhysicalAnchor::default(),
        });
        let core = super::retirement::Metal4Completions::new(
            Arc::as_ptr(&queue) as usize,
            failure,
            queue.ready.clone(),
        );
        assert!(queue.completions.set(Arc::new(core)).is_ok());
        Ok(queue)
    }

    pub(super) fn attach_physical_owner(
        &self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        self.physical.install(owner).map_err(|owner| {
            crate::kernels::err::CompletionRegistrationError::new(
                MetalKernelError::QueueUnavailable("M4 physical owner already installed"),
                Some(owner),
            )
            .into()
        })
    }

    /// Acquisition and every submission wait use the same Wake. Poison closes
    /// admission without claiming native completion or releasing any payload.
    pub(super) fn check_failure(&self) -> Result<(), MetalKernelError> {
        if self.failure.get().is_some() {
            Err(MetalKernelError::QueueUnavailable(
                "retirement queue failed",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn poison(&self, reason: crate::retirement_owners::Failure) {
        self.failure.publish(reason, || self.ready.notify());
    }

    pub(super) fn return_retirement_slot(&self, slot: Slot) -> Result<(), Slot> {
        self.available.push(slot)?;
        self.ready.notify();
        self.notify_slot_wake();
        Ok(())
    }

    pub(super) fn discard_retirement_slot(&self, slot: Slot) {
        drop(slot);
        self.allocated.fetch_sub(1, Ordering::AcqRel);
        self.ready.notify();
        self.notify_slot_wake();
    }

    /// Cold, once: the Wake a freed slot notifies besides `ready`.
    pub(super) fn install_slot_wake(&self, wake: Arc<Wake>) -> Result<(), Arc<Wake>> {
        match self.slot_wake.get() {
            Some(installed) if Arc::ptr_eq(installed, &wake) => Ok(()),
            Some(_) => Err(wake),
            None => self.slot_wake.set(wake),
        }
    }

    /// After the slot was made free: the free, then the look at the waiter,
    /// against the waiter's flag, then its look at the slots
    /// (`free_slot_or_wait`), so one of the two sees the other.
    fn notify_slot_wake(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.slot_waiter.load(Ordering::Relaxed) {
            if let Some(wake) = self.slot_wake.get() {
                wake.notify();
            }
        }
    }

    fn has_free_slot(&self) -> bool {
        !self.available.is_empty() || self.allocated.load(Ordering::Acquire) < SLOTS
    }

    /// Whether an acquisition would find a slot without parking. When it
    /// would not, the next freed slot notifies the installed slot Wake. Only
    /// an owner with no concurrent acquirer may act on `true` (the final
    /// owner, which never waits for a slot).
    pub(super) fn free_slot_or_wait(&self) -> bool {
        if self.has_free_slot() {
            return true;
        }
        self.slot_waiter.store(true, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        self.has_free_slot()
    }

    /// What this queue's command buffers declare (`kernels.residency.metal4`).
    pub(super) fn declare(&self) -> super::Metal4Declare {
        self.declare
    }

    /// The device's sets this queue declares with, once the caller has put
    /// them on the queue (`queue-*`) or for every command buffer to use
    /// (`cb-*`). Once, before the first acquisition: a slot reads it when it
    /// is made.
    pub(super) fn use_device_sets(&self, sets: Arc<super::ResidencySet>) {
        assert!(
            self.device_sets.set(sets).is_ok(),
            "Metal 4 device sets installed once, before the first command buffer"
        );
    }

    pub fn attach_ledger(&self, ledger: Arc<super::ledger::Ledger>) {
        assert!(
            self.ledger.set(ledger).is_ok(),
            "Metal 4 ledger installed once at device construction"
        );
    }

    /// The completion hook every flagged submission of this queue runs; once,
    /// before the first commit.
    pub(super) fn install_hook(
        &self,
        hook: Arc<super::completion_hook::CompletionHook>,
    ) -> Result<(), MetalKernelError> {
        self.completions
            .get()
            .expect("queue initialized")
            .install_hook(hook)
    }

    pub fn acquire(self: &Arc<Self>) -> Result<Arc<Submission>, MetalKernelError> {
        self.acquire_impl(None, &mut Self::park)
    }

    /// The default park of an acquisition that found no free slot.
    fn park(ready: &Wake, seen: usize) -> Result<(), MetalKernelError> {
        ready.wait(seen);
        Ok(())
    }

    /// `(allocated, available)` native slots: a test's leak observation.
    #[cfg(test)]
    pub(super) fn slots_for_test(&self) -> (usize, usize) {
        (self.allocated.load(Ordering::Acquire), self.available.len())
    }

    /// A thread is parked on this queue's slot wake: a test's observation of
    /// an owner parked for a slot behind held work.
    #[cfg(test)]
    pub(super) fn ready_parked_for_test(&self) -> bool {
        self.ready.test_is_parked()
    }

    /// Wake an acquisition parked for a slot, so it re-runs its park hook: a
    /// producer that released held recordings the parked owner must submit
    /// before any slot can come back (`Commands::produced`).
    pub(super) fn notify_ready(&self) {
        self.ready.notify();
    }

    /// Signal `event` to `value` on the queue itself, after every submission
    /// committed so far: a signal that belonged to a recording cancelled
    /// before submission, re-homed so the work ordered behind it still runs
    /// after everything it was ordered after.
    pub(super) fn signal_after_committed(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        let _order = self.commit_order.lock();
        self.check_failure()?;
        self.raw
            .signalEvent_value(ProtocolObject::from_ref(event), value);
        Ok(())
    }

    /// Cold/preflight constructor. Actual Queue failure/Wake identities are
    /// preheld in the rooted relay. No generation is silently inferred from
    /// a raw Buffer or a namespace; caller carries the genuine submission clock.
    /// This additive route is not yet enabled by default/scoped factories.
    pub(super) unsafe fn cold_binding_inventory(
        &self,
        context: &crate::memory_admission::AllocationContext,
        clock: Arc<super::buffer_pool::GpuClock>,
    ) -> Result<super::native_generation_bindings::BindingInventory, MetalKernelError> {
        let relay = Arc::new(crate::deferred_failure_relay::DeferredFailureRelay::cold(
            self.failure.clone(),
            self.ready.clone(),
        ));
        unsafe {
            super::native_generation_bindings::BindingInventory::cold_in(
                context,
                clock,
                relay,
                self.failure.clone(),
            )
        }
    }

    pub(super) fn acquire_bound(
        self: &Arc<Self>,
        prepared: super::native_generation_bindings::PreparedGeneration,
    ) -> Result<Arc<Submission>, MetalKernelError> {
        self.acquire_with(Some(prepared), &mut Self::park)
    }

    /// Acquire a recording, running `park` instead of the plain wait whenever
    /// no slot is free. An owner that holds recordings back from submission
    /// (`Commands::commit_when_produced`) submits the released ones there:
    /// their slots come back only once they have run.
    pub(super) fn acquire_with(
        self: &Arc<Self>,
        prepared: Option<super::native_generation_bindings::PreparedGeneration>,
        park: &mut dyn FnMut(&Wake, usize) -> Result<(), MetalKernelError>,
    ) -> Result<Arc<Submission>, MetalKernelError> {
        if prepared
            .as_ref()
            .is_some_and(|prepared| !prepared.matches_queue(&self.failure))
        {
            return Err(MetalKernelError::QueueUnavailable(
                "binding inventory belongs to another native queue",
            ));
        }
        self.acquire_impl(prepared, park)
    }

    fn acquire_impl(
        self: &Arc<Self>,
        prepared: Option<super::native_generation_bindings::PreparedGeneration>,
        park: &mut dyn FnMut(&Wake, usize) -> Result<(), MetalKernelError>,
    ) -> Result<Arc<Submission>, MetalKernelError> {
        // Obtain the real default pin before removing any reusable/native slot.
        // Counted-clone refusal leaves queue state/capacity unchanged. Initial
        // cold acquire has None; pristine attachment fills that State below.
        self.check_failure()?;
        let physical = self
            .physical
            .try_owner(crate::metal_backend::device_retirement::NamespaceOwner::try_clone)?;
        let mut slot = loop {
            let seen = self.ready.snapshot();
            self.check_failure()?;
            if let Some(slot) = self.available.pop() {
                self.reused.fetch_add(1, Ordering::Relaxed);
                break slot;
            }
            if self
                .allocated
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    (n < SLOTS).then_some(n + 1)
                })
                .is_ok()
            {
                match Slot::new(self) {
                    Ok(slot) => {
                        self.created.fetch_add(1, Ordering::Relaxed);
                        break slot;
                    }
                    Err(e) => {
                        self.allocated.fetch_sub(1, Ordering::AcqRel);
                        self.ready.notify();
                        self.notify_slot_wake();
                        return Err(e);
                    }
                }
            }
            crate::completion_scope::before_park(Arc::as_ptr(self) as usize)
                .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
            self.acquire_parks.fetch_add(1, Ordering::Relaxed);
            park(&self.ready, seen)?;
        };
        // Only completed submissions return slots; abandoned uncommitted ones
        // are destroyed instead. No reset races outstanding GPU commands.
        slot.reset();
        let front_event = slot.front_event.clone();
        let generation = slot.generation;
        // Capacity reservation precedes Commands' epoch/callback side effects.
        // The explicit retained quota permit is an integration prerequisite;
        // this source draft still has the earlier uncharged growth hook.
        let retirement = match self.retirement.prepare(
            super::retirement::Captures::new(Vec::new()),
            super::retirement::NativeOwners::vacant(),
            |_| Ok(()),
        ) {
            Ok(record) => record,
            Err(_rejected) => {
                self.discard_retirement_slot(slot);
                return Err(MetalKernelError::QueueUnavailable(
                    "retirement metadata unavailable",
                ));
            }
        };
        // Cold/native failures above leave preparation unconsumed; its Drop
        // cancels the genuine never-attached witness. After this move no fallible
        // operation occurs before BOTH immutable Submission and native Slot own
        // their exact heads. Escaped encoders keep Submission, never Queue.current.
        let bindings = prepared.map(|prepared| {
            let (submission, native) = prepared.take();
            slot.bindings = Some(native);
            submission
        });
        Ok(Arc::new(Submission {
            front_event,
            generation,
            queue: Arc::clone(self),
            retirement,
            native_reuse: crate::native_reuse::NativeReuse::new(),
            command_lifecycle: crate::command_lifecycle::CommandLifecycle::new(),
            committed_events: OnceLock::new(),
            status: AtomicUsize::new(MTLCommandBufferStatus::NotEnqueued.0),
            state: Mutex::new(State {
                slot: Some(slot),
                callbacks: Vec::new(),
                hook_runs: 0,
                events: Vec::new(),
                physical,
            }),
            feedback: OnceLock::new(),
            callbacks_finished: AtomicBool::new(false),
            done: self.ready.clone(),
            bindings,
        }))
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        eprintln!(
            "[metal4] native_buffers_created={} native_buffers_reused={} acquire_parks={}",
            self.created.load(Ordering::Relaxed),
            self.reused.load(Ordering::Relaxed),
            self.acquire_parks.load(Ordering::Relaxed)
        );
    }
}

/// What the per-slot residency sets do, counted and timed only when
/// `kernels.residency.slot_trace` is on: one load per bind or commit when off.
#[derive(Default)]
struct SlotTrace {
    on: bool,
    adds: std::sync::atomic::AtomicU64,
    add_ns: std::sync::atomic::AtomicU64,
    pipeline_adds: std::sync::atomic::AtomicU64,
    covered: std::sync::atomic::AtomicU64,
    set_uses: std::sync::atomic::AtomicU64,
    commits: std::sync::atomic::AtomicU64,
    commit_ns: std::sync::atomic::AtomicU64,
    commit_max_ns: std::sync::atomic::AtomicU64,
    last_bytes: std::sync::atomic::AtomicU64,
    max_bytes: std::sync::atomic::AtomicU64,
    /// `[listing][class column][adds, bytes]` (`SlotResidency::census`).
    census: [[[std::sync::atomic::AtomicU64; 2]; crate::residency_class::CENSUS_CLASSES];
        crate::residency_class::SlotListing::ALL.len()],
    /// The census by `(listing, class column, listing site)`.
    sites: Mutex<HashMap<(u8, u8, u32), crate::residency_class::CensusCell>>,
    /// The device's sets, read for the class and site a binding is listed
    /// under.
    sets: OnceLock<Arc<super::ResidencySet>>,
}

/// How a slot met a resource: bound by address as a buffer, a constant page
/// of its own, or named through a raw path (a blit operand, an ICB run's
/// `useResources`, the ICB itself).
#[derive(Clone, Copy, PartialEq, Eq)]
enum BindKind {
    Buffer,
    Page,
    Raw,
}

impl SlotTrace {
    /// Counts `resource`, met for the first time in a generation, in the
    /// census: why it was or was not left to the device's sets, the class
    /// they list it under, its bytes, and the site that listed it. The
    /// instrument only: takes the device set's lock and a map lock.
    fn census(
        &self,
        resource: &ProtocolObject<dyn MTLResource>,
        listing: super::DeviceListing,
        kind: BindKind,
    ) {
        use crate::residency_class::{census_column, ResidencyClass, SlotListing};
        let key = resource as *const _ as usize;
        let placed = self.sets.get().and_then(|sets| sets.listing_of(key));
        let why = match (kind, listing) {
            (BindKind::Page, _) => SlotListing::Page,
            (BindKind::Raw, _) => SlotListing::Raw,
            (BindKind::Buffer, super::DeviceListing::Pool) => SlotListing::Covered,
            (BindKind::Buffer, super::DeviceListing::Held) => SlotListing::Held,
            (BindKind::Buffer, super::DeviceListing::No) if placed.is_some() => SlotListing::Listed,
            (BindKind::Buffer, super::DeviceListing::No) => SlotListing::Unlisted,
        };
        let class = placed
            .map(|(class, _)| class)
            .or_else(|| resource.heap().map(|_| ResidencyClass::HeapChunk));
        let (row, col) = (why as usize, census_column(class));
        let bytes = MTLResource::allocatedSize(resource) as u64;
        self.census[row][col][0].fetch_add(1, Ordering::Relaxed);
        self.census[row][col][1].fetch_add(bytes, Ordering::Relaxed);
        let site = placed.map_or(0, |(_, site)| site);
        let mut sites = self.sites.lock();
        let cell = sites.entry((row as u8, col as u8, site)).or_default();
        cell.adds += 1;
        cell.bytes += bytes;
    }
}

impl Queue {
    /// The per-slot sets' counts (`kernels.residency.slot_trace`); `None`
    /// when the instrument is off.
    pub(crate) fn slot_residency(&self) -> Option<super::SlotResidency> {
        let t = &self.slot_trace;
        t.on.then(|| super::SlotResidency {
            adds: t.adds.load(Ordering::Relaxed),
            add_ns: t.add_ns.load(Ordering::Relaxed),
            pipeline_adds: t.pipeline_adds.load(Ordering::Relaxed),
            covered: t.covered.load(Ordering::Relaxed),
            set_uses: t.set_uses.load(Ordering::Relaxed),
            commits: t.commits.load(Ordering::Relaxed),
            commit_ns: t.commit_ns.load(Ordering::Relaxed),
            commit_max_ns: t.commit_max_ns.load(Ordering::Relaxed),
            last_bytes: t.last_bytes.load(Ordering::Relaxed),
            max_bytes: t.max_bytes.load(Ordering::Relaxed),
            census: std::array::from_fn(|row| {
                std::array::from_fn(|col| crate::residency_class::CensusCell {
                    adds: t.census[row][col][0].load(Ordering::Relaxed),
                    bytes: t.census[row][col][1].load(Ordering::Relaxed),
                })
            }),
        })
    }

    /// The slot census by the site that listed each allocation, most
    /// resources first; empty when the instrument is off. Symbolizes each
    /// site once.
    pub(crate) fn slot_sites(&self) -> Vec<crate::residency_class::SlotSite> {
        use crate::residency_class::{ResidencyClass, SlotListing, SlotSite};
        if !self.slot_trace.on {
            return Vec::new();
        }
        let cells: Vec<_> = self
            .slot_trace
            .sites
            .lock()
            .iter()
            .map(|(&k, &c)| (k, c))
            .collect();
        let mut out: Vec<SlotSite> = cells
            .into_iter()
            .map(|((row, col, site), cell)| SlotSite {
                listing: SlotListing::ALL[row as usize],
                class: ResidencyClass::ALL.get(col as usize).copied(),
                frames: super::residency_set::listing_sites::frames(site),
                cell,
            })
            .collect();
        out.sort_by(|a, b| b.cell.adds.cmp(&a.cell.adds).then(a.frames.cmp(&b.frames)));
        out
    }

    /// The device's sets the slot census reads classes and sites from.
    pub(super) fn census_sets(&self, sets: Arc<super::ResidencySet>) {
        let _ = self.slot_trace.sets.set(sets);
    }

    /// Commits a slot's set, timing the commit when the instrument is on.
    fn commit_slot_set(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        let t = &self.slot_trace;
        if !t.on {
            set.commit();
            return;
        }
        let start = std::time::Instant::now();
        set.commit();
        let ns = start.elapsed().as_nanos() as u64;
        let bytes = set.allocatedSize();
        t.commits.fetch_add(1, Ordering::Relaxed);
        t.commit_ns.fetch_add(ns, Ordering::Relaxed);
        t.commit_max_ns.fetch_max(ns, Ordering::Relaxed);
        t.last_bytes.store(bytes, Ordering::Relaxed);
        t.max_bytes.fetch_max(bytes, Ordering::Relaxed);
    }
}

pub(super) struct Slot {
    front_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    generation: u64,
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    // Retained across safe slot reuse; only the current generation's prefix is
    // encoded. No allocator reset until every segment and handoff have finished.
    extra_cbs: Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    segment: usize,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    trace: Arc<SlotTrace>,
    /// Leave a binding the device's sets cover out of this slot's set.
    diff: bool,
    /// The device's sets, used by each command buffer (`cb-*`), and the
    /// router generation `used` was read at.
    device: Option<Arc<super::ResidencySet>>,
    used: Vec<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    used_generation: u64,
    resources: Vec<Retained<ProtocolObject<dyn MTLResource>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    fences: Vec<Retained<ProtocolObject<dyn MTLFence>>>,
    seen: HashSet<usize>,
    pages: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    page_reservations: Vec<super::ledger::Reservation>,
    page: usize,
    offset: usize,
    // LAST: release native resources/allocator/table before root publication.
    namespaces: Vec<crate::metal_backend::device_retirement::NamespaceOwner>,
    // LAST: exact old generation remains through native cached-reference reset
    // and through final Slot native/resource destruction. Header release atoms
    // cannot run arbitrary native/charge Drop on this producer.
    bindings: Option<super::native_generation_bindings::GenerationBindings>,
}
unsafe impl Send for Slot {}

impl Slot {
    fn new(queue: &Queue) -> Result<Self, MetalKernelError> {
        let device = queue.device.as_ref();
        let fail = |name: &str| MetalKernelError::FailedToCreateResource(name.into());
        let cb = device
            .newCommandBuffer()
            .ok_or_else(|| fail("MTL4CommandBuffer"))?;
        let allocator = device
            .newCommandAllocator()
            .ok_or_else(|| fail("MTL4CommandAllocator"))?;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(31);
        desc.setInitializeBindings(true);
        let table = device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| fail(&format!("MTL4ArgumentTable: {e}")))?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| fail(&format!("MTL4 residency: {e}")))?;
        let front_event = device
            .newSharedEvent()
            .ok_or_else(|| fail("MTL4 queue-front event"))?;
        queue.raw.addResidencySet(&residency);
        Ok(Self {
            front_event,
            generation: 0,
            cb,
            extra_cbs: Vec::new(),
            segment: 0,
            allocator,
            table,
            residency,
            queue: queue.raw.clone(),
            trace: queue.slot_trace.clone(),
            // A covered binding is left out only when the device's sets are
            // wired into this queue's command buffers.
            diff: queue.declare.diff() && queue.device_sets.get().is_some(),
            device: queue
                .declare
                .per_command_buffer()
                .then(|| queue.device_sets.get().cloned())
                .flatten(),
            used: Vec::new(),
            used_generation: u64::MAX,
            resources: Vec::with_capacity(256),
            pipelines: Vec::with_capacity(64),
            fences: Vec::with_capacity(64),
            seen: HashSet::with_capacity(512),
            pages: Vec::new(),
            page_reservations: Vec::new(),
            page: 0,
            offset: 0,
            namespaces: Vec::new(),
            bindings: None,
        })
    }

    fn reset(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("MTL4 generation exhausted");
        self.residency.removeAllAllocations();
        self.resources.clear();
        self.pipelines.clear();
        self.fences.clear();
        self.seen.clear();
        self.page = 0;
        self.offset = 0;
        for index in 0..31 {
            // SAFETY: table was allocated with 31 buffer bindings.
            unsafe {
                self.table.setAddress_atIndex(0, index);
            }
        }
        self.allocator.reset();
        self.segment = 0;
        self.cb.beginCommandBufferWithAllocator(&self.allocator);
        // Native cached references of the prior generation were cleared/reset.
        // Persistent scratch pages retain their own native reservation owners.
        self.namespaces.clear();
        // Only actual dual-sided completion returns Slot to the available pool.
        // Clear old native caches BEFORE dropping old head. Any escaped old
        // Submission/Encoder keeps its own old head and cannot borrow a successor.
        self.bindings.take();
    }

    fn command_buffer(&self, index: usize) -> &ProtocolObject<dyn MTL4CommandBuffer> {
        if index == 0 {
            &self.cb
        } else {
            &self.extra_cbs[index - 1]
        }
    }
    fn current(&self) -> &ProtocolObject<dyn MTL4CommandBuffer> {
        self.command_buffer(self.segment)
    }

    fn next_segment(&mut self, device: &Device) -> Result<(), MetalKernelError> {
        if self.segment == self.extra_cbs.len() {
            self.extra_cbs
                .try_reserve(1)
                .map_err(|_| MetalKernelError::QueueUnavailable("segment storage exhausted"))?;
            let next = device.as_ref().newCommandBuffer().ok_or_else(|| {
                MetalKernelError::FailedToCreateResource("MTL4 event segment".into())
            })?;
            next.setLabel(self.cb.label().as_deref());
            self.extra_cbs.push(next);
        }
        self.use_device_sets(self.segment);
        self.current().endCommandBuffer();
        self.segment += 1;
        self.current()
            .beginCommandBufferWithAllocator(&self.allocator);
        Ok(())
    }

    /// Lists `resource` in this slot's set and retains it until the slot
    /// completes, once a generation. `listing`: whether the device's
    /// residency sets list it until its last GPU reader completes
    /// (`Buffer::device_listing`); under a `*-diff` declaration the slot's
    /// set leaves such a binding to them.
    fn resource(
        &mut self,
        resource: &ProtocolObject<dyn MTLResource>,
        listing: super::DeviceListing,
        kind: BindKind,
    ) {
        let covered = listing.covered();
        if self.seen.insert(resource as *const _ as usize) {
            if self.trace.on {
                self.trace.census(resource, listing, kind);
            }
            // SAFETY: the caller's borrowed resource is live; the retained
            // reference remains until this slot completes.
            self.resources.push(
                unsafe { Retained::retain(std::ptr::from_ref(resource).cast_mut()) }.unwrap(),
            );
            if covered && self.diff {
                if self.trace.on {
                    self.trace.covered.fetch_add(1, Ordering::Relaxed);
                }
            } else if self.trace.on {
                let start = std::time::Instant::now();
                self.residency
                    .addAllocation(ProtocolObject::from_ref(resource));
                let ns = start.elapsed().as_nanos() as u64;
                self.trace.adds.fetch_add(1, Ordering::Relaxed);
                self.trace.add_ns.fetch_add(ns, Ordering::Relaxed);
            } else {
                self.residency
                    .addAllocation(ProtocolObject::from_ref(resource));
            }
        }
    }
}
impl Slot {
    /// Marks the device's sets as part of segment `index`'s execution
    /// (`cb-*`), just before the segment ends: a set the router made while
    /// the segment recorded is in the list, since every binding of the
    /// segment followed its allocation. Re-reads the sets only when the
    /// router has made one since the last read.
    fn use_device_sets(&mut self, index: usize) {
        let Some(device) = &self.device else {
            return;
        };
        let generation = device.natives_generation();
        if generation != self.used_generation {
            let (generation, sets) = device.natives();
            self.used = sets;
            self.used_generation = generation;
        }
        let cb = if index == 0 {
            &self.cb
        } else {
            &self.extra_cbs[index - 1]
        };
        for set in &self.used {
            cb.useResidencySet(set);
        }
        if self.trace.on {
            self.trace
                .set_uses
                .fetch_add(self.used.len() as u64, Ordering::Relaxed);
        }
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.residency);
    }
}

pub(super) struct EventBoundary {
    after_segment: usize,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
    wait: bool,
}

struct State {
    slot: Option<Slot>,
    callbacks: Vec<Callback>,
    /// Epochs this submission completes on the queue's hook (`run_hook`).
    hook_runs: u32,
    events: Vec<EventBoundary>,
    // Prepared once before exposure. Commit/abandonment move this actual pin;
    // final Submission Drop never needs a fallible owner acquisition.
    physical: Option<crate::metal_backend::device_retirement::NamespaceOwner>,
}

pub(crate) struct Submission {
    front_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    generation: u64,
    pub queue: Arc<Queue>,
    retirement: super::retirement::Record,
    native_reuse: crate::native_reuse::NativeReuse<Slot>,
    pub(super) command_lifecycle: crate::command_lifecycle::CommandLifecycle,
    committed_events: OnceLock<Vec<EventBoundary>>,
    status: AtomicUsize,
    state: Mutex<State>,
    feedback: OnceLock<Feedback>,
    callbacks_finished: AtomicBool,
    done: Arc<Wake>,
    // LAST: escaped logical wrappers retain this immutable exact generation.
    bindings: Option<super::native_generation_bindings::GenerationBindings>,
}
unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl Submission {
    pub(super) fn bindings(
        &self,
    ) -> Result<&super::native_generation_bindings::GenerationBindings, MetalKernelError> {
        self.bindings
            .as_ref()
            .ok_or(MetalKernelError::QueueUnavailable(
                "submission is explicit legacy/untracked route",
            ))
    }
    pub(super) fn stage_binding_epoch(
        &self,
        epoch: crate::retire::Epoch,
    ) -> Result<(), MetalKernelError> {
        if let Some(bindings) = &self.bindings {
            bindings.stage_epoch(epoch)?;
        }
        Ok(())
    }
    fn binding_failure(&self, reason: crate::retirement_owners::Failure) {
        if let Some(bindings) = &self.bindings {
            bindings.quarantine(reason); // counted root; atoms only
        } else {
            self.queue.poison(reason); // unchanged explicit compatibility path
        }
    }
    fn binding_cancelled(&self) {
        if let Some(bindings) = &self.bindings {
            // SAFETY: caller has actual no-native-handoff cancellation proof.
            // Cached native objects and escaped wrappers retain SAME head.
            if unsafe { bindings.terminal_cancelled() }.is_err() {
                bindings.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            }
        }
    }
    pub(super) fn attach_physical_owner(
        &self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        let mut input = Some(owner);
        let result = (|| {
            let mut state = self.recording_state()?;
            if state.physical.is_some() {
                return Err(MetalKernelError::QueueUnavailable(
                    "M4 recording physical owner already installed",
                ));
            }
            // Cold pristine queue/submission attachment before any public
            // capability escapes. Refusal precedes taking the original input.
            let queue_pin = input.as_ref().expect("physical input").try_clone()?;
            self.queue.attach_physical_owner(queue_pin)?;
            state.physical = input.take(); // Nonpanicking move after publication.
            Ok(())
        })();
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                Err(crate::kernels::err::CompletionRegistrationError::new(error, input).into())
            } // Recording guard gone before rejected input can drop.
        }
    }
    pub(super) fn retain_namespace(
        &self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        let mut input = Some(owner);
        let result = (|| {
            let mut state = self.recording_state()?;
            let slot = state.slot.as_mut().expect("checked recording");
            let incoming = input.as_ref().expect("prepared namespace input");
            if slot.namespaces.iter().any(|old| old.same(incoming)) {
                return Ok(());
            }
            slot.namespaces.try_reserve(1).map_err(|_| {
                MetalKernelError::QueueUnavailable("M4 namespace storage exhausted")
            })?;
            // Capacity prepared before taking input; no allocation after take.
            slot.namespaces
                .push(input.take().expect("prepared namespace input"));
            Ok(())
        })();
        // Existing recording_state guard is gone before input/error destruction.
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                Err(crate::kernels::err::CompletionRegistrationError::new(error, input).into())
            }
        }
    }

    pub(super) fn notify_lifecycle(&self) {
        self.retirement.notify_lifecycle();
    }
    pub(super) fn check_lifecycle_failure(&self) -> Result<(), MetalKernelError> {
        self.queue.check_failure()
    }
    pub(super) fn fail_lifecycle(&self) {
        self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
        if self.bindings.is_none() {
            self.notify_lifecycle(); // unchanged legacy service + waiter wakes
        }
        // Enrolled failure publishes through the genuine HeaderStrong only.
        // Its immutable polling root/pending relay remains registered through
        // the next service pass; no producer parking-queue lock is needed.
        // Normal lifecycle release notifications remain separate and unchanged.
    }

    pub fn enqueue(&self) -> Result<(), MetalKernelError> {
        let _state = self.recording_state()?;
        // Commands has one open recording and commits it in producer order.
        // MTL4 has no native enqueue; this is logical state, not a queue reservation.
        let _ = self.status.compare_exchange(
            MTLCommandBufferStatus::NotEnqueued.0,
            MTLCommandBufferStatus::Enqueued.0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        Ok(())
    }
    pub fn set_label(&self, label: &str) -> Result<(), MetalKernelError> {
        let state = self.recording_state()?;
        let slot = state.slot.as_ref().expect("checked recording");
        let label = objc2_foundation::NSString::from_str(label);
        slot.cb.setLabel(Some(&label));
        for cb in &slot.extra_cbs {
            cb.setLabel(Some(&label));
        }
        Ok(())
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        let status = MTLCommandBufferStatus(self.status.load(Ordering::Acquire));
        // MTL4 exposes no live native Scheduled property. This queue-front
        // signal says all work ahead of this submission has finished. This is a
        // conservative queue-readiness observation, not native Scheduled or
        // a GPU timestamp; its cadence cost must be measured.
        if status == MTLCommandBufferStatus::Committed
            && self.front_event.signaledValue() >= self.generation
        {
            MTLCommandBufferStatus::Scheduled
        } else {
            status
        }
    }
    pub fn gpu_start_time(&self) -> f64 {
        self.feedback.get().map_or(0.0, |f| f.start)
    }
    pub fn gpu_end_time(&self) -> f64 {
        self.feedback.get().map_or(0.0, |f| f.end)
    }
    pub fn error(&self) -> Option<String> {
        self.feedback.get().and_then(|f| f.error.clone())
    }
    pub fn device(&self) -> &Device {
        &self.queue.device
    }
    pub fn encoder(self: &Arc<Self>) -> Result<Encoder, MetalKernelError> {
        let mut state = self.recording_state()?;
        let last_event_segment = state.events.last().map(|event| event.after_segment);
        let slot = state
            .slot
            .as_mut()
            .ok_or(MetalKernelError::QueueUnavailable(
                "submission no longer recording",
            ))?;
        // A cancelled preparation can leave a terminal signal in this recording.
        // Preserve that boundary if recording resumes, instead of moving the
        // signal after newly encoded work. Ordinary prepared commit never resumes.
        if last_event_segment == Some(slot.segment) {
            slot.next_segment(&self.queue.device)?;
        }
        let raw = slot.current().computeCommandEncoder().ok_or_else(|| {
            MetalKernelError::FailedToCreateResource("MTL4 compute encoder".into())
        })?;
        raw.setArgumentTable(Some(&slot.table));
        Ok(Encoder {
            raw,
            submission: Arc::clone(self),
        })
    }
    /// Returns rejected captures intact. Error propagation must carry this
    /// owner beyond any caller's recording locks before its eventual Drop.
    pub fn on_completion(
        &self,
        f: impl Fn(&Feedback) + Send + Sync + 'static,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        let mut state = match self.recording_state() {
            Ok(state) => state,
            Err(error) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    error,
                    Some(f),
                ))
            }
        };
        Self::append_callback(&mut state, f)
    }
    fn append_callback(
        state: &mut State,
        f: impl Fn(&Feedback) + Send + Sync + 'static,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        if state.callbacks.try_reserve(1).is_err() {
            return Err(crate::kernels::err::CompletionRegistrationError::new(
                MetalKernelError::QueueUnavailable("callback metadata allocation refused"),
                Some(f),
            ));
        }
        state.callbacks.push(Box::new(f));
        Ok(())
    }
    fn recording_state(&self) -> Result<parking_lot::MutexGuard<'_, State>, MetalKernelError> {
        self.recording_state_mode(false)
    }
    fn recording_state_mode(
        &self,
        nonblocking: bool,
    ) -> Result<parking_lot::MutexGuard<'_, State>, MetalKernelError> {
        self.queue.check_failure()?;
        let state = match self.state.try_lock() {
            Some(state) => state,
            None => {
                if nonblocking {
                    return Err(MetalKernelError::RecordingAdmission(
                        crate::command_lifecycle::LifecycleError::Busy,
                    ));
                }

                crate::completion_scope::before_park(Arc::as_ptr(&self.queue) as usize)
                    .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
                self.state.lock()
            }
        };
        self.queue.check_failure()?;
        if self.status().0 >= MTLCommandBufferStatus::Committed.0 || state.slot.is_none() {
            return Err(MetalKernelError::QueueUnavailable(
                "submission no longer recording",
            ));
        }
        Ok(state)
    }
    pub fn signal(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        self.event_boundary(event, value, false)
    }
    /// Prepared commit may append terminal engine signals, but cannot encode
    /// later work. Keep them after the frozen final segment without splitting.
    fn append_tail(
        state: &mut State,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        state
            .events
            .try_reserve(1)
            .map_err(|_| MetalKernelError::QueueUnavailable("terminal event storage exhausted"))?;
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }
            .ok_or(MetalKernelError::QueueUnavailable("event retain refused"))?;
        let after_segment = state.slot.as_ref().expect("checked recording").segment;
        state.events.push(EventBoundary {
            after_segment,
            event,
            value,
            wait: false,
        });
        Ok(())
    }
    pub fn wait_event(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        self.event_boundary(event, value, true)
    }
    /// Record a boundary, never a queue operation. The wrapper's boundary
    /// admission excludes open encoders. All fallible preparation precedes end.
    fn event_boundary(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
        wait: bool,
    ) -> Result<(), MetalKernelError> {
        let mut state = self.recording_state()?;
        state
            .events
            .try_reserve(1)
            .map_err(|_| MetalKernelError::QueueUnavailable("event storage exhausted"))?;
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }
            .ok_or(MetalKernelError::QueueUnavailable("event retain refused"))?;
        let slot = state.slot.as_mut().expect("checked recording");
        let after_segment = slot.segment;
        slot.next_segment(&self.queue.device)?;
        state.events.push(EventBoundary {
            after_segment,
            event,
            value,
            wait,
        });
        Ok(())
    }
    /// Reserve ordering and native handler allocations before engine epoch and
    /// subscriber registration. Dropping this value consumes no sequence.
    pub(super) fn prepare_commit(self: &Arc<Self>) -> Result<PreparedCommit<'_>, MetalKernelError> {
        self.prepare_commit_mode(false, None)
    }
    pub(super) fn prepare_commit_mode(
        self: &Arc<Self>,
        nonblocking: bool,
        authority: Option<&super::commands::OwnerSubmission<'_>>,
    ) -> Result<PreparedCommit<'_>, MetalKernelError> {
        if let Some(generation) = &self.bindings {
            if !authority.is_some_and(|owner| owner.authorizes_generation(generation)) {
                return Err(MetalKernelError::OwnerSubmissionRequired);
            }
        }
        self.queue.check_failure()?;
        let order = match self.queue.commit_order.try_lock() {
            Some(guard) => guard,
            None => {
                if nonblocking {
                    return Err(MetalKernelError::RecordingAdmission(
                        crate::command_lifecycle::LifecycleError::Busy,
                    ));
                }

                crate::completion_scope::before_park(Arc::as_ptr(&self.queue) as usize)
                    .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
                self.queue.commit_order.lock()
            }
        };
        self.queue.check_failure()?;
        if self.status().0 >= MTLCommandBufferStatus::Committed.0 {
            return Err(MetalKernelError::QueueUnavailable(
                "submission already committed",
            ));
        }
        let sequence = *order;
        sequence
            .checked_add(1)
            .ok_or(MetalKernelError::QueueUnavailable("sequence exhausted"))?;
        let state = self.recording_state_mode(nonblocking)?;
        let slot = state.slot.as_ref().expect("checked recording");
        let count = slot
            .segment
            .checked_add(1)
            .ok_or(MetalKernelError::QueueUnavailable(
                "segment count exhausted",
            ))?;
        let first = slot.cb.clone();
        let mut extra = Vec::new();
        extra.try_reserve_exact(count - 1).map_err(|_| {
            MetalKernelError::QueueUnavailable("segment dispatch storage exhausted")
        })?;
        extra.extend(slot.extra_cbs[..count - 1].iter().cloned());
        // Keep recording exclusion through registration/tail signals and owner
        // extraction. It is released before the first native queue entry.
        // No extra feedback join or options Vec on the one-segment fast path.
        let join = if count == 1 {
            None
        } else {
            Some(Arc::new(
                crate::command_segments::FeedbackJoin::try_new(count)
                    .map_err(MetalKernelError::QueueUnavailable)?,
            ))
        };
        let make_options = |index| {
            let options = MTL4CommitOptions::new();
            let record = self.retirement.clone();
            let core = self
                .queue
                .completions
                .get()
                .expect("queue initialized")
                .clone();
            let join = join.clone();
            let block = RcBlock::new(
                move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    core.native_feedback_if_ready(sequence, record.clone(), || {
                        let feedback = unsafe { feedback.as_ref() };
                        let feedback = Feedback {
                            error: feedback
                                .error()
                                .map(|e| e.localizedDescription().to_string()),
                            start: feedback.GPUStartTime(),
                            end: feedback.GPUEndTime(),
                        };
                        let Some(join) = &join else {
                            return Some(feedback);
                        };
                        if feedback.error.is_some() {
                            core.segment_failed();
                        }
                        let values = join
                            .publish(index, feedback)
                            .expect("distinct segment feedback")?;
                        // Deterministic first error in recording order, not delivery order.
                        Some(Feedback {
                            error: values
                                .iter()
                                .find_map(|f| f.get().expect("joined feedback").error.clone()),
                            start: values
                                .iter()
                                .map(|f| f.get().unwrap().start)
                                .fold(f64::INFINITY, f64::min),
                            end: values
                                .iter()
                                .map(|f| f.get().unwrap().end)
                                .fold(0.0, f64::max),
                        })
                    });
                },
            );
            unsafe {
                options.addFeedbackHandler(RcBlock::as_ptr(&block));
            }
            options
        };
        let options = if count == 1 {
            CommitOptions::Single(make_options(0))
        } else {
            let mut options = Vec::new();
            options.try_reserve_exact(count).map_err(|_| {
                MetalKernelError::QueueUnavailable("segment options allocation refused")
            })?;
            for index in 0..count {
                options.push(make_options(index));
            }
            CommitOptions::Segments(options)
        };
        Ok(PreparedCommit {
            #[cfg(test)]
            test_control: None,
            submission: self,
            order,
            state: std::cell::RefCell::new(Some(state)),
            options,
            first,
            extra,
            phase: CommitPhase::Unused,
        })
    }

    /// Abandon this recording before submission, whoever else still holds a
    /// handle to it: its native slot and owners go to host retirement now,
    /// and no later commit can enter native work. For a recording held back
    /// for an input that will never be produced (`Commands::cancel_produced`).
    /// `false` when it was already submitted or abandoned.
    pub(super) fn abandon_unsubmitted(&self) -> Result<bool, MetalKernelError> {
        let (slot, callbacks, events, physical) = {
            let mut state = self.state.lock();
            if self.status().0 >= MTLCommandBufferStatus::Committed.0 {
                return Ok(false);
            }
            let Some(slot) = state.slot.take() else {
                return Ok(false);
            };
            (
                slot,
                std::mem::take(&mut state.callbacks),
                std::mem::take(&mut state.events),
                state.physical.take(),
            )
        };
        let owners = super::retirement::NativeOwners::metal4_abandoned(
            slot,
            self.queue.clone(),
            events,
            physical,
        );
        match self
            .retirement
            .replace_prepared(super::retirement::Captures::new(callbacks), owners)
        {
            Ok(displaced) => drop(displaced),
            Err(rejected) => {
                std::mem::forget(rejected.value);
                self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
                return Err(MetalKernelError::QueueUnavailable(
                    "abandoned recording owner installation failed",
                ));
            }
        }
        if self.retirement.cancel_before_handoff().is_err() {
            self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            return Err(MetalKernelError::QueueUnavailable(
                "abandoned recording cancellation refused",
            ));
        }
        self.binding_cancelled();
        Ok(true)
    }

    /// Submit an unbound interoperability recording through ordered retirement.
    /// Bound recordings refuse before native preparation; submit those through
    /// their owning Commands transaction instead.
    pub fn commit(self: &Arc<Self>) -> Result<(), MetalKernelError> {
        self.prepare_commit()?.commit()
    }
    pub(super) fn publish_retirement_feedback(&self, feedback: &Feedback) {
        let failed = feedback.error.is_some();
        assert!(
            self.feedback
                .set(Feedback {
                    error: feedback.error.clone(),
                    start: feedback.start,
                    end: feedback.end,
                })
                .is_ok(),
            "completion once"
        );
        publish_terminal_status(&self.status, failed);
        // Data/error consumers may continue while this record's arbitrary
        // callbacks run. They do not claim logical prefix or teardown quiescence.
        self.done.notify();
    }

    /// Native data/error readiness is weaker than logical callback retirement.
    pub(super) fn check_data_wait(&self) -> Result<(), MetalKernelError> {
        if terminal_data_ready(&self.status) {
            return Ok(());
        }
        self.queue.check_failure()?;
        crate::completion_scope::before_park(Arc::as_ptr(&self.queue) as usize)
            .map_err(|_| MetalKernelError::CallbackWouldBlock)
    }
    /// The retained front event belongs to this logical submission's captured
    /// generation. It is queued under commit_order before every own segment;
    /// Apple's MTL4 signalEvent contract completes all earlier queue GPU work.
    /// Own terminal feedback cannot substitute for this receipt, including an
    /// empty submission. No arbitrary callback join is required here.
    pub(super) fn try_wait_prefix_front(&self) -> Result<(), MetalKernelError> {
        self.queue.check_failure()?;
        crate::prefix_data::wait_front(
            || self.front_event.signaledValue() >= self.generation,
            || {
                self.queue.check_failure()?;
                crate::completion_scope::before_park(Arc::as_ptr(&self.queue) as usize)
                    .map_err(|_| MetalKernelError::CallbackWouldBlock)
            },
            || {
                self.front_event
                    .waitUntilSignaledValue_timeoutMS(self.generation, 50)
            },
        )?;
        self.queue.check_failure()
    }

    pub(super) fn try_wait_data_ready(&self) -> Result<(), MetalKernelError> {
        loop {
            let seen = self.done.snapshot();
            if terminal_data_ready(&self.status) {
                return Ok(());
            }
            self.check_data_wait()?;
            self.done.wait(seen);
        }
    }

    pub(super) fn install_native_slot(&self, slot: Slot) {
        if let Err((_error, slot)) = self.native_reuse.install(slot) {
            // Invariant failure before handoff: preserve rather than release an
            // owner on an exceptional producer stack. Queue refuses new work.
            std::mem::forget(slot);
            self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            panic!("native slot installed twice");
        }
    }
    pub(super) fn publish_native_reuse(&self, side: crate::native_reuse::ReuseSide) {
        let ready = match self.native_reuse.publish(side) {
            Ok(Some(ready)) => ready,
            Ok(None) => return,
            Err(_) => {
                self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
                return;
            }
        };
        // Unique winner acquired actual returned native handoff AND ordered
        // callback completion. Feedback alone may precede native call return.
        if let Some(bindings) = &self.bindings {
            // SAFETY: the two immutable adapter facts prove this generation's
            // terminal prefix. SAME Slot head and escaped-head holders preserve
            // cached-reference/future-emission custody until they really close.
            if unsafe { bindings.terminal_submitted() }.is_err() {
                drop(ready); // restores exact Slot, not native/charge destruction
                bindings.quarantine(crate::retirement_owners::Failure::AdapterFailed);
                return;
            }
        }
        if ready
            .release(|slot| self.queue.return_retirement_slot(slot))
            .is_err()
        {
            self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
        }
    }
    /// Called only by native-owner host cleanup. If no side published, cleanup
    /// implies pre-handoff cancellation, never unknown or active native work.
    pub(super) fn discard_cancelled_native_slot(&self) {
        // SAFETY: registry does not detach unknown handoff or unfinished native
        // work. Successful records have both flags and cannot enter this arm.
        if let Some(slot) = unsafe { self.native_reuse.take_cancelled() } {
            self.queue.discard_retirement_slot(slot);
        }
    }

    pub(super) fn notify_retirement_callbacks_finished(&self) {
        self.callbacks_finished.store(true, Ordering::Release);
        self.done.notify();
    }

    /// Additive fallible wait seam. Native status alone is not evidence that
    /// logical callbacks returned. Ok reports terminality, not GPU success;
    /// callers must still inspect native error and the engine failure latch.
    /// The compatibility wait remains separate
    /// until queue poison custody and its failure policy are wired together.
    pub fn try_wait(&self) -> Result<(), MetalKernelError> {
        loop {
            let seen = self.done.snapshot();
            if self.callbacks_finished.load(Ordering::Acquire) {
                return Ok(());
            }
            if let Some(reason) = self.queue.failure.get() {
                return Err(MetalKernelError::QueueUnavailable(match reason {
                    crate::retirement_owners::Failure::UnknownHandoff => "unknown native handoff",
                    crate::retirement_owners::Failure::CallbackPanic => {
                        "completion callback panicked"
                    }
                    crate::retirement_owners::Failure::WorkerFailed => {
                        "host retirement worker failed"
                    }
                    crate::retirement_owners::Failure::Closing => "queue admission closed",
                    crate::retirement_owners::Failure::AdapterFailed => "retirement adapter failed",
                }));
            }
            crate::completion_scope::before_park(Arc::as_ptr(&self.queue) as usize)
                .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
            self.done.wait(seen);
        }
    }

    /// Nonblocking logical retirement check for final Commands teardown.
    /// Native terminal status alone does not join arbitrary completion callbacks.
    pub(super) fn poll_logical_completed(&self) -> Result<bool, MetalKernelError> {
        use crate::command_lifecycle::LifecycleError;
        self.queue.check_failure()?;
        match self.command_lifecycle.check_wait_submission() {
            Ok(()) => {}
            Err(LifecycleError::Busy | LifecycleError::NotSubmitted) => return Ok(false),
            Err(error) => return Err(error.into()),
        }
        if !self.callbacks_finished.load(Ordering::Acquire) {
            return Ok(false);
        }
        if let Some(error) = self.error() {
            return Err(MetalKernelError::CommandBufferError(error));
        }
        self.queue.check_failure()?;
        Ok(true)
    }

    pub fn wait(&self) -> Result<(), MetalKernelError> {
        self.try_wait()?;
        if let Some(error) = self.error() {
            return Err(MetalKernelError::CommandBufferError(error));
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum CommitPhase {
    Unused,
    Installed,
    EnteredNative,
    Returned,
}

enum CommitOptions {
    Single(Retained<MTL4CommitOptions>),
    Segments(Vec<Retained<MTL4CommitOptions>>),
}
impl CommitOptions {
    fn get(&self, index: usize) -> &MTL4CommitOptions {
        match self {
            Self::Single(options) => {
                assert_eq!(index, 0);
                options
            }
            Self::Segments(options) => &options[index],
        }
    }
}

pub(super) struct PreparedCommit<'a> {
    #[cfg(test)]
    pub(super) test_control: Option<Arc<super::commands::submission_test::Control>>,
    submission: &'a Arc<Submission>,
    order: parking_lot::MutexGuard<'a, u64>,
    state: std::cell::RefCell<Option<parking_lot::MutexGuard<'a, State>>>,
    options: CommitOptions,
    first: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    extra: Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    phase: CommitPhase,
}
impl PreparedCommit<'_> {
    /// Flag this submission for the queue's completion hook, completing `runs`
    /// epochs: one field of the recording state this preparation already
    /// holds. No allocation, no lock.
    pub(super) fn run_hook(&self, runs: u32) -> Result<(), MetalKernelError> {
        self.submission.queue.check_failure()?;
        if !self
            .submission
            .queue
            .completions
            .get()
            .expect("queue initialized")
            .has_hook()
        {
            return Err(MetalKernelError::QueueUnavailable(
                "Metal 4 completion hook not installed",
            ));
        }
        self.state
            .borrow_mut()
            .as_mut()
            .expect("prepared recording")
            .hook_runs = runs;
        Ok(())
    }
    /// After the epoch is taken (`Commands::submit_prepared`): a refusal fails
    /// the queue, since the hook counts every epoch taken (`completion_hook.rs`).
    pub(super) fn stage_binding_epoch(
        &self,
        epoch: crate::retire::Epoch,
    ) -> Result<(), MetalKernelError> {
        self.submission.stage_binding_epoch(epoch).inspect_err(|_| {
            self.submission
                .binding_failure(crate::retirement_owners::Failure::AdapterFailed)
        })
    }
    pub(super) fn on_completion(
        &self,
        f: impl Fn(&Feedback) + Send + Sync + 'static,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        if let Err(error) = self.submission.queue.check_failure() {
            return Err(crate::kernels::err::CompletionRegistrationError::new(
                error,
                Some(f),
            ));
        }
        Submission::append_callback(
            self.state
                .borrow_mut()
                .as_mut()
                .expect("prepared recording"),
            f,
        )
    }
    pub(super) fn signal_tail(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        self.submission.queue.check_failure()?;
        Submission::append_tail(
            self.state
                .borrow_mut()
                .as_mut()
                .expect("prepared recording"),
            event,
            value,
        )
    }
    pub(super) fn commit(mut self) -> Result<(), MetalKernelError> {
        let submission = self.submission;
        submission.queue.check_failure()?;
        if let Some(bindings) = &submission.bindings {
            bindings.require_staged()?;
        }
        let mut state = self.state.get_mut().take().expect("prepared recording");
        let mut slot = state.slot.take().expect("prepared recording slot");
        // The recording is closed: every binding of the last segment is in.
        let last = slot.segment;
        slot.use_device_sets(last);
        let physical = state.physical.take();
        let residency = slot.residency.clone();
        let callbacks = std::mem::take(&mut state.callbacks);
        let hook_runs = state.hook_runs;
        let events = std::mem::take(&mut state.events);
        assert!(
            submission.committed_events.set(events).is_ok(),
            "events installed once"
        );
        // Close recording before releasing state. Native feedback may arrive
        // during raw commit and reenter public methods; it must never wait on
        // this producer-held recording mutex.
        submission
            .status
            .store(MTLCommandBufferStatus::Committed.0, Ordering::Release);
        drop(state);
        let owners = super::retirement::NativeOwners::metal4(slot, submission.clone(), physical);
        match submission.retirement.replace_prepared(
            super::retirement::Captures {
                callbacks,
                hook_runs,
                bindings: None,
            },
            owners,
        ) {
            Ok(displaced) => drop(displaced), // Empty pre-reserved record only.
            Err(rejected) => {
                // An invariant failure must not drop the only physical owner.
                // Queue closure prevents repeated attempts; retain this payload.
                std::mem::forget(rejected.value);
                submission.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
                return Err(MetalKernelError::QueueUnavailable(
                    "prepared owner installation failed",
                ));
            }
        }
        self.phase = CommitPhase::Installed;
        // End/commit of the recording's private residency is not native CB
        // execution. Any unwind before queue entry cancels via rooted custody.
        submission.queue.commit_slot_set(&residency);
        self.extra.last().unwrap_or(&self.first).endCommandBuffer();
        #[cfg(test)]
        if let Some(control) = &self.test_control {
            control.before_entry()?;
        }
        submission.retirement.begin_handoff().map_err(|_| {
            submission.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            MetalKernelError::QueueUnavailable("native handoff refused")
        })?;
        *self.order = self
            .order
            .checked_add(1)
            .expect("checked during preparation");
        if let Some(bindings) = &submission.bindings {
            bindings.entered().inspect_err(|_| {
                submission.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            })?;
        }
        self.phase = CommitPhase::EnteredNative;
        submission
            .status
            .store(MTLCommandBufferStatus::Committed.0, Ordering::Release);
        // All queue calls for this logical recording remain under commit_order.
        // Partial/unknown entry retains every segment via the same rooted Slot.
        submission.queue.raw.signalEvent_value(
            ProtocolObject::from_ref(&*submission.front_event),
            submission.generation,
        );
        let events = submission
            .committed_events
            .get()
            .expect("closed recording events");
        crate::command_segments::dispatch(
            self.extra.len() + 1,
            events,
            |boundary| boundary.after_segment,
            |index| {
                let buffer = if index == 0 {
                    &self.first
                } else {
                    &self.extra[index - 1]
                };
                let mut raw = NonNull::from(buffer.as_ref());
                unsafe {
                    submission.queue.raw.commit_count_options(
                        NonNull::from(&mut raw),
                        1,
                        self.options.get(index),
                    );
                }
                #[cfg(test)]
                if let Some(control) = &self.test_control {
                    if control.native_call_returned(index) {
                        // Independent native prefix observer: it requires no
                        // logical owner publication or unissued later segment.
                        let event = control.observer();
                        submission
                            .queue
                            .raw
                            .signalEvent_value(ProtocolObject::from_ref(&*event), 1);
                        panic!("controlled partial native handoff");
                    }
                }
            },
            |boundary| {
                let event = ProtocolObject::from_ref(&*boundary.event);
                if boundary.wait {
                    submission
                        .queue
                        .raw
                        .waitForEvent_value(event, boundary.value);
                } else {
                    submission
                        .queue
                        .raw
                        .signalEvent_value(event, boundary.value);
                }
            },
        );
        #[cfg(test)]
        if let Some(control) = &self.test_control {
            control.before_return();
        }
        // Publish RETURNED before this side can elect the terminal winner.
        // All native queue entries returned; callbacks may already be complete.
        if let Some(bindings) = &submission.bindings {
            bindings.returned().inspect_err(|_| {
                submission.binding_failure(crate::retirement_owners::Failure::UnknownHandoff);
            })?;
        }
        submission.publish_native_reuse(crate::native_reuse::ReuseSide::HandoffReturned);
        submission.retirement.handoff_returned().map_err(|_| {
            submission.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            MetalKernelError::QueueUnavailable("handoff return publication failed")
        })?;
        self.phase = CommitPhase::Returned;
        Ok(())
    }
}
impl Drop for PreparedCommit<'_> {
    fn drop(&mut self) {
        match self.phase {
            // Unused preserves actual PREPARED cancellation/retry contract.
            // A merely staged epoch is unobserved, not terminal-cancelled work.
            CommitPhase::Unused | CommitPhase::Returned => {}
            CommitPhase::Installed => {
                if self.submission.retirement.cancel_before_handoff().is_ok() {
                    self.submission.binding_cancelled();
                } else {
                    self.submission
                        .binding_failure(crate::retirement_owners::Failure::AdapterFailed);
                }
                self.submission
                    .binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            }
            CommitPhase::EnteredNative => {
                // Enrolled path does not notify under order/state guards. Root
                // relay's immutable polling pass wakes SAME Queue waiters later.
                if self.submission.bindings.is_some() {
                    let _ = self.submission.retirement.handoff_unknown(|| {});
                } else {
                    let _ = self
                        .submission
                        .retirement
                        .handoff_unknown(|| self.submission.queue.ready.notify());
                }
                self.submission
                    .binding_failure(crate::retirement_owners::Failure::UnknownHandoff);
            }
        }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if let Some(slot) = state.slot.take() {
            let callbacks = std::mem::take(&mut state.callbacks);
            let events = std::mem::take(&mut state.events);
            let physical = state.physical.take();
            let owners = super::retirement::NativeOwners::metal4_abandoned(
                slot,
                self.queue.clone(),
                events,
                physical,
            );
            match self
                .retirement
                .replace_prepared(super::retirement::Captures::new(callbacks), owners)
            {
                Ok(displaced) => drop(displaced),
                Err(rejected) => {
                    std::mem::forget(rejected.value);
                    self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
                    return;
                }
            }
            if self.retirement.cancel_before_handoff().is_err() {
                self.binding_failure(crate::retirement_owners::Failure::AdapterFailed);
            } else {
                // Last Submission/escaped encoder Arc is gone; no native entry
                // occurred. NativeOwners still holds Slot through actual destroy.
                self.binding_cancelled();
            }
        } else {
            // Legacy public commit still owns its old callback payload. Retire
            // its unused reservation without assuming it used the new adapter.
            // A new-path record already detached simply refuses this transition.
            let _ = self.retirement.cancel_before_handoff();
        }
    }
}

#[derive(Clone)]
pub(crate) struct Encoder {
    pub raw: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    pub submission: Arc<Submission>,
}
unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

impl Encoder {
    pub fn resource(&self, resource: &ProtocolObject<dyn MTLResource>) {
        self.submission
            .state
            .lock()
            .slot
            .as_mut()
            .expect("encoding")
            .resource(resource, super::DeviceListing::No, BindKind::Raw);
    }
    /// Binds `buffer` at `index` by GPU address. `listing`: whether the
    /// device's residency sets list it until its last GPU reader completes
    /// (`Buffer::device_listing`).
    pub fn buffer(
        &self,
        index: usize,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        listing: super::DeviceListing,
    ) {
        let mut state = self.submission.state.lock();
        let slot = state.slot.as_mut().expect("encoding");
        let address = buffer.map_or(0, |b| {
            slot.resource(ProtocolObject::from_ref(b), listing, BindKind::Buffer);
            b.gpuAddress() + offset as u64
        });
        unsafe {
            slot.table.setAddress_atIndex(address, index);
        }
    }
    pub unsafe fn bytes(&self, index: usize, bytes: NonNull<std::ffi::c_void>, length: usize) {
        let mut state = self.submission.state.lock();
        let slot = state.slot.as_mut().expect("encoding");
        slot.offset = (slot.offset + 15) & !15;
        while slot
            .pages
            .get(slot.page)
            .is_some_and(|p| slot.offset > p.length() || length > p.length() - slot.offset)
        {
            slot.page += 1;
            slot.offset = 0;
        }
        if slot.page == slot.pages.len() {
            let page = self
                .submission
                .queue
                .device
                .as_ref()
                .newBufferWithLength_options(
                    PAGE_BYTES.max(length),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("MTL4 constant upload page");
            if let Some(ledger) = self.submission.queue.ledger.get() {
                slot.page_reservations
                    .push(ledger.reserve(MTLAllocation::allocatedSize(&*page)));
            }
            slot.pages.push(page);
        }
        let page = slot.pages[slot.page].clone();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().cast::<u8>(),
                page.contents().as_ptr().cast::<u8>().add(slot.offset),
                length,
            );
        }
        slot.resource(
            ProtocolObject::from_ref(&*page),
            super::DeviceListing::No,
            BindKind::Page,
        );
        unsafe {
            slot.table
                .setAddress_atIndex(page.gpuAddress() + slot.offset as u64, index);
        }
        slot.offset += length;
    }
    pub fn pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        let mut state = self.submission.state.lock();
        let slot = state.slot.as_mut().expect("encoding");
        if slot.seen.insert(pipeline as *const _ as usize) {
            // MTL4 ICB execution requires referenced pipeline allocations in
            // a residency set too (MTL4ComputeCommandEncoder.h).
            slot.residency
                .addAllocation(ProtocolObject::from_ref(pipeline));
            if slot.trace.on {
                slot.trace.pipeline_adds.fetch_add(1, Ordering::Relaxed);
            }
            slot.pipelines.push(
                unsafe { Retained::retain(std::ptr::from_ref(pipeline).cast_mut()) }.unwrap(),
            );
        }
        self.raw.setComputePipelineState(pipeline);
    }
    pub fn fence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        let mut state = self.submission.state.lock();
        let slot = state.slot.as_mut().expect("encoding");
        if slot.seen.insert(fence as *const _ as usize) {
            slot.fences
                .push(unsafe { Retained::retain(std::ptr::from_ref(fence).cast_mut()) }.unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::DeviceListing;
    use super::*;
    use objc2_foundation::NSString;
    // CPU-only: deliberately stop between feedback initialization and terminal
    // publication, without constructing a native Device or relying on a race.
    #[test]
    fn data_readiness_waits_for_terminal_error_publication() {
        for failed in [false, true] {
            let status = AtomicUsize::new(MTLCommandBufferStatus::Committed.0);
            let feedback = OnceLock::new();
            assert!(!terminal_data_ready(&status));
            assert!(feedback
                .set(Feedback {
                    error: failed.then(|| String::from("injected native error")),
                    start: 0.0,
                    end: 1.0,
                })
                .is_ok());
            assert!(feedback.get().is_some());
            assert!(
                !terminal_data_ready(&status),
                "feedback initialization exposed data before terminal status"
            );
            publish_terminal_status(&status, failed);
            assert!(terminal_data_ready(&status));
            assert_eq!(
                status.load(Ordering::Acquire),
                if failed {
                    MTLCommandBufferStatus::Error.0
                } else {
                    MTLCommandBufferStatus::Completed.0
                }
            );
            assert_eq!(feedback.get().unwrap().error.is_some(), failed);
        }
    }

    #[test]
    fn completion_wake_covers_notify_before_and_during_registration() {
        crate::completion_wake::tests::registration_control();
        println!("MTL4_WAKE transitions=1001 lost_wakes=0 waiters=0");
    }

    /// The final owner's slot Wake: a slot freed while it waits for one
    /// notifies it; one freed before anything waits does not (the wake costs
    /// the steady state nothing but a flag read).
    #[test]
    fn a_freed_slot_notifies_the_slot_wake_only_while_an_owner_waits() {
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let wake = Arc::new(Wake::default());
        assert!(queue.install_slot_wake(wake.clone()).is_ok());
        assert!(queue.install_slot_wake(Arc::new(Wake::default())).is_err());
        let settle = |seen: usize| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while wake.snapshot() == seen && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            wake.snapshot() != seen
        };
        let unwatched = wake.snapshot();
        let first = queue.acquire().unwrap();
        first.commit().unwrap();
        first.wait().unwrap();
        let freed_unwatched = settle(unwatched);
        let held: Vec<_> = (0..SLOTS).map(|_| queue.acquire().unwrap()).collect();
        let full = !queue.free_slot_or_wait();
        let watched = wake.snapshot();
        held[0].commit().unwrap();
        held[0].wait().unwrap();
        let freed_watched = settle(watched);
        let free_after = queue.free_slot_or_wait();
        for submission in &held[1..] {
            submission.commit().unwrap();
            submission.wait().unwrap();
        }
        println!(
            "MTL4_SLOT_WAKE unwatched_notified={freed_unwatched} full={full} watched_notified={freed_watched} free_after={free_after}"
        );
        assert!(!freed_unwatched, "a slot freed with no waiter notified the slot Wake");
        assert!(full, "every slot held, yet a free slot was reported");
        assert!(freed_watched, "MTL4_SLOT_WAKE_NOTIFIES_A_WAITING_OWNER");
        assert!(free_after, "the freed slot was not seen free");
    }

    #[test]
    fn completion_does_not_take_the_recording_lock() {
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let event = device.as_ref().newSharedEvent().unwrap();
        queue
            .raw
            .waitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let submission = queue.acquire().unwrap();
        let encoder = submission.encoder().unwrap();
        encoder.raw.endEncoding();
        submission.commit().unwrap();
        let recording = submission.state.lock();
        event.setSignaledValue(1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while submission.status().0 < MTLCommandBufferStatus::Completed.0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let completed_without_lock = submission.status() == MTLCommandBufferStatus::Completed;
        drop(recording); // release even on mutation failure; never strand feedback
        submission.wait().unwrap();
        assert!(
            completed_without_lock,
            "completion waited for the encoding-side mutex"
        );
        println!("MTL4_COMPLETION recording_lock_held=true completion_progress=true");
    }

    #[test]
    fn native_reuse_preserves_old_submission_markers_and_constant_bytes() {
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            eprintln!("SKIP: Metal 4 unavailable; Metal 3 remains supported");
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let library = device.as_ref().newLibraryWithSource_options_error(&NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal; kernel void step(device uint* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) { x[i] += n; }"), None).unwrap();
        let function = library
            .newFunctionWithName(&NSString::from_str("step"))
            .unwrap();
        let pipeline = device
            .as_ref()
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        let buffer = device
            .as_ref()
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        unsafe {
            std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, 256);
        }
        let mut old = Vec::new();
        for n in 1u32..=100 {
            let submission = queue.acquire().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let observed = Arc::clone(&submission);
            submission
                .on_completion(move |_| {
                    tx.send(observed.status()).unwrap();
                })
                .unwrap();
            let encoder = submission.encoder().unwrap();
            encoder.pipeline(&pipeline);
            encoder.buffer(0, Some(&buffer), 0, DeviceListing::No);
            unsafe {
                encoder.bytes(1, NonNull::from(&n).cast(), std::mem::size_of_val(&n));
            }
            encoder.raw.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.raw.endEncoding();
            submission.commit().unwrap();
            submission.wait().unwrap();
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap(),
                MTLCommandBufferStatus::Completed,
                "callback observes completed submission"
            );
            assert_eq!(
                submission.status(),
                MTLCommandBufferStatus::Completed,
                "{:?}",
                submission.error()
            );
            for i in 0..64 {
                assert_eq!(
                    unsafe { *buffer.contents().as_ptr().cast::<u32>().add(i) },
                    n * (n + 1) / 2
                );
            }
            for marker in &old {
                let marker: &Arc<Submission> = marker;
                assert_eq!(marker.status(), MTLCommandBufferStatus::Completed);
            }
            old.push(submission);
        }
        assert!(queue.created.load(Ordering::Relaxed) <= SLOTS);
        assert!(queue.reused.load(Ordering::Relaxed) >= 100 - SLOTS);
        println!("MTL4_RUST_REUSE submissions=100 created={} reused={} old_markers=100 words_checked=6400",
            queue.created.load(Ordering::Relaxed), queue.reused.load(Ordering::Relaxed));
    }
    /// Exercise the completion-prefix premise with real pending earlier work.
    /// Report GPU overlap separately: queue signals can serialize this workload,
    /// so a lack of inversion is not evidence about an untested no-signal path.
    #[test]
    fn later_feedback_observes_the_earlier_submission_retired() {
        let _scope = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let library = device.as_ref().newLibraryWithSource_options_error(&NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal; kernel void step(device float* x [[buffer(0)]]) { x[0] = sin(x[0]) + 0.01f; }"), None).unwrap();
        let function = library
            .newFunctionWithName(&NSString::from_str("step"))
            .unwrap();
        let pipeline = device
            .as_ref()
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        let mut pending_at_later_commit = 0;
        let mut gpu_inversions = 0;
        for _ in 0..8 {
            let heavy = queue.acquire().unwrap();
            let tiny = queue.acquire().unwrap();
            for (submission, count) in [(&heavy, 4000), (&tiny, 1)] {
                let buffer = device
                    .as_ref()
                    .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                    .unwrap();
                unsafe {
                    *buffer.contents().as_ptr().cast::<f32>() = 0.;
                }
                let encoder = submission.encoder().unwrap();
                encoder.pipeline(&pipeline);
                encoder.buffer(0, Some(&buffer), 0, DeviceListing::No);
                for _ in 0..count {
                    encoder.raw.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder
                        .raw
                        .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                            MTLStages::Dispatch,
                            MTLStages::Dispatch,
                            MTL4VisibilityOptions::Device,
                        );
                }
                encoder.raw.endEncoding();
            }
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let earlier = heavy.clone();
            tiny.on_completion(move |_| {
                tx.send(earlier.status()).unwrap();
            })
            .unwrap();
            heavy.commit().unwrap();
            pending_at_later_commit +=
                usize::from(heavy.status() != MTLCommandBufferStatus::Completed);
            tiny.commit().unwrap();
            tiny.wait().unwrap();
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap(),
                MTLCommandBufferStatus::Completed,
                "later feedback preceded earlier retirement"
            );
            heavy.wait().unwrap();
            gpu_inversions += usize::from(tiny.gpu_end_time() < heavy.gpu_end_time());
        }
        assert!(
            pending_at_later_commit > 0,
            "earlier work was already retired: vacuous test"
        );
        println!("MTL4_RETIRE_PREFIX trials=8 earlier_pending_at_later_commit={} gpu_end_inversions={} later_feedback_saw_earlier_completed=8",
            pending_at_later_commit, gpu_inversions);
    }

    #[test]
    fn full_native_pool_parks_and_feedback_releases_the_waiter() {
        let _scope = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let event = device.as_ref().newSharedEvent().unwrap();
        struct UnblockOnDrop(Retained<ProtocolObject<dyn MTLSharedEvent>>);
        impl Drop for UnblockOnDrop {
            fn drop(&mut self) {
                self.0.setSignaledValue(1);
            }
        }
        // Acquisition and assertion failures must not strand the queue wait.
        let _unblock = UnblockOnDrop(event.clone());

        queue
            .raw
            .waitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let mut submissions = Vec::new();
        for _ in 0..SLOTS {
            let submission = queue.acquire().unwrap();
            submission.encoder().unwrap().raw.endEncoding();
            submission.commit().unwrap();
            submissions.push(submission);
        }
        assert_eq!(queue.created.load(Ordering::Relaxed), SLOTS);
        let worker_queue = queue.clone();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let _scope = crate::AutoreleaseScope::new();
            let slot = worker_queue.acquire().unwrap();
            tx.send(slot).unwrap();
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while queue.acquire_parks.load(Ordering::Relaxed) == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let parked = queue.acquire_parks.load(Ordering::Relaxed) > 0;
        let blocked = matches!(rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty));
        // Unblock even on a failing assertion: never strand queued GPU work.
        event.setSignaledValue(1);
        let slot = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        for submission in submissions {
            submission.wait().unwrap();
        }
        assert!(
            parked && blocked,
            "the ninth acquisition must encounter the full pool"
        );
        assert_eq!(queue.created.load(Ordering::Relaxed), SLOTS);
        drop(slot);
        println!("MTL4_FULL_POOL slots={} ninth_acquire_blocked=true feedback_unblocked=true acquire_parks={}",
            SLOTS, queue.acquire_parks.load(Ordering::Relaxed));
    }

    #[test]
    fn native_pool_backpressure_and_upload_page_rollover() {
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        let ledger = super::super::ledger::Ledger::new();
        queue.attach_ledger(ledger.clone());
        let library = device.as_ref().newLibraryWithSource_options_error(&NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal; kernel void store(device uint* x [[buffer(0)]], constant uint* n [[buffer(1)]]) { x[0] = n[0]; }"), None).unwrap();
        let function = library
            .newFunctionWithName(&NSString::from_str("store"))
            .unwrap();
        let pipeline = device
            .as_ref()
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        let mut pending = Vec::new();
        for n in 1u32..=128 {
            let buffer = device
                .as_ref()
                .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let submission = queue.acquire().unwrap();
            let encoder = submission.encoder().unwrap();
            encoder.pipeline(&pipeline);
            encoder.buffer(0, Some(&buffer), 0, DeviceListing::No);
            // Each recording crosses the regular page boundary and also uses
            // an oversize page. Constants differ within one native buffer.
            for value in [n + 1000, n] {
                let bytes = vec![value; PAGE_BYTES / 4 + 1];
                unsafe {
                    encoder.bytes(
                        1,
                        NonNull::new(bytes.as_ptr().cast_mut()).unwrap().cast(),
                        bytes.len() * 4,
                    );
                }
                encoder.raw.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder
                    .raw
                    .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                        MTLStages::Dispatch,
                        MTLStages::Dispatch,
                        MTL4VisibilityOptions::Device,
                    );
            }
            encoder.raw.endEncoding();
            submission.commit().unwrap();
            pending.push((submission, buffer, n));
        }
        for (submission, buffer, n) in &pending {
            submission.wait().unwrap();
            assert_eq!(
                submission.status(),
                MTLCommandBufferStatus::Completed,
                "{:?}",
                submission.error()
            );
            assert_eq!(unsafe { *buffer.contents().as_ptr().cast::<u32>() }, *n);
        }
        assert!(queue.created.load(Ordering::Relaxed) <= SLOTS);
        assert!(queue.reused.load(Ordering::Relaxed) >= 128 - SLOTS);
        println!(
            "MTL4_POOL_STRESS submissions=128 created={} reused={} checked=128 page_rollover=true",
            queue.created.load(Ordering::Relaxed),
            queue.reused.load(Ordering::Relaxed)
        );
        assert!(ledger.committed() >= 2 * (PAGE_BYTES + 4));
        let weak = Arc::downgrade(&queue);
        drop(pending);
        drop(queue);
        // A native feedback callback may still be unwinding after publishing
        // completion; it must not leave a retained queue/submission cycle.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while (weak.upgrade().is_some() || ledger.committed() != 0)
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        assert!(
            weak.upgrade().is_none(),
            "native feedback retains queue after completion"
        );
        assert_eq!(
            ledger.committed(),
            0,
            "upload reservations retire with native slots"
        );
    }
    /// Under `cb-*` every segment of a command buffer marks the device's
    /// sets used: `beginCommandBufferWithAllocator` drops an earlier
    /// `useResidencySet`, and a segment boundary begins a new buffer.
    #[test]
    fn each_segment_of_a_command_buffer_uses_the_device_sets() {
        // Read once, before the queue exists (one process per test).
        std::env::set_var("LLOOM_RESIDENCY_METAL4", "cb-full");
        std::env::set_var("LLOOM_METAL4_SLOT_TRACE", "1");
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        assert_eq!(queue.declare(), super::super::Metal4Declare::CbFull);
        let sets = Arc::new(super::super::ResidencySet::new(&device));
        // SAFETY: only asks whether a native set exists; nothing is borrowed.
        if unsafe { sets.raw() }.is_none() {
            return;
        }
        queue.use_device_sets(sets);
        let event = device.as_ref().newSharedEvent().unwrap();
        let submission = queue.acquire().unwrap();
        submission.encoder().unwrap().raw.endEncoding();
        submission.signal(&event, 1).unwrap();
        submission.encoder().unwrap().raw.endEncoding();
        submission.commit().unwrap();
        submission.wait().unwrap();
        let t = queue.slot_residency().expect("the instrument is on");
        println!("MTL4_SEGMENT_SETS set_uses={} commits={}", t.set_uses, t.commits);
        assert_eq!(
            t.set_uses, 2,
            "one device set, used by each of the two segments: {t:?}"
        );
        assert_eq!(event.signaledValue(), 1, "the boundary's signal ran");
    }

    /// A `*-diff` queue the device's sets were never installed on lists a
    /// covered binding itself: nothing else would make it resident.
    #[test]
    fn a_queue_without_the_device_sets_lists_covered_bindings() {
        // Read once, before the queue exists (one process per test).
        std::env::set_var("LLOOM_RESIDENCY_METAL4", "queue-diff");
        std::env::set_var("LLOOM_METAL4_SLOT_TRACE", "1");
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let queue = Queue::new(device.clone()).unwrap();
        assert_eq!(queue.declare(), super::super::Metal4Declare::QueueDiff);
        let buffer = device
            .as_ref()
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let submission = queue.acquire().unwrap();
        let encoder = submission.encoder().unwrap();
        encoder.buffer(0, Some(&buffer), 0, DeviceListing::Pool);
        encoder.raw.endEncoding();
        submission.commit().unwrap();
        submission.wait().unwrap();
        let t = queue.slot_residency().expect("the instrument is on");
        println!("MTL4_UNWIRED_DIFF adds={} covered={}", t.adds, t.covered);
        assert_eq!(
            (t.adds, t.covered),
            (1, 0),
            "the covered binding is listed by the slot: {t:?}"
        );
    }

    #[test]
    fn queue_events_order_cross_queue_work_and_feedback() {
        let _pool = crate::AutoreleaseScope::new();
        let device = Device::system_default().expect("Metal device");
        if !device.as_ref().supportsFamily(MTLGPUFamily::Metal4) {
            return;
        }
        let a = Queue::new(device.clone()).unwrap();
        let b = Queue::new(device.clone()).unwrap();
        let event = device.as_ref().newSharedEvent().unwrap();
        let library = device.as_ref().newLibraryWithSource_options_error(&NSString::from_str(
            "#include <metal_stdlib>\nusing namespace metal; kernel void append(device uint* x [[buffer(0)]], constant uint& n [[buffer(1)]]) { x[0] = x[0] * 10 + n; }"), None).unwrap();
        let function = library
            .newFunctionWithName(&NSString::from_str("append"))
            .unwrap();
        let pipeline = device
            .as_ref()
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        let buffer = device
            .as_ref()
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        unsafe {
            *buffer.contents().as_ptr().cast::<u32>() = 42;
        }
        let record = |queue: &Arc<Queue>, n: u32| {
            let submission = queue.acquire().unwrap();
            let encoder = submission.encoder().unwrap();
            encoder.pipeline(&pipeline);
            encoder.buffer(0, Some(&buffer), 0, DeviceListing::No);
            unsafe {
                encoder.bytes(1, NonNull::from(&n).cast(), 4);
            }
            encoder.raw.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.raw.endEncoding();
            submission
        };
        let first = record(&a, 1);
        first.signal(&event, 1).unwrap();
        b.raw
            .waitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let second = record(&b, 2);
        // Submit the consumer first: only the explicit event can order it
        // after the producer on the independent queue.
        second.commit().unwrap();
        first.commit().unwrap();
        first.wait().unwrap();
        second.wait().unwrap();
        assert_eq!(first.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(second.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(event.signaledValue(), 1);
        assert_eq!(unsafe { *buffer.contents().as_ptr().cast::<u32>() }, 4212);
        println!("MTL4_CROSS_QUEUE consumer_submitted_first=true result=4212 expected=4212");
    }
}
