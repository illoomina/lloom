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
    collections::HashSet,
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
}
unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl Queue {
    pub fn new(device: Device) -> Result<Arc<Self>, MetalKernelError> {
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
        });
        let core = super::retirement::Metal4Completions::new(
            Arc::as_ptr(&queue) as usize,
            failure,
            queue.ready.clone(),
        );
        assert!(queue.completions.set(Arc::new(core)).is_ok());
        Ok(queue)
    }

    /// Acquisition and every submission wait use the same Wake. Poison closes
    /// admission without claiming native completion or releasing any payload.
    fn check_failure(&self) -> Result<(), MetalKernelError> {
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
        Ok(())
    }

    pub(super) fn discard_retirement_slot(&self, slot: Slot) {
        drop(slot);
        self.allocated.fetch_sub(1, Ordering::AcqRel);
        self.ready.notify();
    }

    pub fn attach_ledger(&self, ledger: Arc<super::ledger::Ledger>) {
        assert!(
            self.ledger.set(ledger).is_ok(),
            "Metal 4 ledger installed once at device construction"
        );
    }

    pub fn acquire(self: &Arc<Self>) -> Result<Arc<Submission>, MetalKernelError> {
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
                        return Err(e);
                    }
                }
            }
            crate::completion_scope::before_park(Arc::as_ptr(self) as usize)
                .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
            self.acquire_parks.fetch_add(1, Ordering::Relaxed);
            self.ready.wait(seen);
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
            super::retirement::Captures(Vec::new()),
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
        Ok(Arc::new(Submission {
            front_event,
            generation,
            queue: Arc::clone(self),
            retirement,
            native_reuse: crate::native_reuse::NativeReuse::new(),
            command_lifecycle: crate::command_lifecycle::CommandLifecycle::new(),
            committed_signals: OnceLock::new(),
            status: AtomicUsize::new(MTLCommandBufferStatus::NotEnqueued.0),
            state: Mutex::new(State {
                slot: Some(slot),
                callbacks: Vec::new(),
                signals: Vec::new(),
            }),
            feedback: OnceLock::new(),
            callbacks_finished: AtomicBool::new(false),
            done: self.ready.clone(),
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

pub(super) struct Slot {
    front_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    generation: u64,
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    resources: Vec<Retained<ProtocolObject<dyn MTLResource>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    fences: Vec<Retained<ProtocolObject<dyn MTLFence>>>,
    seen: HashSet<usize>,
    pages: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    page_reservations: Vec<super::ledger::Reservation>,
    page: usize,
    offset: usize,
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
            allocator,
            table,
            residency,
            queue: queue.raw.clone(),
            resources: Vec::with_capacity(256),
            pipelines: Vec::with_capacity(64),
            fences: Vec::with_capacity(64),
            seen: HashSet::with_capacity(512),
            pages: Vec::new(),
            page_reservations: Vec::new(),
            page: 0,
            offset: 0,
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
        self.cb.beginCommandBufferWithAllocator(&self.allocator);
    }

    fn resource(&mut self, resource: &ProtocolObject<dyn MTLResource>) {
        if self.seen.insert(resource as *const _ as usize) {
            // SAFETY: the caller's borrowed resource is live; the retained
            // reference and residency remain until this slot completes.
            self.resources.push(
                unsafe { Retained::retain(std::ptr::from_ref(resource).cast_mut()) }.unwrap(),
            );
            self.residency
                .addAllocation(ProtocolObject::from_ref(resource));
        }
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.residency);
    }
}

struct State {
    slot: Option<Slot>,
    callbacks: Vec<Callback>,
    signals: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
}

pub(crate) struct Submission {
    front_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    generation: u64,
    pub queue: Arc<Queue>,
    retirement: super::retirement::Record,
    native_reuse: crate::native_reuse::NativeReuse<Slot>,
    pub(super) command_lifecycle: crate::command_lifecycle::CommandLifecycle,
    committed_signals: OnceLock<Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>>,
    status: AtomicUsize,
    state: Mutex<State>,
    feedback: OnceLock<Feedback>,
    callbacks_finished: AtomicBool,
    done: Arc<Wake>,
}
unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl Submission {
    pub(super) fn notify_lifecycle(&self) {
        self.retirement.notify_lifecycle();
    }
    pub fn enqueue(&self) {
        // Commands has one open recording and commits it in producer order.
        // MTL4 has no native enqueue; this is logical state, not a queue reservation.
        let _ = self.status.compare_exchange(
            MTLCommandBufferStatus::NotEnqueued.0,
            MTLCommandBufferStatus::Enqueued.0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
    pub fn set_label(&self, label: &str) {
        self.state
            .lock()
            .slot
            .as_ref()
            .expect("recording")
            .cb
            .setLabel(Some(&objc2_foundation::NSString::from_str(label)));
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
        let state = self.recording_state()?;
        let slot = state
            .slot
            .as_ref()
            .ok_or(MetalKernelError::QueueUnavailable(
                "submission no longer recording",
            ))?;
        let raw = slot.cb.computeCommandEncoder().ok_or_else(|| {
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
        self.queue.check_failure()?;
        let state = match self.state.try_lock() {
            Some(state) => state,
            None => {
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
    pub fn signal(&self, event: &ProtocolObject<dyn MTLSharedEvent>, value: u64) {
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }.unwrap();
        self.state.lock().signals.push((event, value));
    }
    /// Reserve ordering and native handler allocations before engine epoch and
    /// subscriber registration. Dropping this value consumes no sequence.
    pub(super) fn prepare_commit(self: &Arc<Self>) -> Result<PreparedCommit<'_>, MetalKernelError> {
        self.queue.check_failure()?;
        let order = match self.queue.commit_order.try_lock() {
            Some(guard) => guard,
            None => {
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
        let options = MTL4CommitOptions::new();
        let record = self.retirement.clone();
        let core = self
            .queue
            .completions
            .get()
            .expect("queue initialized")
            .clone();
        let block = RcBlock::new(
            move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                core.native_feedback(sequence, record.clone(), || {
                    // SAFETY: Metal owns its feedback for this invocation.
                    let feedback = unsafe { feedback.as_ref() };
                    Feedback {
                        error: feedback
                            .error()
                            .map(|e| e.localizedDescription().to_string()),
                        start: feedback.GPUStartTime(),
                        end: feedback.GPUEndTime(),
                    }
                });
            },
        );
        // SAFETY: options copies the handler. It retains only rooted metadata.
        unsafe {
            options.addFeedbackHandler(RcBlock::as_ptr(&block));
        }
        Ok(PreparedCommit {
            submission: self,
            order,
            options,
            phase: CommitPhase::Unused,
        })
    }

    /// Public submissions use the same prepared transaction as Commands. There
    /// is no second legacy path that can bypass ordered retirement/custody.
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
            self.queue
                .poison(crate::retirement_owners::Failure::AdapterFailed);
            panic!("native slot installed twice");
        }
    }
    pub(super) fn publish_native_reuse(&self, side: crate::native_reuse::ReuseSide) {
        let result = self
            .native_reuse
            .publish(side)
            .and_then(|ready| match ready {
                Some(ready) => ready.release(|slot| self.queue.return_retirement_slot(slot)),
                None => Ok(()),
            });
        if result.is_err() {
            self.queue
                .poison(crate::retirement_owners::Failure::AdapterFailed);
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

pub(super) struct PreparedCommit<'a> {
    submission: &'a Arc<Submission>,
    order: parking_lot::MutexGuard<'a, u64>,
    options: Retained<MTL4CommitOptions>,
    phase: CommitPhase,
}
impl PreparedCommit<'_> {
    pub(super) fn commit(mut self) -> Result<(), MetalKernelError> {
        let submission = self.submission;
        let mut state = submission.state.lock();
        let slot = state.slot.take().expect("prepared recording slot");
        let cb = slot.cb.clone();
        let residency = slot.residency.clone();
        let callbacks = std::mem::take(&mut state.callbacks);
        let signals = std::mem::take(&mut state.signals);
        assert!(
            submission.committed_signals.set(signals).is_ok(),
            "signals installed once"
        );
        // Close recording before releasing state. Native feedback may arrive
        // during raw commit and reenter public methods; it must never wait on
        // this producer-held recording mutex.
        submission
            .status
            .store(MTLCommandBufferStatus::Committed.0, Ordering::Release);
        drop(state);
        let owners = super::retirement::NativeOwners::metal4(slot, submission.clone(), None);
        match submission
            .retirement
            .replace_prepared(super::retirement::Captures(callbacks), owners)
        {
            Ok(displaced) => drop(displaced), // Empty pre-reserved record only.
            Err(rejected) => {
                // An invariant failure must not drop the only physical owner.
                // Queue closure prevents repeated attempts; retain this payload.
                std::mem::forget(rejected.value);
                submission
                    .queue
                    .poison(crate::retirement_owners::Failure::AdapterFailed);
                return Err(MetalKernelError::QueueUnavailable(
                    "prepared owner installation failed",
                ));
            }
        }
        self.phase = CommitPhase::Installed;
        // End/commit of the recording's private residency is not native CB
        // execution. Any unwind before queue entry cancels via rooted custody.
        residency.commit();
        cb.endCommandBuffer();
        submission.retirement.begin_handoff().map_err(|_| {
            submission
                .queue
                .poison(crate::retirement_owners::Failure::AdapterFailed);
            MetalKernelError::QueueUnavailable("native handoff refused")
        })?;
        *self.order = self
            .order
            .checked_add(1)
            .expect("checked during preparation");
        self.phase = CommitPhase::EnteredNative;
        submission
            .status
            .store(MTLCommandBufferStatus::Committed.0, Ordering::Release);
        let mut raw = NonNull::from(cb.as_ref());
        // SAFETY: payload roots CB/allocator/resources through real feedback and
        // handoff-return; options roots metadata, never final physical owners.
        unsafe {
            submission.queue.raw.signalEvent_value(
                ProtocolObject::from_ref(&*submission.front_event),
                submission.generation,
            );
            submission
                .queue
                .raw
                .commit_count_options(NonNull::from(&mut raw), 1, &self.options);
        }
        for (event, value) in submission
            .committed_signals
            .get()
            .expect("closed recording signals")
        {
            submission
                .queue
                .raw
                .signalEvent_value(ProtocolObject::from_ref(&**event), *value);
        }
        // Reuse publication is independent of cleanup readiness. It happens
        // only after all native entry calls return; callbacks may precede it.
        submission.publish_native_reuse(crate::native_reuse::ReuseSide::HandoffReturned);
        submission.retirement.handoff_returned().map_err(|_| {
            submission
                .queue
                .poison(crate::retirement_owners::Failure::AdapterFailed);
            MetalKernelError::QueueUnavailable("handoff return publication failed")
        })?;
        self.phase = CommitPhase::Returned;
        Ok(())
    }
}
impl Drop for PreparedCommit<'_> {
    fn drop(&mut self) {
        match self.phase {
            CommitPhase::Unused | CommitPhase::Returned => {}
            CommitPhase::Installed => {
                let _ = self.submission.retirement.cancel_before_handoff();
                self.submission
                    .queue
                    .poison(crate::retirement_owners::Failure::AdapterFailed);
            }
            CommitPhase::EnteredNative => {
                let _ = self
                    .submission
                    .retirement
                    .handoff_unknown(|| self.submission.queue.ready.notify());
                self.submission
                    .queue
                    .poison(crate::retirement_owners::Failure::UnknownHandoff);
            }
        }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if let Some(slot) = state.slot.take() {
            let callbacks = std::mem::take(&mut state.callbacks);
            let signals = std::mem::take(&mut state.signals);
            let owners = super::retirement::NativeOwners::metal4_abandoned(
                slot,
                self.queue.clone(),
                signals,
                None,
            );
            match self
                .retirement
                .replace_prepared(super::retirement::Captures(callbacks), owners)
            {
                Ok(displaced) => drop(displaced),
                Err(rejected) => {
                    std::mem::forget(rejected.value);
                    self.queue
                        .poison(crate::retirement_owners::Failure::AdapterFailed);
                    return;
                }
            }
            if self.retirement.cancel_before_handoff().is_err() {
                self.queue
                    .poison(crate::retirement_owners::Failure::AdapterFailed);
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
            .resource(resource);
    }
    pub fn buffer(
        &self,
        index: usize,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
    ) {
        let mut state = self.submission.state.lock();
        let slot = state.slot.as_mut().expect("encoding");
        let address = buffer.map_or(0, |b| {
            slot.resource(ProtocolObject::from_ref(b));
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
        slot.resource(ProtocolObject::from_ref(&*page));
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
            encoder.buffer(0, Some(&buffer), 0);
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
                encoder.buffer(0, Some(&buffer), 0);
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
            encoder.buffer(0, Some(&buffer), 0);
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
            encoder.buffer(0, Some(&buffer), 0);
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
        first.signal(&event, 1);
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
