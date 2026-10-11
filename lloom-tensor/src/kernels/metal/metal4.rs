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
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

const SLOTS: usize = 8;
const PAGE_BYTES: usize = 64 * 1024;

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
    ready: Wake,
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
        Ok(Arc::new(Self {
            raw,
            device,
            available: ArrayQueue::new(SLOTS),
            allocated: AtomicUsize::new(0),
            ready: Wake::default(),
            acquire_parks: AtomicUsize::new(0),
            ledger: OnceLock::new(),
            created: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
        }))
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
            self.acquire_parks.fetch_add(1, Ordering::Relaxed);
            self.ready.wait(seen);
        };
        // Only completed submissions return slots; abandoned uncommitted ones
        // are destroyed instead. No reset races outstanding GPU commands.
        slot.reset();
        let front_event = slot.front_event.clone();
        let generation = slot.generation;
        Ok(Arc::new(Submission {
            front_event,
            generation,
            queue: Arc::clone(self),
            status: AtomicUsize::new(MTLCommandBufferStatus::NotEnqueued.0),
            state: Mutex::new(State {
                slot: Some(slot),
                callbacks: Vec::new(),
                signals: Vec::new(),
            }),
            feedback: OnceLock::new(),
            done: Wake::default(),
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
    status: AtomicUsize,
    state: Mutex<State>,
    feedback: OnceLock<Feedback>,
    done: Wake,
}
unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl Submission {
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
    pub fn encoder(self: &Arc<Self>) -> Encoder {
        let state = self.state.lock();
        let slot = state.slot.as_ref().expect("uncommitted submission");
        let raw = slot
            .cb
            .computeCommandEncoder()
            .expect("MTL4 compute encoder");
        raw.setArgumentTable(Some(&slot.table));
        Encoder {
            raw,
            submission: Arc::clone(self),
        }
    }
    pub fn on_completion(&self, f: impl Fn(&Feedback) + Send + Sync + 'static) {
        let mut state = self.state.lock();
        assert!(self.status().0 < MTLCommandBufferStatus::Committed.0);
        state.callbacks.push(Box::new(f));
    }
    pub fn signal(&self, event: &ProtocolObject<dyn MTLSharedEvent>, value: u64) {
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }.unwrap();
        self.state.lock().signals.push((event, value));
    }
    pub fn commit(self: &Arc<Self>) {
        let mut state = self.state.lock();
        if self.status().0 >= MTLCommandBufferStatus::Committed.0 {
            return;
        }
        let slot = state.slot.take().expect("uncommitted slot");
        slot.residency.commit();
        slot.cb.endCommandBuffer();
        let options = MTL4CommitOptions::new();
        let mut cb = NonNull::from(slot.cb.as_ref());
        // Completion owns this payload exclusively. Its mutex is only needed
        // because Metal's block is Fn, not FnOnce; the encoding thread never
        // acquires it. In particular feedback never locks recording state.
        let retired = Mutex::new(Some((slot, std::mem::take(&mut state.callbacks))));
        let this = Arc::clone(self);
        let block = RcBlock::new(
            move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                let feedback = unsafe { feedback.as_ref() };
                let info = Feedback {
                    error: feedback
                        .error()
                        .map(|e| e.localizedDescription().to_string()),
                    start: feedback.GPUStartTime(),
                    end: feedback.GPUEndTime(),
                };
                let (slot, callbacks) = retired.lock().take().expect("completion once");
                this.finish(info, slot, callbacks);
            },
        );
        self.status
            .store(MTLCommandBufferStatus::Committed.0, Ordering::Release);
        // SAFETY: the array has one live command buffer; Metal retains the
        // feedback block. the feedback payload retains the slot until feedback arrives.
        unsafe {
            options.addFeedbackHandler(RcBlock::as_ptr(&block));
            self.queue.raw.signalEvent_value(
                ProtocolObject::from_ref(&*self.front_event),
                self.generation,
            );
            self.queue
                .raw
                .commit_count_options(NonNull::from(&mut cb), 1, &options);
        }
        for (event, value) in state.signals.drain(..) {
            self.queue
                .raw
                .signalEvent_value(ProtocolObject::from_ref(&*event), value);
        }
    }
    fn finish(&self, feedback: Feedback, slot: Slot, callbacks: Vec<Callback>) {
        let failed = feedback.error.is_some();
        assert!(self.feedback.set(feedback).is_ok(), "completion once");
        self.status.store(
            if failed {
                MTLCommandBufferStatus::Error.0
            } else {
                MTLCommandBufferStatus::Completed.0
            },
            Ordering::Release,
        );
        self.done.notify();
        // Preserve callback-before-slot-reuse semantics, including error latches
        // and epoch retirement. Old markers read the immutable feedback above.
        let feedback = self.feedback.get().expect("published above");
        for callback in callbacks {
            callback(feedback);
        }
        assert!(
            self.queue.available.push(slot).is_ok(),
            "one return per native slot"
        );
        self.queue.ready.notify();
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
        self.status.store(
            if failed {
                MTLCommandBufferStatus::Error.0
            } else {
                MTLCommandBufferStatus::Completed.0
            },
            Ordering::Release,
        );
    }

    pub(super) fn notify_retirement_callbacks_finished(&self) {
        self.done.notify();
    }

    pub fn wait(&self) {
        loop {
            let seen = self.done.snapshot();
            if self.status().0 >= MTLCommandBufferStatus::Completed.0 {
                return;
            }
            self.done.wait(seen);
        }
    }
}
impl Drop for Submission {
    fn drop(&mut self) {
        if self.state.get_mut().slot.take().is_some() {
            // An abandoned, never submitted recording is not reset/reused.
            self.queue.allocated.fetch_sub(1, Ordering::AcqRel);
            self.queue.ready.notify();
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
        let encoder = submission.encoder();
        encoder.raw.endEncoding();
        submission.commit();
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
        submission.wait();
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
            submission.on_completion(move |_| {
                tx.send(observed.status()).unwrap();
            });
            let encoder = submission.encoder();
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
            submission.commit();
            submission.wait();
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
                let encoder = submission.encoder();
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
            });
            heavy.commit();
            pending_at_later_commit +=
                usize::from(heavy.status() != MTLCommandBufferStatus::Completed);
            tiny.commit();
            tiny.wait();
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap(),
                MTLCommandBufferStatus::Completed,
                "later feedback preceded earlier retirement"
            );
            heavy.wait();
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
            submission.encoder().raw.endEncoding();
            submission.commit();
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
            submission.wait();
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
            let encoder = submission.encoder();
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
            submission.commit();
            pending.push((submission, buffer, n));
        }
        for (submission, buffer, n) in &pending {
            submission.wait();
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
            let encoder = submission.encoder();
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
        second.commit();
        first.commit();
        first.wait();
        second.wait();
        assert_eq!(first.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(second.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(event.signaledValue(), 1);
        assert_eq!(unsafe { *buffer.contents().as_ptr().cast::<u32>() }, 4212);
        println!("MTL4_CROSS_QUEUE consumer_submitted_first=true result=4212 expected=4212");
    }
}
