use super::{
    executor::ExecutorSlot, BlitCommandEncoder, ComputeCommandEncoder, Device, Fence, PrevCeOutputs,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTL4CommandQueue, MTLCommandBuffer, MTLCommandBufferStatus, MTLDispatchType};
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone)]
pub struct CommandBuffer {
    backend: Backend,
}

#[derive(Clone)]
enum Backend {
    Metal3(Retained<ProtocolObject<dyn MTLCommandBuffer>>),
    Metal4(Arc<super::metal4::Submission>),
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("metal4", &self.is_metal4())
            .field("status", &self.status())
            .finish()
    }
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            backend: Backend::Metal3(raw),
        }
    }

    pub(crate) fn from_metal4(submission: Arc<super::metal4::Submission>) -> Self {
        Self {
            backend: Backend::Metal4(submission),
        }
    }
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        match &self.backend {
            Backend::Metal3(raw) => raw,
            Backend::Metal4(_) => unreachable!("Metal 3-only helper"),
        }
    }
    pub(crate) fn is_metal4(&self) -> bool {
        matches!(self.backend, Backend::Metal4(_))
    }
    pub(crate) fn identity(&self) -> usize {
        match &self.backend {
            Backend::Metal3(raw) => Retained::as_ptr(raw) as usize,
            Backend::Metal4(s) => Arc::as_ptr(s) as usize,
        }
    }
    pub(crate) fn device(&self) -> Device {
        match &self.backend {
            Backend::Metal3(raw) => Device::new(raw.device()),
            Backend::Metal4(s) => s.device().clone(),
        }
    }

    /// Create a compute command encoder with the provided per-encoder fence and global output map.
    pub fn compute_command_encoder(
        &self,
        fence: &Arc<Fence>,
        prev_ce_outputs: &PrevCeOutputs,
    ) -> ComputeCommandEncoder {
        self.compute_command_encoder_with_executor(
            fence,
            prev_ce_outputs,
            &Arc::new(ExecutorSlot::Classical),
        )
    }

    /// As [`Self::compute_command_encoder`], but submitting dispatches through
    /// `executor` (`DESIGN.md` §11.1).
    pub fn compute_command_encoder_with_executor(
        &self,
        fence: &Arc<Fence>,
        prev_ce_outputs: &PrevCeOutputs,
        executor: &Arc<ExecutorSlot>,
    ) -> ComputeCommandEncoder {
        if let Backend::Metal4(s) = &self.backend {
            return ComputeCommandEncoder::with_backend(
                super::raw_encoder::Compute::Metal4(s.encoder()),
                self.clone(),
                Arc::clone(fence),
                Arc::clone(prev_ce_outputs),
                Arc::clone(executor),
            );
        }
        let raw = match crate::kernels::metal::profile::sample_encoder(self.raw()) {
            Some((samples, index)) => self.sampled_encoder(&samples, index),
            None => self
                .raw()
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent),
        };
        raw.map(|raw| {
            ComputeCommandEncoder::with_backend(
                super::raw_encoder::Compute::Metal3(raw),
                self.clone(),
                Arc::clone(fence),
                Arc::clone(prev_ce_outputs),
                Arc::clone(executor),
            )
        })
        .unwrap()
    }

    /// A concurrent compute encoder whose stage start and end write timestamps
    /// into `samples` at `index` and `index + 1` (the profiler's counter arm,
    /// `profile::sample_encoder`).
    fn sampled_encoder(
        &self,
        samples: &ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>,
        index: usize,
    ) -> Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>> {
        let d = objc2_metal::MTLComputePassDescriptor::computePassDescriptor();
        d.setDispatchType(MTLDispatchType::Concurrent);
        // SAFETY: attachment 0 always exists (the array is preallocated), and
        // `sample_encoder` keeps `index + 1` below the buffer's sample count.
        unsafe {
            let a = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
            a.setSampleBuffer(Some(samples));
            a.setStartOfEncoderSampleIndex(index);
            a.setEndOfEncoderSampleIndex(index + 1);
        }
        self.raw().computeCommandEncoderWithDescriptor(&d)
    }

    /// Create a compute command encoder with freshly allocated fence and a standalone output map.
    /// Used by tests and `EncoderProvider` implementations that don't share a global fence map.
    pub fn compute_command_encoder_no_fence(&self) -> ComputeCommandEncoder {
        let fence = Arc::new(Fence::new(&self.device()));
        self.compute_command_encoder(
            &fence,
            &Arc::new(crate::kernels::metal::contention::Censused::new(
                parking_lot::RwLock::new(std::collections::HashMap::new()),
            )),
        )
    }

    pub fn blit_command_encoder(
        &self,
        fence: &Arc<Fence>,
        prev_ce_outputs: &PrevCeOutputs,
    ) -> BlitCommandEncoder {
        if let Backend::Metal4(s) = &self.backend {
            return BlitCommandEncoder::with_backend(
                super::raw_encoder::Blit::Metal4(s.encoder()),
                Arc::clone(fence),
                Arc::clone(prev_ce_outputs),
            );
        }
        self.raw()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(fence), Arc::clone(prev_ce_outputs)))
            .unwrap()
    }

    pub fn commit(&self) {
        match &self.backend {
            Backend::Metal3(raw) => raw.commit(),
            Backend::Metal4(s) => s.commit(),
        }
    }

    pub fn enqueue(&self) {
        match &self.backend {
            Backend::Metal3(raw) => raw.enqueue(),
            Backend::Metal4(s) => s.enqueue(),
        }
    }

    pub fn set_label(&self, label: &str) {
        match &self.backend {
            Backend::Metal3(raw) => raw.setLabel(Some(&NSString::from_str(label))),
            Backend::Metal4(s) => s.set_label(label),
        }
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        match &self.backend {
            Backend::Metal3(raw) => raw.status(),
            Backend::Metal4(s) => s.status(),
        }
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        match &self.backend {
            Backend::Metal3(raw) => raw
                .error()
                .map(|e| Cow::Owned(e.localizedDescription().to_string())),
            Backend::Metal4(s) => s.error().map(Cow::Owned),
        }
    }

    /// GPU start timestamp, valid after completion, in the native uptime clock.
    pub fn gpu_start_time(&self) -> f64 {
        match &self.backend {
            Backend::Metal3(raw) => raw.GPUStartTime(),
            Backend::Metal4(s) => s.gpu_start_time(),
        }
    }

    /// GPU end timestamp, valid after completion, in the native uptime clock.
    pub fn gpu_end_time(&self) -> f64 {
        match &self.backend {
            Backend::Metal3(raw) => raw.GPUEndTime(),
            Backend::Metal4(s) => s.gpu_end_time(),
        }
    }

    pub fn wait_until_completed(&self) {
        match &self.backend {
            Backend::Metal3(raw) => raw.waitUntilCompleted(),
            Backend::Metal4(s) => s.wait(),
        }
    }

    /// Encode a signal of `event` to `value` once every command encoded so far
    /// has completed. Metal's precondition — no open encoder on the buffer —
    /// is the caller's (`Commands` ends its encoder first).
    pub fn signal_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) {
        if let Backend::Metal4(s) = &self.backend {
            s.signal(event, value);
            return;
        }
        self.raw()
            .encodeSignalEvent_value(ProtocolObject::from_ref(event), value);
    }

    /// Encode a wait for `event` to reach `value` before any command encoded
    /// after this one runs. Same precondition as [`Self::signal_event`].
    pub fn wait_for_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) {
        if let Backend::Metal4(s) = &self.backend {
            s.queue
                .raw
                .waitForEvent_value(ProtocolObject::from_ref(event), value);
            return;
        }
        self.raw()
            .encodeWaitForEvent_value(ProtocolObject::from_ref(event), value);
    }

    /// Run `f` once this command buffer has completed on the GPU.
    ///
    /// Must be called before `commit`: Metal rejects handlers added to a buffer
    /// that has already been committed.
    pub fn on_completion<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        if let Backend::Metal4(s) = &self.backend {
            s.on_completion(move |_| f());
            return;
        }
        use block2::RcBlock;
        use std::ptr::NonNull;

        let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| f());
        // SAFETY: `addCompletedHandler` copies the block, and `RcBlock` keeps it
        // alive until Metal releases it. The block captures only owned state.
        unsafe { self.raw().addCompletedHandler(RcBlock::as_ptr(&block)) };
    }

    /// As [`Self::on_completion`], handing `f` the final error, if any,
    /// independently of the native command API.
    pub fn on_completion_of<F>(&self, f: F)
    where
        F: Fn(Option<String>) + Send + Sync + 'static,
    {
        if let Backend::Metal4(s) = &self.backend {
            s.on_completion(move |feedback| f(feedback.error.clone()));
            return;
        }
        use block2::RcBlock;
        use std::ptr::NonNull;

        let block = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal passes its handler the buffer that completed,
            // valid for the duration of the call.
            let cb = unsafe { cb.as_ref() };
            f((cb.status() == MTLCommandBufferStatus::Error).then(|| {
                cb.error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into())
            }))
        });
        // SAFETY: as `on_completion`.
        unsafe { self.raw().addCompletedHandler(RcBlock::as_ptr(&block)) };
    }

    /// Register a completion handler that records this buffer's GPU execution
    /// interval, when profiling is enabled.
    ///
    /// Must be called before `commit`: Metal rejects handlers added to a buffer
    /// that has already been committed. `GPUStartTime`/`GPUEndTime` are only
    /// meaningful once the buffer has completed, which is what makes a handler
    /// the only place to read them.
    pub fn record_gpu_time_on_completion(&self) {
        use block2::RcBlock;
        use std::ptr::NonNull;

        use crate::kernels::metal::profile;
        if !profile::enabled() {
            return;
        }
        // The commit stamp and the step are taken HERE, on the committing
        // thread and in commit order -- the handler runs later, on Metal's
        // thread, when neither is knowable. Registration immediately precedes
        // the `commit` at both call sites, so the stamp reads a few
        // microseconds early (issue #702: the bias is toward "seam").
        let committed = profile::now_s();
        let step = profile::current_step();
        if let Backend::Metal4(s) = &self.backend {
            s.on_completion(move |feedback| {
                profile::record_command_buffer(feedback.start, feedback.end, committed, step);
            });
            return;
        }
        let block = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal hands the completed command buffer to its own
            // completion handler, so the pointer is valid and non-null for the
            // duration of the call. Only the two timing properties are read,
            // and both are plain scalars valid in the completed state.
            let key = cb.as_ptr() as *const () as usize;
            let cb = unsafe { cb.as_ref() };
            let (start, end) = (cb.GPUStartTime(), cb.GPUEndTime());
            profile::record_command_buffer(start, end, committed, step);
            profile::resolve_encoder_samples(key, start, end);
        });
        unsafe { self.raw().addCompletedHandler(RcBlock::as_ptr(&block)) };
    }
}
