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
    Metal3Associated(super::metal3_retirement::AssociatedBuffer),
    Metal4(Arc<super::metal4::Submission>),
}

/// Prepared transaction shared by public submission and engine callers.
pub(super) enum PreparedCommandCommit<'a> {
    Metal3(&'a Retained<ProtocolObject<dyn MTLCommandBuffer>>),
    Metal3Associated(super::metal3_retirement::PreparedCommit<'a>),
    Metal4(super::metal4::PreparedCommit<'a>),
}
impl PreparedCommandCommit<'_> {
    pub(super) fn commit(self) -> Result<(), crate::kernels::MetalKernelError> {
        match self {
            Self::Metal3(raw) => {
                raw.commit();
                Ok(())
            }
            Self::Metal3Associated(prepared) => prepared.commit(),
            Self::Metal4(prepared) => prepared.commit(),
        }
    }
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
    pub(super) fn prepare_commit(
        &self,
    ) -> Result<PreparedCommandCommit<'_>, crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3(raw) => {
                if raw.status().0 >= MTLCommandBufferStatus::Committed.0 {
                    return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                        "standalone buffer already committed",
                    ));
                }
                Ok(PreparedCommandCommit::Metal3(raw))
            }
            Backend::Metal3Associated(buffer) => buffer
                .prepare_commit()
                .map(PreparedCommandCommit::Metal3Associated),
            Backend::Metal4(submission) => submission
                .prepare_commit()
                .map(PreparedCommandCommit::Metal4),
        }
    }

    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            backend: Backend::Metal3(raw),
        }
    }

    pub(super) fn from_metal3_associated(
        buffer: super::metal3_retirement::AssociatedBuffer,
    ) -> Self {
        Self {
            backend: Backend::Metal3Associated(buffer),
        }
    }

    pub(super) fn associate_metal3(
        self,
        core: Arc<super::metal3_retirement::Completions>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Result<Self, crate::kernels::MetalKernelError> {
        match self.backend {
            Backend::Metal3(raw) => {
                super::metal3_retirement::AssociatedBuffer::new(raw, core, physical)
                    .map(Self::from_metal3_associated)
            }
            backend => Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "buffer already queue-associated",
                ),
                Some(Self { backend }),
            )
            .into()),
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
            Backend::Metal3Associated(buffer) => buffer.raw(),
            Backend::Metal4(_) => unreachable!("Metal 3-only helper"),
        }
    }
    pub(crate) fn is_metal4(&self) -> bool {
        matches!(self.backend, Backend::Metal4(_))
    }
    pub(crate) fn identity(&self) -> usize {
        match &self.backend {
            Backend::Metal3(raw) => Retained::as_ptr(raw) as usize,
            Backend::Metal3Associated(buffer) => Retained::as_ptr(buffer.raw()) as usize,
            Backend::Metal4(s) => Arc::as_ptr(s) as usize,
        }
    }
    pub(crate) fn device(&self) -> Device {
        match &self.backend {
            Backend::Metal3(raw) => Device::new(raw.device()),
            Backend::Metal3Associated(buffer) => Device::new(buffer.raw().device()),
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

    pub fn commit(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.prepare_commit()?.commit()
    }

    pub fn enqueue(&self) {
        match &self.backend {
            Backend::Metal3(raw) => raw.enqueue(),
            Backend::Metal3Associated(buffer) => buffer.raw().enqueue(),
            Backend::Metal4(s) => s.enqueue(),
        }
    }

    pub fn set_label(&self, label: &str) {
        match &self.backend {
            Backend::Metal3(raw) => raw.setLabel(Some(&NSString::from_str(label))),
            Backend::Metal3Associated(buffer) => {
                buffer.raw().setLabel(Some(&NSString::from_str(label)))
            }
            Backend::Metal4(s) => s.set_label(label),
        }
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        match &self.backend {
            Backend::Metal3(raw) => raw.status(),
            Backend::Metal3Associated(buffer) => buffer.raw().status(),
            Backend::Metal4(s) => s.status(),
        }
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        match &self.backend {
            Backend::Metal3(raw) => raw
                .error()
                .map(|e| Cow::Owned(e.localizedDescription().to_string())),
            Backend::Metal3Associated(buffer) => buffer
                .raw()
                .error()
                .map(|e| Cow::Owned(e.localizedDescription().to_string())),
            Backend::Metal4(s) => s.error().map(Cow::Owned),
        }
    }

    /// GPU start timestamp, valid after completion, in the native uptime clock.
    pub fn gpu_start_time(&self) -> f64 {
        match &self.backend {
            Backend::Metal3(raw) => raw.GPUStartTime(),
            Backend::Metal3Associated(buffer) => buffer.raw().GPUStartTime(),
            Backend::Metal4(s) => s.gpu_start_time(),
        }
    }

    /// GPU end timestamp, valid after completion, in the native uptime clock.
    pub fn gpu_end_time(&self) -> f64 {
        match &self.backend {
            Backend::Metal3(raw) => raw.GPUEndTime(),
            Backend::Metal3Associated(buffer) => buffer.raw().GPUEndTime(),
            Backend::Metal4(s) => s.gpu_end_time(),
        }
    }

    pub(super) fn data_ready(&self) -> bool {
        self.status().0 >= MTLCommandBufferStatus::Completed.0
    }

    /// Internal data/error boundary: does not promise own callback return.
    /// Engine-owned payload custody remains registered independently.
    pub(super) fn try_wait_data_ready(&self) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3(raw) => {
                if raw.status().0 < MTLCommandBufferStatus::Completed.0 {
                    raw.waitUntilCompleted();
                }
                Ok(())
            }
            Backend::Metal3Associated(buffer) => buffer.try_wait(false),
            Backend::Metal4(submission) => submission.try_wait_data_ready(),
        }
    }
    pub(super) fn check_data_wait(&self) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3(_) => Ok(()), // Standalone raw wrappers have no engine queue poison.
            Backend::Metal3Associated(buffer) => buffer.check_data_wait(),
            Backend::Metal4(submission) => submission.check_data_wait(),
        }
    }

    /// Wait for own logical completion, returning native or adapter failures.
    /// Queue-associated wrappers additionally preserve engine prefix ordering.
    /// Standalone M3 retains the native handler-join contract, not a cross-buffer
    /// logical prefix. Error never establishes teardown quiescence.
    pub fn wait_until_completed(&self) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3(raw) => {
                let queue = raw.commandQueue();
                crate::completion_scope::before_park(Retained::as_ptr(&queue) as usize)
                    .map_err(|_| crate::kernels::MetalKernelError::CallbackWouldBlock)?;
                raw.waitUntilCompleted();
            }
            Backend::Metal3Associated(buffer) => buffer.try_wait(true)?,
            Backend::Metal4(submission) => submission.wait()?,
        }
        if self.status() == MTLCommandBufferStatus::Error {
            return Err(crate::kernels::MetalKernelError::CommandBufferError(
                self.error()
                    .map(|error| error.into_owned())
                    .unwrap_or_else(|| "unknown error".into()),
            ));
        }
        Ok(())
    }
    pub fn try_wait_until_completed(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.wait_until_completed()
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

    /// Register before commit; rejected captures remain owned by the returned
    /// error. Propagate that error beyond recording locks before dropping it.
    pub fn on_completion<F: Fn() + Send + Sync + 'static>(
        &self,
        f: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        if let Backend::Metal3(raw) = &self.backend {
            if raw.status().0 >= MTLCommandBufferStatus::Committed.0 {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    crate::kernels::MetalKernelError::QueueUnavailable(
                        "standalone buffer already committed",
                    ),
                    Some(f),
                ));
            }
            let queue = raw.commandQueue();
            let identity = Retained::as_ptr(&queue) as usize;
            // Preserve the no-feedback standalone spelling: no new native
            // error/timestamp reads merely to run a zero-argument callback.
            let block = block2::RcBlock::new(
                move |_cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                    crate::completion_scope::with_callback(identity, &f);
                },
            );
            // SAFETY: native API copies this owned block before return.
            unsafe {
                raw.addCompletedHandler(block2::RcBlock::as_ptr(&block));
            }
            return Ok(());
        }
        self.on_feedback(move |_| f())
    }
    pub fn on_completion_of<F>(
        &self,
        f: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError>
    where
        F: Fn(Option<String>) + Send + Sync + 'static,
    {
        self.on_feedback(move |feedback| f(feedback.error.clone()))
    }
    fn on_feedback<F>(&self, f: F) -> Result<(), crate::kernels::err::CompletionRegistrationError>
    where
        F: Fn(&super::metal4::Feedback) + Send + Sync + 'static,
    {
        use crate::kernels::{err::CompletionRegistrationError, MetalKernelError};
        match &self.backend {
            Backend::Metal4(submission) => return submission.on_completion(f),
            Backend::Metal3Associated(buffer) => {
                return buffer.try_on_completion(Box::new(f)).map_err(|rejected| {
                    CompletionRegistrationError::new(rejected.error, rejected.callback)
                });
            }
            Backend::Metal3(raw) => {
                if raw.status().0 >= MTLCommandBufferStatus::Committed.0 {
                    return Err(CompletionRegistrationError::new(
                        MetalKernelError::QueueUnavailable("standalone buffer already committed"),
                        Some(f),
                    ));
                }
                let queue = raw.commandQueue();
                let identity = Retained::as_ptr(&queue) as usize;
                let block = block2::RcBlock::new(
                    move |cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        // SAFETY: Metal supplies the completed CB for this invocation.
                        let cb = unsafe { cb.as_ref() };
                        let feedback = super::metal4::Feedback {
                            error: (cb.status() == MTLCommandBufferStatus::Error).then(|| {
                                cb.error()
                                    .map(|error| error.localizedDescription().to_string())
                                    .unwrap_or_else(|| "unknown error".into())
                            }),
                            start: cb.GPUStartTime(),
                            end: cb.GPUEndTime(),
                        };
                        crate::completion_scope::with_callback(identity, || f(&feedback));
                    },
                );
                // SAFETY: native API copies the block. Standalone wrappers keep
                // native custody semantics; they do not join engine registry.
                unsafe {
                    raw.addCompletedHandler(block2::RcBlock::as_ptr(&block));
                }
            }
        }
        Ok(())
    }

    pub fn record_gpu_time_on_completion(
        &self,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        use crate::kernels::metal::profile;
        if !profile::enabled() {
            return Ok(());
        }
        let committed = profile::now_s();
        let step = profile::current_step();
        let native_key = (!self.is_metal4()).then(|| self.identity());
        self.on_feedback(move |feedback| {
            profile::record_command_buffer(feedback.start, feedback.end, committed, step);
            if let Some(key) = native_key {
                profile::resolve_encoder_samples(key, feedback.start, feedback.end);
            }
        })
    }
}
