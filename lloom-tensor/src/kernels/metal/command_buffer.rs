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
    Metal3(Arc<Standalone>),
    Metal3Associated(super::metal3_retirement::AssociatedBuffer),
    Metal4(Arc<super::metal4::Submission>),
}

/// Cold raw-interop wrapper state shared by public clones. Engine-associated
/// M3 construction bypasses this Arc and uses its reusable metadata cell.
/// Independently rewrapping/committing an escaped raw ObjC object remains an
/// explicit raw-interop boundary, not covered by a wrapper-family gate.
struct Standalone {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    lifecycle: crate::command_lifecycle::CommandLifecycle,
    wake: crate::completion_wake::Wake,
    failure: crate::retirement_owners::FailureSignal,
}
impl std::ops::Deref for Standalone {
    type Target = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

/// Prepared native ordering is released before the lifecycle waiter is woken.
/// A returned error after commit entry closes recording admission; it does not
/// establish native terminality or release registry-owned resources.
pub(super) struct PreparedCommandCommit<'a> {
    native: Option<PreparedNative<'a>>,
    admission: RecordingAdmission<'a>,
}
enum PreparedNative<'a> {
    Metal3(&'a Retained<ProtocolObject<dyn MTLCommandBuffer>>),
    Metal3Associated(super::metal3_retirement::PreparedCommit<'a>),
    Metal4(super::metal4::PreparedCommit<'a>),
}
struct RecordingAdmission<'a> {
    buffer: &'a CommandBuffer,
    ticket: crate::command_lifecycle::Ticket,
    entered: bool,
    editing: bool,
    released: bool,
    notify_unused: bool,
}
fn lifecycle_error(
    reason: crate::command_lifecycle::LifecycleError,
) -> crate::kernels::MetalKernelError {
    crate::kernels::MetalKernelError::RecordingAdmission(reason)
}
impl RecordingAdmission<'_> {
    fn enter_commit(&mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.buffer
            .lifecycle()
            .entered_native(&mut self.ticket)
            .map_err(lifecycle_error)?;
        self.entered = true;
        Ok(())
    }
    fn returned(&mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.buffer
            .lifecycle()
            .handoff_returned(&mut self.ticket)
            .map_err(lifecycle_error)?;
        self.released = true;
        self.buffer.notify_lifecycle();
        Ok(())
    }
}
impl Drop for RecordingAdmission<'_> {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // Exact phase/serial ownership prevents a stale guard closing a later
        // encoder. Unknown commit exit is not reset to a reusable recording.
        let transition = if self.entered || std::thread::panicking() {
            self.buffer.lifecycle().abandon(&mut self.ticket)
        } else if self.editing {
            self.buffer.lifecycle().finish_edit(&mut self.ticket)
        } else {
            self.buffer.lifecycle().cancel_prepared(&mut self.ticket)
        };
        if transition.is_err() {
            // Refusal is not native terminality and must not be hidden by the
            // private retry's no-self-wake optimization. Custody remains owned.
            self.buffer.fail_lifecycle();
        } else if self.notify_unused || self.entered || self.editing || std::thread::panicking() {
            self.buffer.notify_lifecycle();
        }
    }
}
impl PreparedCommandCommit<'_> {
    pub(super) fn on_completion_of<F>(
        &self,
        f: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError>
    where
        F: Fn(Option<String>) + Send + Sync + 'static,
    {
        self.on_feedback(move |feedback| f(feedback.error.clone()))
    }
    pub(super) fn record_gpu_time_on_completion(
        &self,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        if let Some(callback) = self.admission.buffer.gpu_time_callback() {
            self.on_feedback(callback)?;
        }
        Ok(())
    }
    pub(super) fn signal_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        if let Some(PreparedNative::Metal4(prepared)) = &self.native {
            return prepared.signal_tail(event, value);
        }
        self.admission.buffer.signal_event_unchecked(event, value)
    }
    fn on_feedback<F>(
        &self,
        callback: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError>
    where
        F: Fn(&super::metal4::Feedback) + Send + Sync + 'static,
    {
        if let Some(PreparedNative::Metal4(prepared)) = &self.native {
            return prepared.on_completion(callback);
        }
        self.admission.buffer.on_feedback(callback)
    }
    pub(super) fn commit(mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.admission.enter_commit()?;
        let result = match self.native.take().expect("prepared native transaction") {
            PreparedNative::Metal3(raw) => {
                raw.commit();
                Ok(())
            }
            PreparedNative::Metal3Associated(prepared) => prepared.commit(),
            PreparedNative::Metal4(prepared) => prepared.commit(),
        };
        // Consuming the native transaction released its ordering guard before
        // either this notification or the admission guard's error notification.
        result?;
        self.admission.returned()
    }
}

/// One encoder lifetime, shared through the existing EncoderState for compute.
/// The parent wrapper pins lifecycle metadata and native physical custody.
pub(super) struct EncoderAdmission {
    buffer: CommandBuffer,
    ticket: Option<crate::command_lifecycle::Ticket>,
}
/// Trusted registration capability borrowed only while closing an encoder.
/// It avoids reacquiring public EDIT admission while ENDING owns the buffer.
pub(super) struct EndingEncoder<'a> {
    buffer: &'a CommandBuffer,
}
impl<'a> EndingEncoder<'a> {
    pub(super) fn raw_compat(buffer: &'a CommandBuffer) -> Self {
        Self { buffer }
    }
    pub(super) fn on_completion<F: Fn() + Send + Sync + 'static>(
        &self,
        f: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        self.buffer.on_completion_unchecked(f)
    }
}
impl EncoderAdmission {
    pub(super) fn notify_lifecycle(&self) {
        self.buffer.notify_lifecycle();
    }
    fn begin(buffer: &CommandBuffer) -> Result<Self, crate::kernels::MetalKernelError> {
        buffer.check_lifecycle_failure()?;
        let buffer = buffer.clone();
        let ticket = buffer
            .lifecycle()
            .begin_encoder()
            .map_err(lifecycle_error)?;
        Ok(Self {
            buffer,
            ticket: Some(ticket),
        })
    }
    fn cancel_creation(&mut self) -> Result<(), crate::kernels::MetalKernelError> {
        let ticket = self.ticket.as_mut().ok_or_else(|| {
            crate::kernels::MetalKernelError::QueueUnavailable("encoder already closed")
        })?;
        self.buffer
            .lifecycle()
            .cancel_encoder_creation(ticket)
            .map_err(lifecycle_error)?;
        self.ticket = None;
        self.buffer.notify_lifecycle();
        Ok(())
    }
    /// The caller releases shared encoder state BEFORE notify_lifecycle. The
    /// close callback contains native endEncoding even on registration refusal.
    pub(super) fn end(
        &mut self,
        ended: &mut bool,
        notify_after_close: bool,
        close: impl FnOnce(EndingEncoder<'_>) -> Result<(), crate::kernels::MetalKernelError>,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        self.end_prepared(
            ended,
            notify_after_close,
            || Ok(()),
            |scope, ()| close(scope),
        )
    }
    /// Acquire fallible close prerequisites before ENDING. The prepared guard
    /// moves into close (or is dropped on refusal) before the phase Wake fires.
    pub(super) fn end_prepared<P>(
        &mut self,
        ended: &mut bool,
        notify_after_close: bool,
        prepare: impl FnOnce() -> Result<P, crate::kernels::MetalKernelError>,
        close: impl FnOnce(EndingEncoder<'_>, P) -> Result<(), crate::kernels::MetalKernelError>,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let mut wake = EncoderEndWake::pending(&self.buffer);
        let ticket = self.ticket.as_mut().ok_or_else(|| {
            crate::kernels::MetalKernelError::QueueUnavailable("encoder already closed")
        })?;
        let prepared = prepare()?;
        let result = self
            .buffer
            .lifecycle()
            .end_encoder_with(ticket, ended, || {
                if notify_after_close {
                    wake.arm();
                }
                close(
                    EndingEncoder {
                        buffer: &self.buffer,
                    },
                    prepared,
                )
            })
            .map_err(lifecycle_error)?;
        self.ticket = None;
        result
    }
}
impl Drop for EncoderAdmission {
    fn drop(&mut self) {
        if let Some(ticket) = &mut self.ticket {
            if self.buffer.lifecycle().abandon(ticket).is_err() {
                // A refused owner transition cannot be hidden by a wake: it
                // neither closes admission nor proves native completion.
                self.buffer.fail_lifecycle();
            } else {
                self.buffer.notify_lifecycle();
            }
        }
    }
}

/// Declare before acquiring EncoderState so even unwind releases that lock
/// before waking a host lifecycle poller.
pub(super) struct EncoderEndWake<'a> {
    buffer: &'a CommandBuffer,
    changed: bool,
}
impl<'a> EncoderEndWake<'a> {
    pub(super) fn pending(buffer: &'a CommandBuffer) -> Self {
        Self {
            buffer,
            changed: false,
        }
    }
    pub(super) fn armed(buffer: &'a CommandBuffer) -> Self {
        Self {
            buffer,
            changed: true,
        }
    }
    pub(super) fn arm(&mut self) {
        self.changed = true;
    }
}
impl Drop for EncoderEndWake<'_> {
    fn drop(&mut self) {
        if self.changed {
            self.buffer.notify_lifecycle();
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
    fn lifecycle(&self) -> &crate::command_lifecycle::CommandLifecycle {
        match &self.backend {
            Backend::Metal3(standalone) => &standalone.lifecycle,
            Backend::Metal3Associated(buffer) => buffer.command_lifecycle(),
            Backend::Metal4(submission) => &submission.command_lifecycle,
        }
    }
    fn check_lifecycle_failure(&self) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3(raw) if raw.failure.get().is_some() => {
                Err(crate::kernels::MetalKernelError::QueueUnavailable(
                    "standalone recording lifecycle failed",
                ))
            }
            Backend::Metal3(_) => Ok(()),
            Backend::Metal3Associated(buffer) => buffer.check_lifecycle_failure(),
            Backend::Metal4(submission) => submission.check_lifecycle_failure(),
        }
    }
    fn fail_lifecycle(&self) {
        match &self.backend {
            Backend::Metal3(raw) => raw
                .failure
                .publish(crate::retirement_owners::Failure::AdapterFailed, || {
                    raw.wake.notify()
                }),
            Backend::Metal3Associated(buffer) => buffer.fail_lifecycle(),
            Backend::Metal4(submission) => submission.fail_lifecycle(),
        }
    }
    pub(super) fn notify_lifecycle(&self) {
        match &self.backend {
            Backend::Metal3(standalone) => standalone.wake.notify(),
            Backend::Metal3Associated(buffer) => buffer.notify_lifecycle(),
            Backend::Metal4(submission) => submission.notify_lifecycle(),
        }
    }

    fn edit_recording(&self) -> Result<RecordingAdmission<'_>, crate::kernels::MetalKernelError> {
        self.check_lifecycle_failure()?;
        Ok(RecordingAdmission {
            buffer: self,
            ticket: self.lifecycle().begin_edit().map_err(lifecycle_error)?,
            entered: false,
            editing: true,
            released: false,
            notify_unused: true,
        })
    }

    fn edit_boundary(&self) -> Result<RecordingAdmission<'_>, crate::kernels::MetalKernelError> {
        self.check_lifecycle_failure()?;
        Ok(RecordingAdmission {
            buffer: self,
            ticket: self.lifecycle().begin_boundary().map_err(lifecycle_error)?,
            entered: false,
            editing: true,
            released: false,
            notify_unused: true,
        })
    }

    pub(super) fn prepare_commit(
        &self,
    ) -> Result<PreparedCommandCommit<'_>, crate::kernels::MetalKernelError> {
        self.prepare_commit_mode(false)
    }

    /// Sole host-retirement worker: an unused preparation restores recording
    /// without waking itself. Normal producers wait on native-order mutexes;
    /// unsubmitted logical waits refuse rather than parking on PREPARED.
    /// Native return, error unwind and all public admissions retain notification.
    pub(super) fn try_prepare_commit(
        &self,
    ) -> Result<PreparedCommandCommit<'_>, crate::kernels::MetalKernelError> {
        self.prepare_commit_mode(true)
    }

    fn prepare_commit_mode(
        &self,
        nonblocking: bool,
    ) -> Result<PreparedCommandCommit<'_>, crate::kernels::MetalKernelError> {
        self.check_lifecycle_failure()?;
        let admission = RecordingAdmission {
            buffer: self,
            ticket: self.lifecycle().prepare().map_err(lifecycle_error)?,
            entered: false,
            editing: false,
            released: false,
            notify_unused: !nonblocking,
        };
        let native = match &self.backend {
            Backend::Metal3(raw) => {
                if raw.status().0 >= MTLCommandBufferStatus::Committed.0 {
                    return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                        "standalone buffer already committed",
                    ));
                }
                PreparedNative::Metal3(raw)
            }
            Backend::Metal3Associated(buffer) => {
                PreparedNative::Metal3Associated(buffer.prepare_commit_mode(nonblocking)?)
            }
            Backend::Metal4(submission) => {
                PreparedNative::Metal4(submission.prepare_commit_mode(nonblocking)?)
            }
        };
        Ok(PreparedCommandCommit {
            native: Some(native),
            admission,
        })
    }

    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            backend: Backend::Metal3(Arc::new(Standalone {
                raw,
                lifecycle: crate::command_lifecycle::CommandLifecycle::new(),
                wake: crate::completion_wake::Wake::default(),
                failure: crate::retirement_owners::FailureSignal::default(),
            })),
        }
    }

    pub(crate) fn retain_namespace(
        &self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3Associated(buffer) => buffer.retain_namespace(owner),
            Backend::Metal4(submission) => submission.retain_namespace(owner),
            Backend::Metal3(_) => Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "standalone namespace attachment refused",
                ),
                Some(owner),
            )
            .into()),
        }
    }

    pub(crate) fn attach_physical_owner(
        &self,
        owner: Arc<dyn Send + Sync>,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        match &self.backend {
            Backend::Metal3Associated(buffer) => buffer.attach_physical_owner(owner),
            Backend::Metal4(submission) => submission.queue.attach_physical_owner(owner),
            Backend::Metal3(_) => Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "standalone physical attachment refused",
                ),
                Some(owner),
            )
            .into()),
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
            Backend::Metal3(shared) => match Arc::try_unwrap(shared) {
                Ok(standalone) => {
                    super::metal3_retirement::AssociatedBuffer::new(standalone.raw, core, physical)
                        .map(Self::from_metal3_associated)
                }
                Err(shared) => Err(crate::kernels::err::CompletionRegistrationError::new(
                    crate::kernels::MetalKernelError::QueueUnavailable(
                        "cannot re-associate shared standalone wrapper",
                    ),
                    Some((
                        Self {
                            backend: Backend::Metal3(shared),
                        },
                        physical,
                    )),
                )
                .into()),
            },
            backend => Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "buffer already queue-associated",
                ),
                Some((Self { backend }, physical)),
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
    ) -> Result<ComputeCommandEncoder, crate::kernels::MetalKernelError> {
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
    ) -> Result<ComputeCommandEncoder, crate::kernels::MetalKernelError> {
        let mut admission = EncoderAdmission::begin(self)?;
        if let Backend::Metal4(s) = &self.backend {
            let raw = match s.encoder() {
                Ok(raw) => raw,
                Err(error) => {
                    admission.cancel_creation()?;
                    return Err(error);
                }
            };
            return Ok(ComputeCommandEncoder::with_backend(
                super::raw_encoder::Compute::Metal4(raw),
                self.clone(),
                Arc::clone(fence),
                Arc::clone(prev_ce_outputs),
                Arc::clone(executor),
            )
            .with_recording(admission));
        }
        let raw = match crate::kernels::metal::profile::sample_encoder(self.raw()) {
            Some((samples, index)) => self.sampled_encoder(&samples, index),
            None => self
                .raw()
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent),
        };
        let Some(raw) = raw else {
            admission.cancel_creation()?;
            return Err(crate::kernels::MetalKernelError::FailedToCreateResource(
                "compute encoder".into(),
            ));
        };
        Ok(ComputeCommandEncoder::with_backend(
            super::raw_encoder::Compute::Metal3(raw),
            self.clone(),
            Arc::clone(fence),
            Arc::clone(prev_ce_outputs),
            Arc::clone(executor),
        )
        .with_recording(admission))
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
    pub fn compute_command_encoder_no_fence(
        &self,
    ) -> Result<ComputeCommandEncoder, crate::kernels::MetalKernelError> {
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
    ) -> Result<BlitCommandEncoder, crate::kernels::MetalKernelError> {
        let mut admission = EncoderAdmission::begin(self)?;
        if let Backend::Metal4(s) = &self.backend {
            let raw = match s.encoder() {
                Ok(raw) => raw,
                Err(error) => {
                    admission.cancel_creation()?;
                    return Err(error);
                }
            };
            return Ok(BlitCommandEncoder::with_backend(
                super::raw_encoder::Blit::Metal4(raw),
                Arc::clone(fence),
                Arc::clone(prev_ce_outputs),
            )
            .with_recording(admission));
        }
        let Some(raw) = self.raw().blitCommandEncoder() else {
            admission.cancel_creation()?;
            return Err(crate::kernels::MetalKernelError::FailedToCreateResource(
                "blit encoder".into(),
            ));
        };
        Ok(
            BlitCommandEncoder::new(raw, Arc::clone(fence), Arc::clone(prev_ce_outputs))
                .with_recording(admission),
        )
    }

    pub fn commit(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.prepare_commit()?.commit()
    }

    pub fn enqueue(&self) -> Result<(), crate::kernels::MetalKernelError> {
        let _admission = self.edit_recording()?;
        if self.status() != MTLCommandBufferStatus::NotEnqueued {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "command buffer already enqueued",
            ));
        }
        match &self.backend {
            Backend::Metal3(raw) => raw.enqueue(),
            Backend::Metal3Associated(buffer) => buffer.raw().enqueue(),
            Backend::Metal4(s) => s.enqueue()?,
        }
        Ok(())
    }

    pub fn set_label(&self, label: &str) -> Result<(), crate::kernels::MetalKernelError> {
        let _admission = self.edit_recording()?;
        match &self.backend {
            Backend::Metal3(raw) => raw.setLabel(Some(&NSString::from_str(label))),
            Backend::Metal3Associated(buffer) => {
                buffer.raw().setLabel(Some(&NSString::from_str(label)))
            }
            Backend::Metal4(s) => s.set_label(label)?,
        }
        Ok(())
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
        self.check_data_wait()?;
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
    fn check_wait_admission(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.check_lifecycle_failure()?;
        match &self.backend {
            Backend::Metal3(raw) => {
                let imported = matches!(
                    raw.status(),
                    MTLCommandBufferStatus::Committed
                        | MTLCommandBufferStatus::Scheduled
                        | MTLCommandBufferStatus::Completed
                        | MTLCommandBufferStatus::Error
                );
                self.lifecycle().check_standalone_wait(imported)
            }
            _ => self.lifecycle().check_wait_submission(),
        }
        .map_err(lifecycle_error)
    }
    pub(super) fn check_data_wait(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.check_lifecycle_failure()?;
        match &self.backend {
            Backend::Metal3(_) => Ok(()), // Wrapper-family failure was checked above.
            Backend::Metal3Associated(buffer) => buffer.check_data_wait(),
            Backend::Metal4(submission) => submission.check_data_wait(),
        }?;
        // Independently observed native terminality permits data/error inspection,
        // not callback join or physical teardown. Otherwise never park before a
        // known-returned submission, including ENTERED/unknown handoff.
        if !self.data_ready() {
            self.check_wait_admission()?;
            if let Backend::Metal3(raw) = &self.backend {
                let queue = raw.commandQueue();
                crate::completion_scope::before_park(Retained::as_ptr(&queue) as usize)
                    .map_err(|_| crate::kernels::MetalKernelError::CallbackWouldBlock)?;
            }
        }
        Ok(())
    }

    /// Wait for own logical completion, returning native or adapter failures.
    /// Queue-associated wrappers additionally preserve engine prefix ordering.
    /// Standalone M3 retains the native handler-join contract, not a cross-buffer
    /// logical prefix. Before explicit commit returns, this refuses NotSubmitted
    /// or Busy without submitting/cancelling work or awaiting a future producer.
    /// Standalone idle raw wrappers also accept independently observed native
    /// submission, preserving imported/external raw commit semantics. Their owned
    /// ENTERED or unknown handoff still refuses Busy/Abandoned; associated wrappers
    /// do not substitute native status for the logical prefix contract.
    /// Error never establishes teardown quiescence.
    pub fn wait_until_completed(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.check_wait_admission()?;
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

    /// Final teardown's nonblocking associated-queue readiness. A standalone
    /// native status cannot prove that its arbitrary handlers have returned.
    pub(super) fn poll_logical_completed(&self) -> Result<bool, crate::kernels::MetalKernelError> {
        let ready = match &self.backend {
            Backend::Metal3(_) => {
                return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                    "standalone callback join has no nonblocking completion proof",
                ))
            }
            Backend::Metal3Associated(buffer) => buffer.poll_logical_completed()?,
            Backend::Metal4(submission) => submission.poll_logical_completed()?,
        };
        if ready && self.status() == MTLCommandBufferStatus::Error {
            return Err(crate::kernels::MetalKernelError::CommandBufferError(
                self.error()
                    .map(|error| error.into_owned())
                    .unwrap_or_else(|| "unknown error".into()),
            ));
        }
        Ok(ready)
    }

    /// Encode a signal of `event` to `value` once every command encoded so far
    /// has completed. Metal's precondition — no open encoder on the buffer —
    /// is enforced by boundary admission (`Commands` ends its encoder first).
    pub fn signal_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let _admission = self.edit_boundary()?;
        self.signal_event_unchecked(event, value)
    }
    fn signal_event_unchecked(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        if let Backend::Metal4(s) = &self.backend {
            return s.signal(event, value);
        }
        self.raw()
            .encodeSignalEvent_value(ProtocolObject::from_ref(event), value);
        Ok(())
    }

    /// Encode a wait for `event` to reach `value` before any command encoded
    /// after this one runs. Same precondition as [`Self::signal_event`].
    pub fn wait_for_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let _admission = self.edit_boundary()?;
        if let Backend::Metal4(s) = &self.backend {
            return s.wait_event(event, value);
        }
        self.raw()
            .encodeWaitForEvent_value(ProtocolObject::from_ref(event), value);
        Ok(())
    }

    /// Register before commit; rejected captures remain owned by the returned
    /// error. Propagate that error beyond recording locks before dropping it.
    pub fn on_completion<F: Fn() + Send + Sync + 'static>(
        &self,
        f: F,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        let _admission = match self.edit_recording() {
            Ok(admission) => admission,
            Err(error) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    error,
                    Some(f),
                ))
            }
        };
        self.on_completion_unchecked(f)
    }

    fn on_completion_unchecked<F: Fn() + Send + Sync + 'static>(
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
        let _admission = match self.edit_recording() {
            Ok(admission) => admission,
            Err(error) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    error,
                    Some(f),
                ))
            }
        };
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
        let _admission = self.edit_recording().map_err(|error| {
            crate::kernels::err::CompletionRegistrationError::new(error, None::<()>)
        })?;
        self.record_gpu_time_unchecked()
    }

    fn record_gpu_time_unchecked(
        &self,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        if let Some(callback) = self.gpu_time_callback() {
            self.on_feedback(callback)?;
        }
        Ok(())
    }

    fn gpu_time_callback(
        &self,
    ) -> Option<impl Fn(&super::metal4::Feedback) + Send + Sync + 'static> {
        use crate::kernels::metal::profile;
        if !profile::enabled() {
            return None;
        }
        let committed = profile::now_s();
        let step = profile::current_step();
        let native_key = (!self.is_metal4()).then(|| self.identity());
        Some(move |feedback: &super::metal4::Feedback| {
            profile::record_command_buffer(feedback.start, feedback.end, committed, step);
            if let Some(key) = native_key {
                profile::resolve_encoder_samples(key, feedback.start, feedback.end);
            }
        })
    }
}
