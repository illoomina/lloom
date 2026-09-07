use super::{
    executor::ExecutorSlot, BlitCommandEncoder, ComputeCommandEncoder, Device, Fence, PrevCeOutputs,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLDispatchType};
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { raw }
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
        self.as_ref()
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .map(|raw| {
                ComputeCommandEncoder::with_executor(
                    raw,
                    self.raw.clone(),
                    Arc::clone(fence),
                    Arc::clone(prev_ce_outputs),
                    Arc::clone(executor),
                )
            })
            .unwrap()
    }

    /// Create a compute command encoder with freshly allocated fence and a standalone output map.
    /// Used by tests and `EncoderProvider` implementations that don't share a global fence map.
    pub fn compute_command_encoder_no_fence(&self) -> ComputeCommandEncoder {
        let device = Device::new(self.raw.device());
        let fence = Arc::new(Fence::new(&device));
        self.as_ref()
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .map(|raw| {
                ComputeCommandEncoder::new(
                    raw,
                    self.raw.clone(),
                    fence,
                    Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
                )
            })
            .unwrap()
    }

    pub fn blit_command_encoder(
        &self,
        fence: &Arc<Fence>,
        prev_ce_outputs: &PrevCeOutputs,
    ) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(fence), Arc::clone(prev_ce_outputs)))
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }

    /// Run `f` once this command buffer has completed on the GPU.
    ///
    /// Must be called before `commit`: Metal rejects handlers added to a buffer
    /// that has already been committed.
    pub fn on_completion<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        use block2::RcBlock;
        use std::ptr::NonNull;

        let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| f());
        // SAFETY: `addCompletedHandler` copies the block, and `RcBlock` keeps it
        // alive until Metal releases it. The block captures only owned state.
        unsafe { self.raw.addCompletedHandler(RcBlock::as_ptr(&block)) };
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

        if !crate::kernels::metal::profile::enabled() {
            return;
        }
        let block = RcBlock::new(|cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal hands the completed command buffer to its own
            // completion handler, so the pointer is valid and non-null for the
            // duration of the call. Only the two timing properties are read,
            // and both are plain scalars valid in the completed state.
            let cb = unsafe { cb.as_ref() };
            crate::kernels::metal::profile::record_command_buffer(
                cb.GPUStartTime(),
                cb.GPUEndTime(),
            );
        });
        unsafe { self.raw.addCompletedHandler(RcBlock::as_ptr(&block)) };
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
