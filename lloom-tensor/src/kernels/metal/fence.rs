use super::Device;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFence;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cross-encoder synchronization primitive for HazardTrackingModeUntracked resources.
///
/// memoryBarrierWithScope only works within a single encoder. When transitioning between
/// encoders (e.g. compute -> blit for to_cpu, blit -> compute for next dispatch), untracked
/// resources need explicit fence wait/update calls to ensure memory visibility.
pub struct Fence {
    raw: Retained<ProtocolObject<dyn MTLFence>>,
    /// Raised by the completion handler of the command buffer that updates
    /// this fence. A wait on a completed fence orders nothing, so the host
    /// skips it; the handler touches nothing else (`DESIGN.md` §6.4d).
    completed: AtomicBool,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub fn new(device: &Device) -> Self {
        use objc2_metal::MTLDevice as _;
        let raw = device
            .as_ref()
            .newFence()
            .expect("failed to create MTLFence");
        Fence {
            raw,
            completed: AtomicBool::new(false),
        }
    }

    /// Record that the command buffer updating this fence has completed.
    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }

    /// Whether the command buffer updating this fence has completed.
    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLFence> {
        &self.raw
    }
}
