use super::{in_flight::Held, Device, InFlightHold};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFence;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cross-encoder synchronization primitive for HazardTrackingModeUntracked resources.
///
/// memoryBarrierWithScope only works within a single encoder. When transitioning between
/// encoders (e.g. compute -> blit for to_cpu, blit -> compute for next dispatch), untracked
/// resources need explicit fence wait/update calls to ensure memory visibility.
///
/// A fence is named by the encoder that updates it and by every later encoder
/// that waits on it, so under an unretained command buffer its `MTLFence` must
/// outlive all of them: the last handle's drop hands it to the queue's
/// [`InFlightHold`] rather than releasing it (`in_flight.rs`).
pub struct Fence {
    raw: ManuallyDrop<Retained<ProtocolObject<dyn MTLFence>>>,
    /// Raised by the completion handler of the command buffer that updates
    /// this fence. A wait on a completed fence orders nothing, so the host
    /// skips it; the handler touches nothing else (`DESIGN.md` §6.4d). Its own
    /// allocation, so the handler holds the flag and never the fence: the
    /// fence's last drop then always happens on the host.
    completed: Arc<AtomicBool>,
    hold: Option<Arc<InFlightHold>>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub fn new(device: &Device) -> Self {
        Self::held_by(device, None)
    }

    /// A fence whose last drop goes to `hold` when there is one.
    pub fn held_by(device: &Device, hold: Option<Arc<InFlightHold>>) -> Self {
        use objc2_metal::MTLDevice as _;
        let raw = device
            .as_ref()
            .newFence()
            .expect("failed to create MTLFence");
        Fence {
            raw: ManuallyDrop::new(raw),
            completed: Arc::new(AtomicBool::new(false)),
            hold,
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

    /// The flag [`Self::mark_completed`] raises, for a completion handler to
    /// capture in place of the fence.
    pub fn completion_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.completed)
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLFence> {
        &self.raw
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: `raw` is taken exactly once, here, and never read after.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        if let Some(hold) = &self.hold {
            hold.hold(Held::Fence(raw));
        }
    }
}
