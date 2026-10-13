use super::{in_flight::Held, Device, GpuClock, InFlightHold};
use crate::retire::Epoch;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFence;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// `closed_at` before the fence's encoder has closed on its queue's clock.
const OPEN: u64 = u64::MAX;

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
///
/// A wait on a completed fence orders nothing, so the host skips it. A fence
/// of a [`Commands`](super::Commands) queue completes with the epoch of the
/// command buffer that updates it, on the queue's clock; any other fence,
/// when a completion handler raises its flag (`DESIGN.md` §6.4d).
pub struct Fence {
    raw: ManuallyDrop<Retained<ProtocolObject<dyn MTLFence>>>,
    /// Raised by the completion handler of the command buffer that updates a
    /// fence with no clock; the handler touches nothing else. Its own
    /// allocation, so the handler holds the flag and never the fence: the
    /// fence's last drop then always happens on the host.
    completed: Arc<AtomicBool>,
    /// The queue clock, and the epoch its updating command buffer commits as
    /// ([`Self::close_on_clock`]); [`OPEN`] until then.
    clock: Option<Arc<GpuClock>>,
    closed_at: AtomicU64,
    hold: Option<Arc<InFlightHold>>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub fn new(device: &Device) -> Self {
        Self::held_by(device, None, None)
    }

    /// A fence on a queue's `clock`, whose last drop goes to `hold` when there
    /// is one.
    pub fn held_by(
        device: &Device,
        clock: Option<Arc<GpuClock>>,
        hold: Option<Arc<InFlightHold>>,
    ) -> Self {
        use objc2_metal::MTLDevice as _;
        let raw = device
            .as_ref()
            .newFence()
            .expect("failed to create MTLFence");
        Fence {
            raw: ManuallyDrop::new(raw),
            completed: Arc::new(AtomicBool::new(false)),
            clock,
            closed_at: AtomicU64::new(OPEN),
            hold,
        }
    }

    /// The fence's encoder has closed: it completes with the epoch the
    /// queue's recording command buffer commits as, the clock's current one.
    /// A queue commits only its one recording buffer, and only after its
    /// encoder has closed, so no epoch is taken between this read and that
    /// commit. False for a fence with no clock, whose closer registers a
    /// completion handler on [`Self::completion_flag`] instead.
    pub fn close_on_clock(&self) -> bool {
        let Some(clock) = &self.clock else {
            return false;
        };
        self.closed_at
            .store(clock.current_epoch().0, Ordering::Release);
        true
    }

    /// Record that the command buffer updating this fence has completed.
    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }

    /// Whether the command buffer updating this fence has completed.
    pub fn is_completed(&self) -> bool {
        if self.completed.load(Ordering::Acquire) {
            return true;
        }
        let Some(clock) = &self.clock else {
            return false;
        };
        let epoch = self.closed_at.load(Ordering::Acquire);
        epoch != OPEN && clock.is_complete(Epoch(epoch))
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
