//! Native payload custody for the shared host retirement registry.
//!
//! Draft adapter types; no native commit site uses them until queue-ordered
//! publication, wait failure propagation and every handoff guard are installed.
use super::metal4;
use crate::{
    retirement_owners::FailureSignal,
    retirement_registry::{OwnerRegistry, Registration, RegistryError},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandBuffer;
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) type Callback = Box<dyn Fn(&metal4::Feedback) + Send + Sync>;
pub(super) type Registry = OwnerRegistry<Captures, NativeOwners, metal4::Feedback>;
pub(super) type Record = Registration<Captures, NativeOwners, metal4::Feedback>;

pub(super) struct Captures(pub Vec<Callback>);
impl Captures {
    pub fn run(&self, feedback: &metal4::Feedback) {
        // Borrow callback boxes; their destruction belongs to the host service.
        for callback in &self.0 {
            callback(feedback);
        }
    }
}

/// No arbitrary callback boxes here. Declaration order keeps native resources
/// and submission state alive before releasing the physical aggregate.
pub(super) struct NativeOwners {
    backend: NativeBackend,
    physical: Option<Arc<dyn Send + Sync>>,
}
enum NativeBackend {
    Metal3(Retained<ProtocolObject<dyn MTLCommandBuffer>>),
    Metal4 {
        slot: Mutex<Option<metal4::Slot>>,
        submission: Arc<metal4::Submission>,
    },
}
// SAFETY: Metal CB wrappers already permit cross-thread retention. The M4 slot
// is accessed through a completion-only mutex after recording has moved it;
// only the elected drainer calls return_slot. Physical owners are Send + Sync.
unsafe impl Send for NativeOwners {}
unsafe impl Sync for NativeOwners {}

impl NativeOwners {
    pub fn metal3(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Self {
        Self {
            backend: NativeBackend::Metal3(raw),
            physical,
        }
    }
    pub fn metal4(
        slot: metal4::Slot,
        submission: Arc<metal4::Submission>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Self {
        Self {
            backend: NativeBackend::Metal4 {
                slot: Mutex::new(Some(slot)),
                submission,
            },
            physical,
        }
    }
    pub fn publish_native_feedback(&self, feedback: &metal4::Feedback) {
        if let NativeBackend::Metal4 { submission, .. } = &self.backend {
            submission.publish_retirement_feedback(feedback);
        }
    }
    /// Invoked after logical callbacks and ordered-prefix publication. This
    /// returns native execution capacity; it does not return metadata capacity.
    pub fn return_slot(&self) {
        if let NativeBackend::Metal4 { slot, submission } = &self.backend {
            let returned = slot.lock().take().expect("one native slot return");
            if let Err(rejected) = submission.queue.return_retirement_slot(returned) {
                // Preserve rejected native ownership before the callback guard
                // catches this invariant failure and poisons the queue.
                *slot.lock() = Some(rejected);
                panic!("native slot return rejected");
            }
            submission.notify_retirement_callbacks_finished();
        }
    }
}

pub(super) fn registry(failure: Arc<FailureSignal>) -> Result<Registry, RegistryError> {
    Registry::new(failure)
}

impl Drop for NativeOwners {
    fn drop(&mut self) {
        if let NativeBackend::Metal4 { slot, submission } = &mut self.backend {
            if let Some(slot) = slot.get_mut().take() {
                // Only cancellation before native handoff reaches destruction
                // with a slot still present. Unknown/failed callbacks stay rooted.
                submission.queue.discard_retirement_slot(slot);
            }
        }
    }
}
