//! Physical ownership shared by Device, submit-capable handles and rooted work.
//!
//! Source-review draft: construction/attachment is not wired. In particular it
//! does not replace DeviceTeardown until BOTH native adapters independently root
//! every handoff and route final payload destruction through host custody.

use super::device::{ConstantsByValue, PassPlans};
use crate::kernels::{
    metal::{
        completion::SharedEvent,
        ledger::{Ledger, Reservation},
        Arena, Backings, Buffer, BufferPool, CommandQueue, Device, DispatchGate, GpuClock,
        InFlightHold, ResidencySet, SessionRegion,
    },
    Kernels,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTL4CommandQueue;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

/// The actual shared containers, built once before Device is exposed. No
/// Arc<Commands>, logical M4 Queue, ExecutorSlot, or Arc<MetalDevice> backedge.
/// The constructor must not substitute new empty arena/pass-plan containers.
///
/// Field order is intentional AFTER PhysicalContext::drop empties membership:
/// cached handles/bases return while pools/backings/hold still exist; pools drop
/// while their observer's backings exist; held unretained objects outlive those
/// returns. Native queues/device/residency and ledger are kept through releases.
pub(crate) struct PhysicalParts {
    pub constants: Arc<ConstantsByValue>,
    pub arena: Arc<RwLock<Option<Arena>>>,
    pub session_region: Arc<RwLock<Option<SessionRegion>>>,
    pub pass_plans: Arc<Mutex<Option<PassPlans>>>,
    pub arena_reservation: Arc<Mutex<Option<Reservation>>>,
    pub seed: Arc<Mutex<Buffer>>,
    pub eos_gate: Arc<DispatchGate>,
    pub buffers: BufferPool,
    pub private_buffers: BufferPool,
    pub backings: Arc<Backings>,
    pub hold: Option<Arc<InFlightHold>>,
    pub kernels: Arc<Kernels>,
    pub clock: Arc<GpuClock>,
    pub completion_event: Option<Arc<SharedEvent>>,
    pub command_queue: CommandQueue,
    pub modern_queue: Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>>,
    pub device: Device,
    pub residency_set: Arc<ResidencySet>,
    pub ledger: Arc<Ledger>,
}

pub(crate) struct PhysicalContext {
    parts: PhysicalParts,
}

impl PhysicalContext {
    /// # Safety
    /// Before any native handoff, an independently host-rooted payload must
    /// retain this same context. That root can be released only after actual
    /// native terminality, callback-borrow end, and known handoff exit. Unknown
    /// handoffs retain/quarantine the root; poisoning is NOT a release proof.
    ///
    /// Every submit-capable Commands and queue-associated wrapper/encoder keeps
    /// its anchor until it can no longer submit. Delayed native blocks own only
    /// empty metadata, never the final strong context. The last context Drop
    /// must therefore occur on a host path after no work can still use its
    /// physical allocations. The constructor does not itself prove this; both
    /// native adapters and escaped wrapper coverage are implementation gates.
    pub(crate) unsafe fn new(parts: PhysicalParts) -> Arc<Self> {
        Arc::new(Self { parts })
    }
}

impl Drop for PhysicalContext {
    fn drop(&mut self) {
        // No wait-on-Commands here: retaining it would introduce a self-cycle,
        // and a poisoned waiter does not prove native completion. The lifetime
        // protocol required by new supplies positive quiescence by root custody.
        // No production call exists until that protocol is wired and reviewed.
        self.parts.residency_set.remove_all();
        // All buffers/arenas/holds remain live above. Their subsequent field
        // drops see membership already empty; no absent allocation is removed.
    }
}
