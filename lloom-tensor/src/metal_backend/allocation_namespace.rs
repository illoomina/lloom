//! Domain-local allocation containers; queue identity and native retirement are
//! separate. Source draft: no engine loader enables admitted execution yet.
use super::device::{
    ArenaAutoRecord, ArenaAutoReport, ConstantsByValue, PassPlans, ResidencyEvictionObserver,
    PRIVATE_RESOURCE_OPTIONS, RESOURCE_OPTIONS,
};
use crate::kernels::metal::{
    heap_chunk_bytes, transient_heaps_on, Arena, ArenaLayout, ArenaRecorder, Backing, Backings,
    BufferHandle, BufferPool, Commit, Device, DeviceBuffer, GpuClock, HeapChunks,
    MTLResourceOptions, Mapped, ResidencySet, SessionRegion,
};
use crate::memory_admission::ModelDomain;
use crate::residency_class::ResidencyClass;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

/// Owns containers, never MetalDevice/Commands. The mirrored Arc fields in a
/// MetalDevice view are immutable aliases, not a mutable current-model slot.
/// Native generations must retain this anchor through real host retirement.
/// Arc/container metadata here is host overhead, not a physical-buffer charge.
pub(crate) struct AllocationNamespace {
    // Some requires explicit model ExecutionScope; None is the default device.
    pub(super) domain: Option<ModelDomain>,
    // Same process account, actual native payer even before the first model.
    // Numerical default enrollment is not machine/model admission.
    pub(super) native_domain: ModelDomain,
    pub(super) membership: Arc<crate::kernels::metal::membership::MembershipDomain>,
    pub(super) constants: Arc<ConstantsByValue>,
    pub(super) dequant_scratch: Mutex<Option<BufferHandle>>,
    pub(super) buffers: BufferPool,
    pub(super) private_buffers: BufferPool,
    pub(super) backings: Arc<Backings>,
    // Inactive cold-installed genuine descriptors; no legacy selector fallback.
    pub(super) output_catalog:
        std::sync::OnceLock<super::planned_output_catalog::PlannedOutputCatalog>,
    pub(super) arena_reservation: Arc<Mutex<Option<crate::kernels::metal::ledger::Reservation>>>,
    pub(super) arena: Arc<RwLock<Option<Arena>>>,
    pub(super) arena_recorder: Arc<Mutex<Option<Arc<Mutex<ArenaRecorder>>>>>,
    pub(super) arena_auto: Arc<Mutex<Option<ArenaAutoRecord>>>,
    pub(super) arena_auto_report: Arc<Mutex<Option<ArenaAutoReport>>>,
    pub(super) arena_rerecord: Arc<Mutex<Option<(ArenaLayout, usize)>>>,
    pub(super) arena_rerecordings: Arc<std::sync::atomic::AtomicU64>,
    pub(super) session_region: Arc<RwLock<Option<SessionRegion>>>,
    // Read/write only while holding session_region, so identity and content agree.
    pub(super) session_generation: Mutex<Arc<()>>,
    pub(super) pass_plans: Arc<Mutex<Option<PassPlans>>>,
    pub(super) base_mode: Arc<std::sync::atomic::AtomicBool>,
    pub(super) pass_state: Arc<std::sync::atomic::AtomicU8>,
    pub(super) served_pass_pool_requests: Arc<[std::sync::atomic::AtomicU64; 2]>,
    pub(super) next_pass_variant: Arc<std::sync::atomic::AtomicU32>,
}
impl AllocationNamespace {
    /// Actual default namespace only; a supplied namespace must use its open
    /// ExecutionScope AllocationContext. Borrow adds no owner/allocation/lock.
    /// Caller must select the active scope FIRST; this is not an ambient fallback.
    pub(super) fn default_operation_context(
        &self,
    ) -> crate::memory_admission::Result<crate::memory_admission::OperationContext<'_>> {
        if self.domain.is_some() {
            return Err(crate::memory_admission::AllocationError::DifferentDomain);
        }
        // SAFETY: exact immutable default namespace domain borrowed from self;
        // no issuer or owner escapes. Caller preserves active-scope selection.
        Ok(unsafe {
            crate::memory_admission::OperationContext::default_namespace(&self.native_domain)
        })
    }
    pub(super) fn new(
        device: &Device,
        clock: Arc<GpuClock>,
        residency_set: &Arc<ResidencySet>,
        ledger: &Arc<crate::kernels::metal::ledger::Ledger>,
        domain: Option<ModelDomain>,
        default_native: ModelDomain,
        pool_policy: Option<(&BufferPool, &BufferPool)>,
    ) -> Arc<Self> {
        let native_domain = domain.clone().unwrap_or(default_native);
        // Both pools decide reuse against the GPU's progress through the command
        // queue, not against CPU refcounts. `Commands` owns that clock and is
        // the only thing that advances it.
        let (buffers, private_buffers) = match pool_policy {
            Some((shared, private)) => (
                shared.empty_like(clock.clone()),
                private.empty_like(clock.clone()),
            ),
            None => (
                BufferPool::with_clock(clock.clone()),
                BufferPool::with_clock(clock.clone()),
            ),
        };
        // The backings (`profiles-and-memory-design.md` §4.1, L0). A pool
        // miss takes from placement heap chunks when `transient_heaps_on()`,
        // else from one `MTLBuffer` per miss with its residency committed at
        // the next cut; a persistent allocation takes from a resident
        // backing, wired when it is Shared.
        let membership = crate::kernels::metal::membership::MembershipDomain::new_in(
            residency_set.clone(),
            super::device::retain_evicted_allocations(),
            Some(native_domain.clone()),
        );
        let transient = |options: MTLResourceOptions| -> Arc<dyn Backing> {
            if transient_heaps_on() {
                Arc::new(
                    HeapChunks::new(
                        device,
                        Arc::clone(residency_set),
                        options,
                        heap_chunk_bytes(),
                        Some(Arc::clone(ledger)),
                    )
                    .with_membership(membership.clone())
                    .with_native_payer(native_domain.clone(), domain.is_some()),
                )
            } else {
                Arc::new(
                    DeviceBuffer::new(
                        device,
                        Arc::clone(residency_set),
                        options,
                        Commit::Deferred,
                        Some(Arc::clone(ledger)),
                    )
                    .with_membership(membership.clone())
                    .with_native_payer(native_domain.clone(), domain.is_some())
                    .with_class(ResidencyClass::Scratch),
                )
            }
        };
        let backings = Arc::new(Backings {
            transient_shared: transient(RESOURCE_OPTIONS),
            transient_private: transient(PRIVATE_RESOURCE_OPTIONS),
            resident_shared: DeviceBuffer::new(
                device,
                Arc::clone(residency_set),
                RESOURCE_OPTIONS,
                Commit::Now,
                Some(Arc::clone(ledger)),
            )
            .with_membership(membership.clone())
            .with_native_payer(native_domain.clone(), domain.is_some()),
            resident_private: DeviceBuffer::new(
                device,
                Arc::clone(residency_set),
                PRIVATE_RESOURCE_OPTIONS,
                Commit::Now,
                Some(Arc::clone(ledger)),
            )
            .with_membership(membership.clone())
            .with_native_payer(native_domain.clone(), domain.is_some()),
            mapped_shared: Mapped::new(
                device,
                Arc::clone(residency_set),
                RESOURCE_OPTIONS,
                Commit::Now,
                Some(Arc::clone(ledger)),
            )
            .with_membership(membership.clone())
            .with_native_payer(native_domain.clone(), domain.is_some()),
            classed_shared: DeviceBuffer::new(
                device,
                Arc::clone(residency_set),
                RESOURCE_OPTIONS,
                Commit::Deferred,
                Some(Arc::clone(ledger)),
            )
            .with_membership(membership.clone())
            .with_native_payer(native_domain.clone(), domain.is_some()),
            classed_private: DeviceBuffer::new(
                device,
                Arc::clone(residency_set),
                PRIVATE_RESOURCE_OPTIONS,
                Commit::Deferred,
                Some(Arc::clone(ledger)),
            )
            .with_membership(membership.clone())
            .with_native_payer(native_domain.clone(), domain.is_some()),
        });
        // The pools do not subscribe to command-buffer completion. The handler
        // advances the clock, and each pool drains what it made reusable at
        // its next `acquire`, on the host: a handler that took the pool's lock
        // could be descheduled holding it and stall the encoding thread
        // (`DESIGN.md` §6.4e).

        {
            // Installed before any allocation, so no buffer is evicted without
            // the residency set hearing about it. The pool bounds its free list
            // by destroying buffers (§6.3b), which is the second place a
            // buffer's existence ends and the one nothing was announcing
            // (`DESIGN.md` §6.3c).
            let observer = Arc::new(ResidencyEvictionObserver::new(
                Arc::clone(residency_set),
                Arc::clone(&backings),
            ));
            buffers.set_eviction_observer(observer.clone());
            private_buffers.set_eviction_observer(observer);
        }

        Arc::new(Self {
            domain,
            native_domain,
            membership,
            constants: Arc::new(ConstantsByValue::new()),
            dequant_scratch: Mutex::new(None),
            buffers,
            private_buffers,
            backings,
            output_catalog: std::sync::OnceLock::new(),
            arena_reservation: Arc::new(Mutex::new(None)),
            arena: Arc::new(RwLock::new(None)),
            arena_recorder: Arc::new(Mutex::new(None)),
            arena_auto: Arc::new(Mutex::new(None)),
            arena_auto_report: Arc::new(Mutex::new(None)),
            arena_rerecord: Arc::new(Mutex::new(None)),
            arena_rerecordings: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            session_region: Arc::new(RwLock::new(None)),
            session_generation: Mutex::new(Arc::new(())),
            pass_plans: Arc::new(Mutex::new(None)),
            base_mode: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pass_state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            served_pass_pool_requests: Arc::new(Default::default()),
            next_pass_variant: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        })
    }
}
