//! Inactive actual descriptor catalog for default/supplied output factories.
//! A lookup owns no second physical inventory: every node owns the existing
//! PreparedPoolSlot, whose sole PhysicalReady authorizes actual reuse.
//! No legacy FreeNode, native pointer, view or BufferHandle can enter this list.
use super::operation_payload::{AffinePreparation, OperationPayload, OperationPreparationError};
use crate::kernels::kernels::affine::PreparedAffinePipeline;
use crate::{
    allocation_header_root::{HeaderPool, HeaderStrong, RootError},
    kernels::metal::{BufferHandle, Commands, PreparedPoolSlot},
    memory_admission::{
        AllocationContext, AllocationError, AllocationTicket, DomainKey, PhysicalCharge,
    },
};
use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicPtr, Ordering},
};

// Size classes index a usize's exact bit width, not an outstanding-count cap.
// A class may contain arbitrary admitted cold growth, including non-power-of-two
// native extents. Warm lookup still checks actual immutable capacity.
const CLASSES: usize = usize::BITS as usize;
const PROFILES: usize = 2;
#[derive(Clone, Copy)]
pub(crate) enum OutputProfile {
    Shared,
    Private,
}
impl OutputProfile {
    fn index(self) -> usize {
        match self {
            Self::Shared => 0,
            Self::Private => 1,
        }
    }
}
struct Node {
    // Once Release-published this link never changes while any catalog holder
    // exists. No warm removal, address recycling or ABA is possible.
    next: AtomicPtr<Node>,
    slot: PreparedPoolSlot,
    writer: AtomicBool,
    payload: UnsafeCell<OperationPayload>,
    charge: Option<PhysicalCharge>,
}
// SAFETY: payload access requires unique writer ownership. Mutable preparation
// ends before consuming that owner into a readonly installed plan. The writer
// stays claimed until native terminal AND future-emission closure; its Release
// Drop synchronizes the next Acquire claim. Readonly shader borrows cannot
// overlap mutable preparation. Every direct/operation/plan capability roots this
// exact immutable node; whole-root service cleanup cannot run while any exists.
unsafe impl Sync for Node {}

struct CatalogState {
    domain: DomainKey,
    heads: [[AtomicPtr<Node>; CLASSES]; PROFILES],
}
impl Drop for CatalogState {
    fn drop(&mut self) {
        // Only actual host HeaderRoot C cleanup reaches here, after the last
        // genuine catalog Strong. Every traversal/append holds that Strong;
        // no pointer/root Arc escapes. Thus whole-root exclusive ownership, not
        // an atomic 'empty' guess, authorizes reclaiming the immutable lists.
        for profile in &mut self.heads {
            for head in profile {
                let mut pointer = *head.get_mut();
                while !pointer.is_null() {
                    // SAFETY: each accepted Node came from exactly one Global
                    // Box, belongs to one head, and has never been removed.
                    let mut node = unsafe { Box::from_raw(pointer) };
                    pointer = *node.next.get_mut();
                    let charge = node.charge.take();
                    // Slot Drop publishes metadata atoms; physical/rich roots
                    // retain native/range/member/account custody independently.
                    // No Slot/root/native/charge finalizer runs under registry.
                    drop(node); // actual Node storage is freed before refund
                    drop(charge);
                }
                *head.get_mut() = std::ptr::null_mut();
            }
        }
    }
}
pub(crate) struct PlannedOutputCatalog {
    // Pool is outside C; opaque final service rooting, never a self-backedge.
    pool: HeaderPool<CatalogState, (), 1>,
    state: HeaderStrong<CatalogState, (), 1>,
}
#[derive(Debug)]
pub(crate) enum CatalogError {
    Admission(AllocationError),
    Root(RootError),
    Checkout(crate::kernels::metal::StagingError),
    NeedsColdGrowth,
    Empty,
    Creator(crate::Error),
    WriterBusy,
    Operation(OperationPreparationError),
}
impl From<AllocationError> for CatalogError {
    fn from(error: AllocationError) -> Self {
        Self::Admission(error)
    }
}
fn class(bytes: usize) -> Result<usize, CatalogError> {
    if bytes == 0 {
        return Err(CatalogError::Empty);
    }
    Ok((usize::BITS - 1 - bytes.leading_zeros()) as usize)
}
impl PlannedOutputCatalog {
    pub(crate) fn requested_root_bytes() -> Result<usize, RootError> {
        HeaderPool::<CatalogState, (), 1>::requested_owned_bytes()
    }
    pub(crate) fn requested_node_bytes() -> Result<usize, AllocationError> {
        std::alloc::Layout::new::<Node>()
            .size()
            .checked_add(AllocationTicket::requested_charge_control_bytes()?)
            .ok_or(AllocationError::Overflow)
    }
    /// # Safety
    /// Actual default/supplied namespace installs this only alongside complete
    /// rich native/IO/raw/foreign/CPU/emission enrollment, using the SAME actual
    /// payer. It does not itself turn any existing Tensor input into rich use.
    /// No backedge from node/physical/header C/A to namespace/Commands/catalog.
    /// All shared/bootstrap/domain/PassState/native opaque terms remain explicit
    /// factory terms, not hidden by these exact requested Layout quotes.
    pub(crate) unsafe fn cold_in(context: &AllocationContext) -> Result<Self, CatalogError> {
        // Compile-time requirement: actual descriptor owners support host
        // service cleanup and simultaneous read-only warm checkout.
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Node>();
        let pool = unsafe {
            HeaderPool::<CatalogState, (), 1>::cold_owned_when_in(context, |_, _| true, None)
        }
        .map_err(CatalogError::Root)?;
        let state = pool
            .insert(
                CatalogState {
                    domain: context.domain_key(),
                    heads: std::array::from_fn(|_| {
                        std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut()))
                    }),
                },
                (),
            )
            .map_err(|e| CatalogError::Root(e.error))?;
        // Pool capability stays outside C. Producer final Drop only releases
        // metadata counts; the independently enrolled host service owns cleanup.
        Ok(Self { pool, state })
    }
    /// Admitted cold growth only, BEFORE any Commands/encoder/native EDIT.
    /// Node entitlement precedes the actual backing/capsule creator callback.
    /// No lock is held during callback, destruction, admission or Box birth.
    /// # Safety
    /// create returns a genuine whole PreparedPoolSlot for exactly profile and
    /// context domain/device/creator clock. Never reconstruct one from a legacy
    /// Buffer/Handle/raw pointer. Constructor's closed future-use contract holds.
    pub(crate) unsafe fn append_cold_with(
        &self,
        context: &AllocationContext,
        profile: OutputProfile,
        create: impl FnOnce() -> Result<(PreparedPoolSlot, OperationPayload), CatalogError>,
    ) -> Result<DirectOutputPosition, CatalogError> {
        if context.domain_key() != self.state.captures().domain {
            return Err(CatalogError::Admission(AllocationError::DifferentDomain));
        }
        context.check_open().map_err(CatalogError::Admission)?;
        self.pool.check_failed().map_err(CatalogError::Root)?;
        // Bind the returned plan position to THIS actual constructor node.
        // Size/profile-only selection cannot identify a shader/schema payload.
        // Count refusal precedes all allocation and native creator callbacks.
        let position_root = self
            .state
            .try_clone()
            .map_err(|e| CatalogError::Root(RootError::Header(e)))?;
        let mut ticket = context
            .take(Self::requested_node_bytes()?)
            .map_err(CatalogError::Admission)?;
        let (slot, payload) = create()?; // caller's actual device backing/capsule factory
        let index = class(slot.capacity())?;
        context.check_open().map_err(CatalogError::Admission)?;
        let mut node = match crate::cold_box::try_new(Node {
            next: AtomicPtr::new(std::ptr::null_mut()),
            slot,
            writer: AtomicBool::new(false),
            payload: UnsafeCell::new(payload),
            charge: None,
        }) {
            Ok(node) => node,
            Err(original) => {
                drop(original); // still-rooted slot publishes atoms only
                return Err(CatalogError::Admission(AllocationError::HostAllocation));
            }
        };
        let charge = match ticket.try_commit_with_control(std::alloc::Layout::new::<Node>().size())
        {
            Ok(charge) => charge,
            Err(error) => {
                drop(node);
                return Err(CatalogError::Admission(error));
            }
        };
        node.charge = Some(charge);
        // Last cancellation/failure checks precede publication. On refusal take
        // charge out, free actual Node Box, then refund; physical slot roots
        // remain independently service-owned, never counter-reset away.
        if let Err(error) = context.check_open() {
            let charge = node.charge.take();
            drop(node);
            drop(charge);
            return Err(CatalogError::Admission(error));
        }
        if let Err(error) = self.pool.check_failed() {
            let charge = node.charge.take();
            drop(node);
            drop(charge);
            return Err(CatalogError::Root(error));
        }
        let head = &self.state.captures().heads[profile.index()][index];
        let mut seen = head.load(Ordering::Acquire);
        let pointer = (&mut *node) as *mut Node;
        loop {
            node.next.store(seen, Ordering::Relaxed); // still private exclusive Box
            match head.compare_exchange_weak(seen, pointer, Ordering::Release, Ordering::Acquire) {
                Ok(_) => break,
                Err(next) => seen = next,
            }
        }
        // Publication transfers unique Box ownership; no arbitrary/fallible
        // operation or Drop may occur between successful CAS and relinquishing.
        let _ = Box::into_raw(node);
        Ok(DirectOutputPosition {
            root: position_root,
            node: pointer,
        })
    }
    /// COLD plan-position selection only, outside Commands/native EDIT. The
    /// ordinary operation never repeats this size-class/list walk. Growth or
    /// alternate position selection is recoverable admitted cold work. This
    /// size/profile lookup does not select an operation schema; typed factories
    /// retain append_cold_with's exact freshly created operation position.
    pub(crate) fn select_cold(
        &self,
        bytes: usize,
        profile: OutputProfile,
    ) -> Result<DirectOutputPosition, CatalogError> {
        self.pool.check_failed().map_err(CatalogError::Root)?;
        for head in &self.state.captures().heads[profile.index()][class(bytes)?..] {
            let mut pointer = head.load(Ordering::Acquire);
            while !pointer.is_null() {
                // SAFETY: genuine catalog Strong excludes C cleanup; no warm
                // unlink/address recycling occurs. Link is immutable post-CAS.
                let node = unsafe { &*pointer };
                if node.slot.capacity() >= bytes {
                    let root = self
                        .state
                        .try_clone()
                        .map_err(|error| CatalogError::Root(RootError::Header(error)))?;
                    return Ok(DirectOutputPosition {
                        root,
                        node: pointer,
                    });
                }
                pointer = node.next.load(Ordering::Relaxed);
            }
        }
        Err(CatalogError::NeedsColdGrowth)
    }
}
/// Authentic direct position: inseparable opaque catalog root and exact stable
/// node. No public/raw constructor/Node pointer/native authority can escape.
/// Not Clone: caller's actual operation plan owns this position; concurrent
/// operation uses must additionally own its exclusive parameter writer guard.
pub(crate) struct DirectOutputPosition {
    root: HeaderStrong<CatalogState, (), 1>,
    node: *mut Node,
}
// SAFETY: the opaque Strong prevents node reclamation; links/slot are immutable.
// Mutable payload access exists only through the unique Acquire-CAS writer,
// retained until installed native terminal AND future-emission closure. Neither
// cold selection nor ordinary readonly output checkout touches that payload.
unsafe impl Send for DirectOutputPosition {}
unsafe impl Sync for DirectOutputPosition {}
impl DirectOutputPosition {
    pub(crate) fn capacity(&self) -> usize {
        // SAFETY: genuine root holder prevents node storage cleanup/recycling.
        unsafe { (&*self.node).slot.capacity() }
    }
    /// O(1) exact-position attempt BEFORE command_encoder/native EDIT.
    /// Busy/Full retains original generation/cutoff/credit; caller obtains
    /// admitted cold alternate/growth, not a warm scan or new Box/Arc.
    /// # Safety
    /// Same complete role/native/IO/foreign/CPU/emission contract as real slot;
    /// output uninitialized, caller must establish write-before-read. Original
    /// input and params/EOS must carry their own genuine producer authority.
    pub(crate) unsafe fn try_checkout(
        &self,
        context: &AllocationContext,
        commands: &Commands,
    ) -> Result<BufferHandle, CatalogError> {
        unsafe { self.try_checkout_under(&context.operation_context(), commands) }
    }
    /// # Safety
    /// Same originating cold-price/closed-use/initialized-range obligations as
    /// the AllocationContext compatibility wrapper; warm permission alone is
    /// neither a native-use receipt nor a write-before-read certificate.
    pub(crate) unsafe fn try_checkout_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &Commands,
    ) -> Result<BufferHandle, CatalogError> {
        if context.domain_key() != self.root.captures().domain {
            return Err(CatalogError::Admission(AllocationError::DifferentDomain));
        }
        self.root.check_failed().map_err(CatalogError::Root)?;
        // SAFETY: exact inseparable root/node binding; immutable node lives
        // until this root holder ends. Actual slot repeats context/failure checks
        // around exclusive PhysicalReady reservation and publish-last.
        unsafe { (&*self.node).slot.try_checkout_under(context, commands) }
            .map_err(CatalogError::Checkout)
    }
}

/// Owned pre-effect writer, not a lexical method guard. No root/Node pointer,
/// context/Commands backedge, general payload ref or independent writer token
/// escapes. Forgetting this owner retains root and writer, never permits reuse.
pub(crate) struct CatalogOperationLease {
    root: HeaderStrong<CatalogState, (), 1>,
    node: *mut Node,
}
// SAFETY: unique writer ownership and inseparable counted root survive transfer.
// Payload's cold objects must be Send; mutable access requires &mut this owner.
unsafe impl Send for CatalogOperationLease {}

/// Readonly installed plan owns the WHOLE writer lease, not a second root clone.
/// Actual generation custody keeps it until SAME immutable native terminal AND
/// future-emission closure. Unknown handoff remains initialized and false-ready.
/// Dropping uninstalled/pre-effect cancellation is distinct from native cleanup.
pub(crate) struct CatalogPlanLease {
    operation: CatalogOperationLease,
}
// SAFETY: ownership still excludes every mutable payload accessor. The only
// shared projection is immutable PreparedAffinePipeline, asserted Sync below.
// Its borrow cannot outlive this genuine installed writer/root owner.
unsafe impl Sync for CatalogPlanLease {}
impl DirectOutputPosition {
    pub(crate) fn try_operation(
        &self,
        context: &AllocationContext,
        commands: &Commands,
    ) -> Result<CatalogOperationLease, CatalogError> {
        self.try_operation_under(&context.operation_context(), commands)
    }
    pub(crate) fn try_operation_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &Commands,
    ) -> Result<CatalogOperationLease, CatalogError> {
        if context.domain_key() != self.root.captures().domain {
            return Err(CatalogError::Admission(AllocationError::DifferentDomain));
        }
        context.check_open().map_err(CatalogError::Admission)?;
        self.root.check_failed().map_err(CatalogError::Root)?;
        commands
            .check_failed_atomic()
            .map_err(|e| CatalogError::Checkout(crate::kernels::metal::StagingError::Native(e)))?;
        // Count refusal precedes the writer claim; checked clone leaves original
        // root/count unchanged. The rejected local clone releases atoms only.
        let root = self
            .root
            .try_clone()
            .map_err(|e| CatalogError::Root(RootError::Header(e)))?;
        // SAFETY: original position plus clone protect this exact Node.
        let node = unsafe { &*self.node };
        node.writer
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| CatalogError::WriterBusy)?;
        let operation = CatalogOperationLease {
            root,
            node: self.node,
        };
        // The concrete owner exists before rechecks. Refusal drops ONLY this
        // writer+metadata count; no payload mutation or global counter rollback.
        context.check_open().map_err(CatalogError::Admission)?;
        operation.root.check_failed().map_err(CatalogError::Root)?;
        commands
            .check_failed_atomic()
            .map_err(|e| CatalogError::Checkout(crate::kernels::metal::StagingError::Native(e)))?;
        Ok(operation)
    }
}
impl CatalogOperationLease {
    pub(crate) fn affine_preparation(
        &mut self,
        rank: Option<usize>,
    ) -> Result<AffinePreparation<'_>, CatalogError> {
        // SAFETY: exact rooted node, unique writer. Projection can mutate only
        // trusted params/shadow state, not enum/pipeline/slot ownership. Returned
        // &mut borrow prevents consuming into_plan while it is still in use.
        match unsafe { &mut *(*self.node).payload.get() } {
            OperationPayload::Affine(operation) => {
                operation.preparation(rank).map_err(CatalogError::Operation)
            }
        }
    }
    /// Borrowed context is validated again, never stored in catalog C. Outputs
    /// are uninitialized; real lowering must establish write-before-read.
    /// # Safety
    /// Exact closed creator/native/IO/foreign/emission contract of the real slot
    /// holds; all actual operation users acquire corresponding native leases.
    pub(crate) unsafe fn checkout_output(
        &self,
        context: &AllocationContext,
        commands: &Commands,
    ) -> Result<BufferHandle, CatalogError> {
        unsafe { self.checkout_output_under(&context.operation_context(), commands) }
    }
    /// # Safety
    /// Same originating cold-price/closed-use/initialized-range obligations as
    /// the AllocationContext compatibility wrapper; warm permission alone is
    /// neither a native-use receipt nor a write-before-read certificate.
    pub(crate) unsafe fn checkout_output_under(
        &self,
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &Commands,
    ) -> Result<BufferHandle, CatalogError> {
        if context.domain_key() != self.root.captures().domain {
            return Err(CatalogError::Admission(AllocationError::DifferentDomain));
        }
        self.root.check_failed().map_err(CatalogError::Root)?;
        // SAFETY: unique opaque writer/root still spans exact Node/slot borrow.
        unsafe { (&*self.node).slot.try_checkout_under(context, commands) }
            .map_err(CatalogError::Checkout)
    }
    /// Consumes ALL mutable preparation authority. No clone/allocation/release.
    /// Actual original EDIT reserves the entire buffer+Plan batch BEFORE effects
    /// and installs this owner into SAME generation page terminal custody.
    pub(crate) fn into_plan(self) -> CatalogPlanLease {
        CatalogPlanLease { operation: self }
    }
}
impl CatalogPlanLease {
    pub(crate) fn affine_shader(&self) -> &PreparedAffinePipeline {
        fn require_sync<T: Sync>() {}
        require_sync::<PreparedAffinePipeline>();
        // SAFETY: owned exact Node writer remains claimed, no mutable projection
        // is available on installed type; its Strong protects payload storage.
        match unsafe { &*(*self.operation.node).payload.get() } {
            OperationPayload::Affine(operation) => operation.shader(),
        }
    }
}
impl Drop for CatalogOperationLease {
    fn drop(&mut self) {
        // Known pre-effects cancellation or actual page host cleanup only.
        // SAFETY: this unique writer still owns the genuine Strong; its field
        // drops AFTER writer Release, so Node cannot be freed during this store.
        unsafe { (&*self.node).writer.store(false, Ordering::Release) };
    }
}
