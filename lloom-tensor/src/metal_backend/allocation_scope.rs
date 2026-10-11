//! Explicit host execution scope. No automatic inheritance across threads/tasks.
//! Source draft: engine loaders do not enable this until native custody enrollment.
use super::{DeviceId, MetalDevice};
use crate::memory_admission::AllocationContext;
use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

thread_local! {
    static ACTIVE: RefCell<Vec<Arc<ExecutionScope>>> = const { RefCell::new(Vec::new()) };
}

/// Explicit immutable namespace for one accounted model on an existing native
/// device. Clones share that namespace; different owners do not share warm pools.
/// This is a source integration API, not yet enabled by engine family loaders.
/// Numerical accounting does not itself establish machine-admitted coverage.
#[derive(Clone)]
pub struct MetalAllocationDomain {
    view: MetalDevice,
}
impl MetalAllocationDomain {
    pub fn new(
        device: &MetalDevice,
        domain: crate::memory_admission::ModelDomain,
    ) -> crate::Result<Self> {
        // Validate nested-device/cancellation rules before constructing containers.
        // A same-device active payer supplies policy, not ownership of the new domain.
        let source = active(device)?.map_or_else(|| device.clone(), |scope| scope.view.clone());
        Ok(Self {
            view: source.allocation_view(domain),
        })
    }

    pub fn domain(&self) -> &crate::memory_admission::ModelDomain {
        self.view
            .allocation_namespace
            .domain
            .as_ref()
            .expect("accounted namespace")
    }

    /// Context must name this exact domain and remain open. The scope retains
    /// context, never PassReservation; dropping the reservation closes issuance.
    /// Retain/reuse this Arc through worker requests rather than allocate per job.
    pub fn scope(&self, context: AllocationContext) -> crate::Result<Arc<ExecutionScope>> {
        ExecutionScope::new(self.view.clone(), context)
    }
}

/// Clone the Arc explicitly into a worker, then call run there. This host object
/// retains a device; it is NOT the native namespace anchor (which has no Commands).
pub struct ExecutionScope {
    pub(super) view: MetalDevice,
    pub(super) context: AllocationContext,
}
impl ExecutionScope {
    pub(crate) fn new(view: MetalDevice, context: AllocationContext) -> crate::Result<Arc<Self>> {
        let Some(domain) = view.allocation_namespace.domain.as_ref() else {
            crate::bail!("an execution scope requires an explicit model allocation namespace");
        };
        if domain.key() != context.domain_key() {
            crate::bail!("execution namespace and allocation context have different domains");
        }
        context.check_open().map_err(crate::Error::wrap)?;
        Ok(Arc::new(Self { view, context }))
    }

    /// Lexical guard cannot escape or be dropped out of order. Nesting selects
    /// ONLY the top scope: an operation on a different device errors until that
    /// inner scope exits, rather than silently bypassing an outer admitted scope.
    pub fn run<R>(
        self: &Arc<Self>,
        f: impl FnOnce(&MetalDevice) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.context.check_open().map_err(crate::Error::wrap)?;
        ACTIVE.with(|s| s.borrow_mut().push(self.clone()));
        let _guard = ScopeGuard {
            _not_send: PhantomData,
        };
        f(&self.view)
    }
}
struct ScopeGuard {
    _not_send: PhantomData<Rc<()>>,
}
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        // Drop retained owners AFTER releasing TLS borrow: destructors may enter
        // another execution scope. No user callback runs under a RefCell borrow.
        let popped = ACTIVE.with(|s| s.borrow_mut().pop());
        drop(popped);
    }
}

pub(super) fn active(device: &MetalDevice) -> crate::Result<Option<Arc<ExecutionScope>>> {
    let scope = ACTIVE.with(|s| s.borrow().last().cloned());
    if let Some(scope) = &scope {
        check_device(device.id(), scope.view.id())?;
        // An operand may originate in ANOTHER domain on the same device. Its
        // owner remains unchanged; new outputs pay the active scope's domain.
        scope.context.check_open().map_err(crate::Error::wrap)?;
    } else if device.allocation_namespace.domain.is_some() {
        crate::bail!("admitted Metal device requires an active execution scope");
    }
    Ok(scope)
}
fn check_device(requested: DeviceId, active: DeviceId) -> crate::Result<()> {
    if requested != active {
        crate::bail!("Metal operation targets {requested:?} inside execution scope for {active:?}");
    }
    Ok(())
}

/// A fixed allocation namespace owned by a worker service. Each job supplies its
/// own context; this object never retains a load-lifetime issuance promise.
/// Not a native-submission anchor: the view owns Commands.
#[derive(Clone)]
pub struct AllocationWorker {
    view: MetalDevice,
}
/// A job retains the existing pass scope, not its PassReservation. Cancellation
/// still closes issuance. A service dropping requests under a lock must retain
/// the same AllocationWorker outside that locked state to prevent final owner
/// destruction there. This is not a native-submission anchor.
#[derive(Clone)]
pub struct AllocationRequest(Arc<ExecutionScope>);
impl AllocationWorker {
    /// Resolve the active payer once, before installing the worker service.
    pub fn capture(device: &MetalDevice) -> crate::Result<Self> {
        let view = match active(device)? {
            Some(scope) => scope.view.clone(),
            None => device.clone(),
        };
        Ok(Self { view })
    }
    /// Compatibility constructor for an explicitly legacy service. Accounted
    /// callers use capture; misuse is a programming error, not quota refusal.
    pub fn legacy(device: &MetalDevice) -> Self {
        assert!(
            device.allocation_namespace.domain.is_none() && top_is_empty(),
            "legacy worker API inside an accounted scope; use fallible capture"
        );
        Self {
            view: device.clone(),
        }
    }
    pub fn device(&self) -> &MetalDevice {
        &self.view
    }
    /// Validate before any producer queue mutation, including warm-hit takes.
    pub fn capture_request(&self) -> crate::Result<Option<AllocationRequest>> {
        let scope = active(&self.view)?;
        let payer = scope
            .as_ref()
            .map_or(crate::memory_admission::AllocationOrigin::Legacy, |s| {
                crate::memory_admission::AllocationOrigin::Domain(s.context.domain_key())
            });
        self.view
            .declared_allocation_origin()
            .validate_reuse(&payer)
            .map_err(crate::Error::wrap)?;
        if let Some(scope) = &scope {
            if !Arc::ptr_eq(
                &scope.view.allocation_namespace,
                &self.view.allocation_namespace,
            ) {
                crate::bail!("worker request has a different allocation namespace");
            }
        }
        Ok(scope.map(AllocationRequest))
    }
    pub fn assert_legacy_request(&self) {
        assert!(
            self.view.allocation_namespace.domain.is_none() && top_is_empty(),
            "legacy worker request API inside an accounted scope; use try_request/try_submit_now"
        );
    }
    /// Install only around this job's allocation work. A cancelled request is
    /// refused, not reopened, and issued physical owners remain independent.
    pub fn run<R>(
        &self,
        request: Option<&AllocationRequest>,
        f: impl FnOnce() -> crate::Result<R>,
    ) -> crate::Result<R> {
        match request {
            Some(request) => {
                if !Arc::ptr_eq(
                    &request.0.view.allocation_namespace,
                    &self.view.allocation_namespace,
                ) {
                    crate::bail!("worker execution has a different allocation namespace");
                }
                request.0.run(|_| f())
            }
            None => {
                if self.view.allocation_namespace.domain.is_some() || !top_is_empty() {
                    crate::bail!("missing allocation context for accounted worker");
                }
                f()
            }
        }
    }
}
fn top_is_empty() -> bool {
    ACTIVE.with(|s| s.borrow().is_empty())
}
