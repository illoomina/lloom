//! Final Commands ownership, progressed by the existing host retirement service.
//! The request handle owns metadata only; the registered inbox owns Core.
use super::{ExecutorSlot, MetalKernelError};
use crate::{
    completion_wake::Wake,
    retirement_owners::{Failure, FailureSignal},
    retirement_service::{ActiveRetirement, HostRetirementService, LifecycleInbox},
};
use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Arc, OnceLock,
};

const OPEN: u8 = 0;
const REQUESTED: u8 = 1;
const SUBMITTED: u8 = 2;
const CAPTURES_STAGED: u8 = 3;
const DONE: u8 = 4;
const QUARANTINED: u8 = 5;

/// Static production seam: tests instantiate this same phase/custody adapter,
/// not a second state machine. Native close/submit implementations stay on Core.
pub(super) trait FinalCore: Send + Sync + 'static {
    type Captures: Default + Send;
    fn install_retirement_wake(&self, wake: Arc<Wake>) -> Result<(), MetalKernelError>;
    fn poll_final_submission(
        &self,
        captures: &mut Self::Captures,
    ) -> Result<bool, MetalKernelError>;
    fn poll_final_terminality(&self) -> Result<bool, MetalKernelError>;
    fn transfer_final_captures(
        &self,
        captures: &mut Self::Captures,
        empty_executor: &Arc<ExecutorSlot>,
    ) -> Result<(), MetalKernelError>;
    fn record_teardown_failure(&self);
}

struct Control {
    requested: AtomicBool,
    wake: Arc<Wake>,
}
/// LAST Commands field: its Drop runs after Commands' Core Arc is released.
/// No physical owner is retained here, and no last native owner is put on this
/// thread. The service already owns the inbox before this handle is returned.
struct TeardownRequest(Arc<Control>);

/// The production owner and fixtures use this exact declaration order. No
/// Clone/Arc of this wrapper escapes: it is the Commands-owned capability.
/// Deref exposes only private Core methods/fields, not a public forwarding API.
pub(super) struct OwnedCore<C: FinalCore> {
    core: Arc<C>,
    // LAST: request follows release of the public owner's Core custody.
    request: TeardownRequest,
}
impl<C: FinalCore> std::ops::Deref for OwnedCore<C> {
    type Target = C;
    fn deref(&self) -> &C {
        &self.core
    }
}
impl Drop for TeardownRequest {
    fn drop(&mut self) {
        self.0.requested.store(true, Ordering::Release);
        self.0.wake.notify();
    }
}

/// Independently rooted throughout native calls, capture Drop and unwind.
/// No Core -> inbox/Commands backedge. No external Arc<inbox> survives register.
struct Teardown<C: FinalCore> {
    core: Arc<C>,
    control: Arc<Control>,
    phase: AtomicU8,
    failure: FailureSignal,
    error: OnceLock<MetalKernelError>,
    empty_executor: Arc<ExecutorSlot>,
    active: ActiveRetirement<C::Captures, Arc<C>>,
}

#[derive(Default)]
pub(super) struct FinalCaptures {
    pub(super) executor: Option<Arc<ExecutorSlot>>,
    /// Every subscriber snapshot the completion hook published (boxed as
    /// published: `completion_hook::Subscribers`).
    #[allow(clippy::vec_box)]
    pub(super) subscribers: Vec<Box<Vec<super::super::completion_hook::Subscriber>>>,
    pub(super) encoder_executor: Option<Arc<ExecutorSlot>>,
    pub(super) removals: Option<super::super::residency_set::PublishedRemovals>,
}

pub(super) fn register<C: FinalCore>(core: Arc<C>) -> Result<OwnedCore<C>, MetalKernelError> {
    let service = HostRetirementService::shared()
        .map_err(|_| MetalKernelError::QueueUnavailable("host teardown service unavailable"))?;
    register_in(core, service)
}
fn register_in<C: FinalCore>(
    core: Arc<C>,
    service: &Arc<HostRetirementService>,
) -> Result<OwnedCore<C>, MetalKernelError> {
    let wake = service.wake();
    // Before publication: residency must notify this exact shared Wake after
    // unlock, including an owner whose acquisition predates installation.
    core.install_retirement_wake(wake.clone())?;
    let control = Arc::new(Control {
        requested: AtomicBool::new(false),
        wake,
    });
    let owner = Arc::new(Teardown {
        core: core.clone(),
        control: control.clone(),
        phase: AtomicU8::new(OPEN),
        failure: FailureSignal::default(),
        error: OnceLock::new(),
        empty_executor: Arc::new(ExecutorSlot::Classical),
        active: ActiveRetirement::new(),
    });
    let inbox: Arc<dyn LifecycleInbox> = owner;
    service.register_lifecycle(&inbox).map_err(|_| {
        // Construction has no external encoder, arbitrary callback or native
        // handoff. Caller still retains its pristine Core on this error.
        MetalKernelError::QueueUnavailable("host teardown registration refused")
    })?;
    drop(inbox);
    Ok(OwnedCore {
        core,
        request: TeardownRequest(control),
    })
}
impl<C: FinalCore> Teardown<C> {
    fn quarantine(&self, error: MetalKernelError) {
        // Single lifecycle worker enters once. Error may carry rejected callback
        // ownership: retain it intact, without formatting/destruction here.
        assert!(self.error.set(error).is_ok(), "one terminal teardown error");
        self.core.record_teardown_failure();
        self.phase.store(QUARANTINED, Ordering::Release);
        self.control.wake.notify();
    }
}

// SAFETY: poll uses only try admissions and returned-handoff/logical predicates;
// no replacement slot is acquired. C is staged before native/retirement edits,
// Core remains in the registered inbox across arbitrary C Drop. All externally
// changeable pending predicates notify the shared service Wake (see report).
// Native API calls themselves are synchronous; this contract excludes host
// locks/GPU waits, not an assertion about driver call latency.
unsafe impl<C: FinalCore> LifecycleInbox for Teardown<C> {
    fn poll(&self) -> bool {
        if !self.control.requested.load(Ordering::Acquire) || self.failure.get().is_some() {
            return false;
        }
        match self.phase.load(Ordering::Acquire) {
            OPEN => {
                // Stage EMPTY C before any residency owner or capture transfer.
                // On a later unwind both transferred C and A stay inspectable.
                self.active
                    .stage(C::Captures::default(), self.core.clone())
                    .unwrap_or_else(|_| unreachable!("single initial teardown stage"));
                self.phase.store(REQUESTED, Ordering::Release);
                true
            }
            REQUESTED => {
                let result = self
                    .active
                    .edit_staged(|captures| self.core.poll_final_submission(captures))
                    .expect("staged final submission");
                match result {
                    Ok(false) => false,
                    Ok(true) => {
                        self.phase.store(SUBMITTED, Ordering::Release);
                        true
                    }
                    Err(error) => {
                        self.quarantine(error);
                        true
                    }
                }
            }
            SUBMITTED => match self.core.poll_final_terminality() {
                Ok(false) => false,
                Err(error) => {
                    self.quarantine(error);
                    true
                }
                Ok(true) => {
                    self.phase.store(CAPTURES_STAGED, Ordering::Release);
                    true
                }
            },
            CAPTURES_STAGED => {
                let result = self
                    .active
                    .edit_staged(|captures| {
                        self.core
                            .transfer_final_captures(captures, &self.empty_executor)
                    })
                    .expect("staged final captures");
                if let Err(error) = result {
                    self.quarantine(error);
                    return true;
                }
                self.active
                    .drop_captures()
                    .expect("exclusive staged teardown");
                let anchor = self.active.finish().expect("capture Drop returned");
                drop(anchor); // nonfinal: self.core still roots physical custody
                self.phase.store(DONE, Ordering::Release);
                true
            }
            DONE | QUARANTINED => false,
            _ => unreachable!("lifecycle phase"),
        }
    }
    fn can_unregister(&self) -> bool {
        self.phase.load(Ordering::Acquire) == DONE && self.active.is_empty()
    }
    fn fail(&self, reason: Failure) {
        self.failure.publish(reason, || self.control.wake.notify());
    }
}

#[cfg(test)]
mod final_core_tests {
    use super::*;
    use parking_lot::Mutex;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::{atomic::AtomicUsize, Weak},
    };

    #[derive(Default)]
    struct Observation {
        external: AtomicBool,
        native_ready: [AtomicBool; 2],
        refuse_submit: AtomicBool,
        submissions: AtomicUsize,
        transfers: AtomicUsize,
        captures_dropped: AtomicUsize,
        cores_dropped: AtomicUsize,
        failed: AtomicBool,
        wake: OnceLock<Arc<Wake>>,
    }
    impl Observation {
        fn notify(&self) {
            self.wake.get().unwrap().notify();
        }
        fn complete(&self, index: usize) {
            self.native_ready[index].store(true, Ordering::Release);
            self.notify();
        }
        fn release_external(&self) {
            self.external.store(false, Ordering::Release);
            self.notify();
        }
    }
    struct Capture {
        core: Weak<FakeCore>,
        observed: Arc<Observation>,
        transferred: bool,
        panic_on_drop: bool,
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            if self.transferred {
                assert!(
                    self.core.upgrade().is_some(),
                    "Core anchor lost before capture Drop"
                );
                assert_eq!(self.observed.cores_dropped.load(Ordering::Acquire), 0);
                assert!(
                    self.observed
                        .native_ready
                        .iter()
                        .all(|ready| ready.load(Ordering::Acquire)),
                    "capture released before every fake native marker became ready"
                );
            }
            self.observed
                .captures_dropped
                .fetch_add(1, Ordering::Release);
            if self.panic_on_drop {
                panic!("final capture Drop fault");
            }
        }
    }
    #[derive(Default)]
    struct FakeCaptures(Option<Capture>);
    struct FakeCore {
        observed: Arc<Observation>,
        capture: Mutex<Option<Capture>>,
    }
    impl FakeCore {
        fn new(observed: Arc<Observation>, panic_on_drop: bool) -> Arc<Self> {
            let core = Arc::new(Self {
                observed: observed.clone(),
                capture: Mutex::new(None),
            });
            *core.capture.lock() = Some(Capture {
                core: Arc::downgrade(&core),
                observed,
                transferred: false,
                panic_on_drop,
            });
            core
        }
    }
    impl Drop for FakeCore {
        fn drop(&mut self) {
            self.observed.cores_dropped.fetch_add(1, Ordering::Release);
        }
    }
    impl FinalCore for FakeCore {
        type Captures = FakeCaptures;
        fn install_retirement_wake(&self, wake: Arc<Wake>) -> Result<(), MetalKernelError> {
            self.observed
                .wake
                .set(wake)
                .map_err(|_| MetalKernelError::QueueUnavailable("fake Wake already installed"))
        }
        fn poll_final_submission(&self, _: &mut FakeCaptures) -> Result<bool, MetalKernelError> {
            if self.observed.external.load(Ordering::Acquire) {
                return Ok(false);
            }
            if self.observed.refuse_submit.load(Ordering::Acquire) {
                return Err(MetalKernelError::QueueUnavailable(
                    "fake submission refused",
                ));
            }
            assert_eq!(
                self.observed.submissions.fetch_add(1, Ordering::AcqRel),
                0,
                "final submission retried"
            );
            Ok(true)
        }
        fn poll_final_terminality(&self) -> Result<bool, MetalKernelError> {
            Ok(self
                .observed
                .native_ready
                .iter()
                .all(|ready| ready.load(Ordering::Acquire)))
        }
        fn transfer_final_captures(
            &self,
            captures: &mut FakeCaptures,
            _: &Arc<ExecutorSlot>,
        ) -> Result<(), MetalKernelError> {
            assert!(captures.0.is_none());
            assert_eq!(self.observed.transfers.fetch_add(1, Ordering::AcqRel), 0);
            captures.0 = self.capture.lock().take();
            captures.0.as_mut().unwrap().transferred = true;
            Ok(())
        }
        fn record_teardown_failure(&self) {
            self.observed.failed.store(true, Ordering::Release);
        }
    }

    // Actual OwnedCore/register_in/Teardown/ActiveRetirement/service paths;
    // only the native Core operations are fake. This does not exercise ObjC,
    // Commands encoder locks, or scheduling at the field-destructor instant.
    #[test]
    fn adapter_retains_core_until_external_release_and_both_markers_ready() {
        let service = HostRetirementService::manual();
        let observed = Arc::new(Observation::default());
        observed.external.store(true, Ordering::Release);
        let core = FakeCore::new(observed.clone(), false);
        let owner = register_in(core.clone(), &service).unwrap();
        drop(core);
        assert!(!service.manual_pass());
        let seen = service.wake().snapshot();
        drop(owner);
        assert_ne!(
            seen,
            service.wake().snapshot(),
            "final owner did not publish request Wake"
        );
        assert!(service.manual_pass()); // stage independently rooted C + A
        assert!(!service.manual_pass());
        assert_eq!(observed.submissions.load(Ordering::Acquire), 0);
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 0);
        observed.release_external();
        assert!(service.manual_pass());
        assert_eq!(observed.submissions.load(Ordering::Acquire), 1);
        assert!(!service.manual_pass());
        observed.complete(1);
        assert!(!service.manual_pass());
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 0);
        observed.complete(0);
        assert!(service.manual_pass());
        assert!(service.manual_pass());
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 1);
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 1);
        assert!(!service.manual_pass());
    }

    #[test]
    fn rejected_registration_preserves_the_callers_original_core() {
        let service = HostRetirementService::manual();
        service.close_admission();
        let observed = Arc::new(Observation::default());
        let core = FakeCore::new(observed.clone(), false);
        assert!(matches!(
            register_in(core.clone(), &service),
            Err(MetalKernelError::QueueUnavailable(
                "host teardown registration refused"
            ))
        ));
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 0);
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 0);
        assert!(!service.manual_pass());
        drop(core);
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 1);
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 1);
    }

    #[test]
    fn submission_refusal_quarantines_core_without_capture_release_or_retry() {
        let service = HostRetirementService::manual();
        let observed = Arc::new(Observation::default());
        observed.refuse_submit.store(true, Ordering::Release);
        let owner = register_in(FakeCore::new(observed.clone(), false), &service).unwrap();
        drop(owner);
        assert!(service.manual_pass());
        assert!(service.manual_pass());
        assert!(observed.failed.load(Ordering::Acquire));
        for _ in 0..3 {
            assert!(!service.manual_pass());
        }
        assert_eq!(observed.submissions.load(Ordering::Acquire), 0);
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 0);
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 0);
        // End only this fake, non-native registry's custody. Production's static
        // service retains uncertain native owners instead of freeing on failure.
        drop(service);
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 1);
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 1);
    }

    #[test]
    fn capture_unwind_keeps_independent_core_custody() {
        let service = HostRetirementService::manual();
        let observed = Arc::new(Observation::default());
        for ready in &observed.native_ready {
            ready.store(true, Ordering::Release);
        }
        let core = FakeCore::new(observed.clone(), true);
        let weak = Arc::downgrade(&core);
        let owner = register_in(core, &service).unwrap();
        drop(owner);
        assert!(service.manual_pass());
        assert!(service.manual_pass());
        assert!(service.manual_pass());
        let payload = catch_unwind(AssertUnwindSafe(|| service.manual_pass())).unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"final capture Drop fault")
        );
        assert!(weak.upgrade().is_some(), "unwind lost service-owned Core");
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 0);
        assert_eq!(observed.captures_dropped.load(Ordering::Acquire), 1);
        // Manual fixtures do not spawn worker/catch loop. Publish its explicit
        // failure disposition before another pass; no retry of arbitrary Drop.
        service.close_admission();
        assert!(!service.manual_pass());
        assert_eq!(observed.transfers.load(Ordering::Acquire), 1);
        drop(service);
        assert!(weak.upgrade().is_none());
        assert_eq!(observed.cores_dropped.load(Ordering::Acquire), 1);
    }
}
