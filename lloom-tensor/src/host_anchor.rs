//! Pre-registered host destruction custody, independent of callback metadata.
//!
//! A permit owns only a counted metadata pin. Its final Drop publishes a
//! host-only cancellation; it never destroys C or A, allocates, or locks a producer.
//! This record never enters a native handoff. A native adapter must keep its
//! own permit in the real native record until that record's terminal/capture
//! proof allows cleanup. Losing native certainty therefore retains the permit.
//!
//! Retirement actions/arbitrary destructor captures belong in C. Physical
//! raw/charge owners belong in A and remain rooted if C destruction panics.
//! A has the same trusted non-panicking engine-anchor contract as the existing
//! retirement service; this helper does not broaden that contract.
use crate::{
    completion_wake::Wake,
    retirement_mailbox::{CleanupPolicy, MailboxError, PreparedEditError},
    retirement_owners::{Failure, FailureSignal},
    retirement_registry::{OwnerRegistry, PinnedRegistration, RegistryError},
};
use std::sync::Arc;

pub(crate) struct HostAnchorRegistry<C: Send + Sync + 'static, A: Send + Sync + 'static> {
    registry: OwnerRegistry<C, A, ()>,
}
#[derive(Debug)]
pub(crate) enum AnchorError {
    Registry(RegistryError),
    Pin(MailboxError),
    PermitExhausted,
}
pub(crate) struct AnchorRejected<C, A> {
    pub error: AnchorError,
    // None means already independently rooted; never synthesize returned A.
    pub owner: Option<(C, A)>,
}
pub(crate) struct HostAnchorPermit<C: Send + Sync + 'static, A: Send + Sync + 'static> {
    pin: PinnedRegistration<C, A, ()>,
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static> HostAnchorRegistry<C, A> {
    pub(crate) fn new(
        failure: Arc<FailureSignal>,
        waiters: Arc<Wake>,
    ) -> Result<Self, RegistryError> {
        Ok(Self {
            registry: OwnerRegistry::new_with_waiters(
                failure,
                waiters,
                CleanupPolicy::TerminalFeedback,
            )?,
        })
    }
    #[cfg(test)]
    pub(crate) fn manual(
        failure: Arc<FailureSignal>,
        service: Arc<crate::retirement_service::HostRetirementService>,
        waiters: Arc<Wake>,
    ) -> Self {
        Self {
            registry: OwnerRegistry::manual_with_waiters(
                failure,
                service,
                waiters,
                CleanupPolicy::TerminalFeedback,
            ),
        }
    }
    pub(crate) fn reserve(
        &self,
        captures: C,
        owner: A,
        admit_growth: impl FnOnce(usize) -> Result<(), RegistryError>,
    ) -> Result<HostAnchorPermit<C, A>, AnchorRejected<C, A>> {
        let record = self
            .registry
            .prepare(captures, owner, admit_growth)
            .map_err(|rejected| AnchorRejected {
                error: AnchorError::Registry(rejected.error),
                owner: Some((rejected.captures, rejected.anchors)),
            })?;
        Self::permit(record)
    }
    /// Retained logical slab-layout admission; the same pin/wrapper protocol as
    /// legacy reserve. The registry holds the ticket's charge until host final
    /// metadata destruction, not only until payload cleanup credit returns.
    pub(crate) fn reserve_admitted(
        &self,
        payer: crate::memory_admission::DomainKey,
        captures: C,
        owner: A,
        admit_growth: impl FnOnce(
            usize,
        )
            -> Result<crate::memory_admission::AllocationTicket, RegistryError>,
    ) -> Result<HostAnchorPermit<C, A>, AnchorRejected<C, A>> {
        let record = self
            .registry
            .prepare_admitted(payer, captures, owner, admit_growth)
            .map_err(|rejected| AnchorRejected {
                error: AnchorError::Registry(rejected.error),
                owner: Some((rejected.captures, rejected.anchors)),
            })?;
        Self::permit(record)
    }
    fn permit(
        record: crate::retirement_registry::Registration<C, A, ()>,
    ) -> Result<HostAnchorPermit<C, A>, AnchorRejected<C, A>> {
        let pin = match record.pin() {
            Ok(pin) => pin,
            Err(error) => {
                if record.cancel_before_handoff().is_err() {
                    record.report_failure(Failure::AdapterFailed);
                }
                return Err(AnchorRejected {
                    error: AnchorError::Pin(error),
                    owner: None,
                });
            }
        };
        if pin.add_wrapper_handle().is_err() {
            if record.cancel_before_handoff().is_err() {
                record.report_failure(Failure::AdapterFailed);
            }
            return Err(AnchorRejected {
                error: AnchorError::PermitExhausted,
                owner: None,
            });
        }
        Ok(HostAnchorPermit { pin })
    }
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static> HostAnchorPermit<C, A> {
    pub(crate) fn try_clone(&self) -> Result<Self, AnchorError> {
        // Existing self prevents zero-count retirement. Pin first excludes
        // generation reuse if count acquisition refuses.
        let pin = self.pin.try_clone().map_err(AnchorError::Pin)?;
        pin.add_wrapper_handle()
            .map_err(|_| AnchorError::PermitExhausted)?;
        Ok(Self { pin })
    }
    /// Trusted host-only edit, same input-custody contract as edit_prepared:
    /// allocate/check before taking input; no panic after take before insertion.
    /// A panic quarantines the registered anchor rather than destroying it.
    pub(crate) fn edit<V, R>(
        &self,
        input: V,
        edit: impl FnOnce(&mut C, &mut A, &mut Option<V>) -> R,
    ) -> Result<(R, Option<V>), PreparedEditError<V>> {
        self.pin.record().edit_prepared(input, edit)
    }
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static> Drop for HostAnchorPermit<C, A> {
    fn drop(&mut self) {
        if self.pin.release_wrapper_handle() {
            // No arbitrary A is on this stack. If a broken phase prevents
            // cancellation, keep service custody and wake/refuse; never free.
            if self.pin.record().cancel_before_handoff().is_err() {
                self.pin.record().report_failure(Failure::AdapterFailed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retirement_service::HostRetirementService;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted(Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Action {
        lock: Arc<Mutex<()>>,
        drops: Arc<AtomicUsize>,
        panic: bool,
    }
    impl Drop for Action {
        fn drop(&mut self) {
            assert!(
                self.lock.try_lock().is_some(),
                "action destroyed under recording lock"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic, "injected action panic");
        }
    }
    fn drain(service: &HostRetirementService) {
        for _ in 0..96 {
            if !service.manual_pass() {
                return;
            }
        }
        panic!("bounded fake cleanup failed to drain");
    }
    #[test]
    fn last_permit_only_publishes_until_host_cleanup() {
        let service = HostRetirementService::manual();
        let registry = HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            Arc::new(Wake::default()),
        );
        let lock = Arc::new(Mutex::new(()));
        let actions = Arc::new(AtomicUsize::new(0));
        let owners = Arc::new(AtomicUsize::new(0));
        let permit = registry
            .reserve(
                Action {
                    lock: lock.clone(),
                    drops: actions.clone(),
                    panic: false,
                },
                Counted(owners.clone()),
                |_| Ok(()),
            )
            .unwrap_or_else(|_| panic!("fake anchor reserve"));
        let cloned = permit.try_clone().unwrap();
        drop(permit);
        assert!(!service.manual_pass(), "nonfinal permit scheduled cleanup");
        let guard = lock.lock();
        drop(cloned);
        assert_eq!(actions.load(Ordering::SeqCst), 0);
        assert_eq!(owners.load(Ordering::SeqCst), 0);
        drop(guard);
        drain(&service);
        assert_eq!(actions.load(Ordering::SeqCst), 1);
        assert_eq!(owners.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn native_capture_and_wrapper_permits_do_not_form_cleanup_cycle() {
        let service = HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let wake = Arc::new(Wake::default());
        let anchors = HostAnchorRegistry::manual(failure.clone(), service.clone(), wake.clone());
        let owners = Arc::new(AtomicUsize::new(0));
        let wrapper = anchors
            .reserve((), Counted(owners.clone()), |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake anchor reserve"));
        let captured_wrapper = wrapper.try_clone().unwrap();
        let native_permit = wrapper.try_clone().unwrap();
        let native = OwnerRegistry::<_, _, ()>::manual_with_waiters(
            failure,
            service.clone(),
            wake,
            CleanupPolicy::ReturnedHandoff,
        );
        let record = native
            .prepare(captured_wrapper, native_permit, |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake native record"));
        drop(wrapper);
        record.begin_handoff().unwrap();
        record.handoff_returned().unwrap();
        record
            .publish_feedback(())
            .unwrap_or_else(|_| panic!("fake feedback"));
        record.run_callbacks(|_, _, _| {}, || {}).unwrap();
        drain(&service);
        assert_eq!(
            owners.load(Ordering::SeqCst),
            1,
            "capture-held wrapper prevented independent capture cleanup"
        );
    }
    #[test]
    fn unknown_native_handoff_keeps_anchor_after_external_roots_drop() {
        let service = HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let wake = Arc::new(Wake::default());
        let anchors = HostAnchorRegistry::manual(failure.clone(), service.clone(), wake.clone());
        let owners = Arc::new(AtomicUsize::new(0));
        let wrapper = anchors
            .reserve((), Counted(owners.clone()), |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake anchor reserve"));
        let native = OwnerRegistry::<(), _, ()>::manual_with_waiters(
            failure,
            service.clone(),
            wake,
            CleanupPolicy::ReturnedHandoff,
        );
        let record = native
            .prepare((), wrapper.try_clone().unwrap(), |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake native record"));
        record.begin_handoff().unwrap();
        record.handoff_unknown(|| {}).unwrap();
        record
            .publish_feedback(())
            .unwrap_or_else(|_| panic!("fake feedback"));
        record.run_callbacks(|_, _, _| {}, || {}).unwrap();
        drop(wrapper);
        drop(record);
        drop(native);
        drop(anchors);
        drain(&service);
        assert_eq!(
            owners.load(Ordering::SeqCst),
            0,
            "unknown handoff lost independently rooted physical anchor"
        );
        // Fake-only fixture teardown, not permission to release native quarantine.
        drop(service);
        assert_eq!(owners.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn action_panic_retains_physical_anchor_and_growth_refusal_returns_owner() {
        let service = HostRetirementService::manual();
        let registry = HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            Arc::new(Wake::default()),
        );
        let owners = Arc::new(AtomicUsize::new(0));
        let rejected = match registry.reserve((), Counted(owners.clone()), |_| {
            Err(RegistryError::GrowthRefused)
        }) {
            Ok(_) => panic!("growth refusal ignored"),
            Err(rejected) => rejected,
        };
        assert!(matches!(
            rejected.error,
            AnchorError::Registry(RegistryError::GrowthRefused)
        ));
        assert_eq!(owners.load(Ordering::SeqCst), 0);
        drop(
            rejected
                .owner
                .expect("pre-registration refusal lost ownership"),
        );
        assert_eq!(owners.load(Ordering::SeqCst), 1);
        drop(registry);
        let actions = Arc::new(AtomicUsize::new(0));
        let registry = HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            Arc::new(Wake::default()),
        );
        let permit = registry
            .reserve(
                Action {
                    lock: Arc::new(Mutex::new(())),
                    drops: actions.clone(),
                    panic: true,
                },
                Counted(owners.clone()),
                |_| Ok(()),
            )
            .unwrap_or_else(|_| panic!("fake action reserve"));
        drop(permit);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| service.manual_pass()))
                .is_err()
        );
        assert_eq!(actions.load(Ordering::SeqCst), 1);
        assert_eq!(
            owners.load(Ordering::SeqCst),
            1,
            "action unwind released physical anchor"
        );
        // The real service catches this panic, closes admission and roots itself
        // for process lifetime. This manual fake service only checks active custody.
        drop(registry);
        drop(service);
        assert_eq!(owners.load(Ordering::SeqCst), 2);
    }
}
