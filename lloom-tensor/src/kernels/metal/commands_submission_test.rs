//! Test-only, per-owner fault points. Never compiled into the shipped library.
//! These controls carry no owner authority and do not replace native admission.
use super::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLSharedEvent;
use std::sync::{
    atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
    Condvar, Mutex,
};

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Fault {
    Registration = 1,
    BeforeEntry = 2,
    PartialEntry = 3,
    HoldReturn = 4,
    UnknownReturn = 5,
}

#[derive(Default)]
pub(crate) struct Control {
    fault: AtomicU8,
    pub native_calls: AtomicUsize,
    pub entry_checks: AtomicUsize,
    required_residency: AtomicU64,
    observed_residency: AtomicU64,
    observer: Mutex<Option<Retained<ProtocolObject<dyn MTLSharedEvent>>>>,
    pause: Mutex<(bool, bool)>, // (at return boundary, released)
    changed: Condvar,
}

impl Control {
    pub fn arm(&self, fault: Fault) {
        self.fault.store(fault as u8, Ordering::Release);
    }
    fn take(&self, fault: Fault) -> bool {
        self.fault
            .compare_exchange(fault as u8, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
    pub fn registration(&self) -> Result<(), MetalKernelError> {
        if self.observed_residency.load(Ordering::Acquire)
            < self.required_residency.load(Ordering::Acquire)
        {
            return Err(MetalKernelError::QueueUnavailable(
                "owner residency prerequisite absent before epoch",
            ));
        }
        if self.take(Fault::Registration) {
            Err(MetalKernelError::QueueUnavailable(
                "injected completion registration refusal",
            ))
        } else {
            Ok(())
        }
    }
    pub fn require_residency_after(&self, baseline: u64) {
        self.required_residency.store(
            baseline.checked_add(1).expect("test counter capacity"),
            Ordering::Release,
        );
    }
    pub fn observe_residency(&self, calls: u64) {
        self.observed_residency.store(calls, Ordering::Release);
    }
    pub fn before_entry(&self) -> Result<(), MetalKernelError> {
        self.entry_checks.fetch_add(1, Ordering::Relaxed);
        if self.take(Fault::BeforeEntry) {
            Err(MetalKernelError::QueueUnavailable(
                "injected known-before-entry refusal",
            ))
        } else {
            Ok(())
        }
    }
    pub fn native_call_returned(&self, index: usize) -> bool {
        self.native_calls.fetch_add(1, Ordering::Relaxed);
        index == 0 && self.take(Fault::PartialEntry)
    }
    pub fn install_observer(&self, event: Retained<ProtocolObject<dyn MTLSharedEvent>>) {
        *self.observer.lock().unwrap() = Some(event);
    }
    pub fn observer(&self) -> Retained<ProtocolObject<dyn MTLSharedEvent>> {
        self.observer.lock().unwrap().as_ref().unwrap().clone()
    }
    pub fn before_return(&self) {
        if self.take(Fault::UnknownReturn) {
            panic!("controlled unknown return publication");
        }
        if !self.take(Fault::HoldReturn) {
            return;
        }
        let mut state = self.pause.lock().unwrap();
        state.0 = true;
        self.changed.notify_all();
        while !state.1 {
            state = self.changed.wait(state).unwrap();
        }
    }
    pub fn wait_for_pause(&self) -> bool {
        let state = self.pause.lock().unwrap();
        self.changed
            .wait_timeout_while(state, std::time::Duration::from_secs(2), |state| !state.0)
            .unwrap()
            .0
             .0
    }
    pub fn release(&self) {
        self.pause.lock().unwrap().1 = true;
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fault_ownership_is_one_shot_and_return_release_is_monotone() {
        let control = Control::default();
        control.arm(Fault::Registration);
        assert!(control.registration().is_err());
        control.registration().unwrap();
        control.arm(Fault::BeforeEntry);
        assert!(control.before_entry().is_err());
        control.before_entry().unwrap();
        assert_eq!(control.native_calls.load(Ordering::Relaxed), 0);
        control.release();
        control.arm(Fault::HoldReturn);
        control.before_return(); // Release-before-arrival must not lose a wake.
        assert!(control.wait_for_pause());
    }

    #[test]
    fn unknown_return_fault_is_one_shot_and_preserves_native_counter() {
        let control = Control::default();
        assert!(!control.native_call_returned(0));
        control.arm(Fault::UnknownReturn);
        let panic = std::panic::catch_unwind(|| control.before_return()).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"controlled unknown return publication")
        );
        control.before_return();
        assert_eq!(control.native_calls.load(Ordering::Relaxed), 1);
        assert_eq!(control.entry_checks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn missing_residency_callsite_is_refused_before_any_native_entry() {
        let absent = crate::kernels::metal::residency_set::PublishedRemovals::empty_for_test();
        assert!(absent.is_empty());
        drop(absent); // No allocation/custody/native publication is fabricated.
        let control = Control::default();
        control.require_residency_after(7);
        control.observe_residency(7);
        assert!(matches!(
            control.registration(),
            Err(MetalKernelError::QueueUnavailable(
                "owner residency prerequisite absent before epoch"
            ))
        ));
        assert_eq!(control.entry_checks.load(Ordering::Relaxed), 0);
        assert_eq!(control.native_calls.load(Ordering::Relaxed), 0);
        control.observe_residency(8);
        control.registration().unwrap();
        // Counter observation proves a call site, not that a native driver
        // applied residency. Actual native readback is a distinct control.
    }
}
