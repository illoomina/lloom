//! Test-only, per-owner fault points. Never compiled into the shipped library.
//! These controls carry no owner authority and do not replace native admission.
use super::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLSharedEvent;
use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    Condvar, Mutex,
};

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Fault {
    Registration = 1,
    BeforeEntry = 2,
    PartialEntry = 3,
    HoldReturn = 4,
}

#[derive(Default)]
pub(crate) struct Control {
    fault: AtomicU8,
    pub native_calls: AtomicUsize,
    pub entry_checks: AtomicUsize,
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
        if self.take(Fault::Registration) {
            Err(MetalKernelError::QueueUnavailable(
                "injected completion registration refusal",
            ))
        } else {
            Ok(())
        }
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
}
