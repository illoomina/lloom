use crate::schedule_hook::schedule_point;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A notification sequence, not a mutex shared with the completion thread.
/// The waiter snapshots BEFORE checking its predicate. Registration plus the
/// sequence check under parking_lot's queue lock closes check -> park; sequential
/// consistency prevents the waker missing registration while validation misses
/// the new sequence. No parking/wake call when nobody needs to block.
#[derive(Default)]
pub(crate) struct Wake {
    epoch: AtomicUsize,
    waiters: AtomicUsize,
}
impl Wake {
    pub(crate) fn snapshot(&self) -> usize {
        let seen = self.epoch();
        schedule_point!(WakeSnapshot);
        seen
    }
    // Parking-queue validation reads this, never `snapshot`: it runs under the
    // queue lock, where no schedule point may run.
    fn epoch(&self) -> usize {
        self.epoch.load(Ordering::SeqCst)
    }
    pub(crate) fn notify(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        schedule_point!(WakeNotified);
        if self.waiters.load(Ordering::SeqCst) != 0 {
            // SAFETY: this object's own stable address is its parking key.
            // Every parked caller keeps this object alive at the same address.
            unsafe {
                parking_lot_core::unpark_all(
                    self as *const Self as usize,
                    parking_lot_core::DEFAULT_UNPARK_TOKEN,
                );
            }
        }
    }
    #[cfg(test)]
    pub(crate) fn test_is_parked(&self) -> bool {
        let mut parked = false;
        // SAFETY: the test retains this Wake at its stable parking address.
        // This mirrors the existing registered-waiter control: observe a
        // validated queue member without removing it or changing the epoch.
        // The filter cannot panic or reenter the parking queue.
        unsafe {
            parking_lot_core::unpark_filter(
                self as *const Self as usize,
                |_| {
                    parked = true;
                    parking_lot_core::FilterOp::Skip
                },
                |_| parking_lot_core::DEFAULT_UNPARK_TOKEN,
            );
        }
        parked
    }

    pub(crate) fn wait(&self, seen: usize) {
        self.park(seen, None);
    }
    pub(crate) fn park(
        &self,
        seen: usize,
        deadline: Option<std::time::Instant>,
    ) -> parking_lot_core::ParkResult {
        self.waiters.fetch_add(1, Ordering::SeqCst);
        schedule_point!(WakeRegistered);
        // SAFETY: same owned key as notify; validation only reads an atomic,
        // cannot panic or reenter parking_lot. The caller keeps self alive.
        let result = unsafe {
            parking_lot_core::park(
                self as *const Self as usize,
                || self.epoch() == seen,
                || {},
                |_, _| {},
                parking_lot_core::DEFAULT_PARK_TOKEN,
                deadline,
            )
        };
        self.waiters.fetch_sub(1, Ordering::SeqCst);
        schedule_point!(WakeParkReturned);
        result
    }
}

/// Residency membership lock only: a successful acquisition notifies the
/// independently owned service AFTER unlocking, including unwind/early return.
/// The wake is installed once before a lifecycle entry becomes pollable. A
/// guard acquired before installation reads the OnceLock at RELEASE, not birth.
/// Busy acquisition does not notify itself. This bounds host lock acquisition,
/// not the synchronous native residency commit performed while holding it.
pub(crate) struct WakeMutex<T> {
    value: parking_lot::Mutex<T>,
    wake: std::sync::OnceLock<std::sync::Arc<Wake>>,
}
impl<T> WakeMutex<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: parking_lot::Mutex::new(value),
            wake: std::sync::OnceLock::new(),
        }
    }
    /// Same exact service is idempotent. Preserve a conflicting supplied owner
    /// on rejection; never replace a wake used by an existing lifecycle poll.
    pub(crate) fn install_wake(
        &self,
        wake: std::sync::Arc<Wake>,
    ) -> Result<(), std::sync::Arc<Wake>> {
        match self.wake.set(wake) {
            Ok(()) => Ok(()),
            Err(wake)
                if std::sync::Arc::ptr_eq(self.wake.get().expect("installed wake"), &wake) =>
            {
                Ok(())
            }
            Err(wake) => Err(wake),
        }
    }
    pub(crate) fn lock(&self) -> WakeGuard<'_, T> {
        WakeGuard {
            value: Some(self.value.lock()),
            wake: &self.wake,
        }
    }
    #[cfg(all(test, feature = "schedule-hooks"))]
    pub(crate) fn test_is_locked(&self) -> bool {
        self.value.is_locked()
    }
    pub(crate) fn try_lock(&self) -> Option<WakeGuard<'_, T>> {
        self.value.try_lock().map(|guard| WakeGuard {
            value: Some(guard),
            wake: &self.wake,
        })
    }
}
pub(crate) struct WakeGuard<'a, T> {
    value: Option<parking_lot::MutexGuard<'a, T>>,
    wake: &'a std::sync::OnceLock<std::sync::Arc<Wake>>,
}
impl<T> std::ops::Deref for WakeGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_deref().expect("live membership guard")
    }
}
impl<T> std::ops::DerefMut for WakeGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_deref_mut().expect("live membership guard")
    }
}
// Static-dispatch ordering seam shared by shipping Drop and scalar control.
// The closure is not stored in the mutex/guard and adds no runtime test hook.
fn unlock_then_notify<G>(guard: G, notify: impl FnOnce()) {
    drop(guard);
    notify();
}
impl<T> Drop for WakeGuard<'_, T> {
    fn drop(&mut self) {
        let wake = self.wake;
        unlock_then_notify(self.value.take(), || {
            schedule_point!(GuardUnlocked);
            // Resolve installation at release, AFTER lock custody has ended.
            if let Some(wake) = wake.get() {
                wake.notify();
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    pub(crate) fn registration_control() {
        use std::time::{Duration, Instant};
        let wake = Arc::new(Wake::default());
        let seen = wake.snapshot();
        wake.notify();
        assert_eq!(
            wake.park(seen, Some(Instant::now() + Duration::from_secs(1))),
            parking_lot_core::ParkResult::Invalid,
            "notification before registration was lost"
        );
        let (tx, rx) = std::sync::mpsc::sync_channel(0);
        let worker_wake = wake.clone();
        let worker = std::thread::spawn(move || {
            for _ in 0..1000 {
                let seen = worker_wake.snapshot();
                tx.send(()).unwrap();
                let result = worker_wake.park(seen, Some(Instant::now() + Duration::from_secs(1)));
                assert_ne!(
                    result,
                    parking_lot_core::ParkResult::TimedOut,
                    "lost completion wake"
                );
            }
        });
        for _ in 0..1000 {
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
            wake.notify();
        }
        worker.join().unwrap();
        assert_eq!(wake.waiters.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn notification_before_and_during_registration_is_not_lost() {
        registration_control();
    }

    #[test]
    fn a_registered_waiter_is_unparked_and_registration_is_released() {
        use std::time::{Duration, Instant};
        let wake = Arc::new(Wake::default());
        let worker_wake = wake.clone();
        let seen = wake.snapshot();
        let worker = std::thread::spawn(move || {
            worker_wake.park(seen, Some(Instant::now() + Duration::from_secs(2)))
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut registered = false;
        while !registered && Instant::now() < deadline {
            // SAFETY: the Arc keeps this parking key stable and alive. The
            // filter only observes the queue; it cannot panic or reenter it.
            unsafe {
                parking_lot_core::unpark_filter(
                    Arc::as_ptr(&wake) as usize,
                    |_| {
                        registered = true;
                        parking_lot_core::FilterOp::Skip
                    },
                    |_| parking_lot_core::DEFAULT_UNPARK_TOKEN,
                );
            }
            std::thread::yield_now();
        }
        wake.notify();
        let result = worker.join().unwrap();
        assert!(registered, "worker never registered its wait");
        assert_ne!(
            result,
            parking_lot_core::ParkResult::TimedOut,
            "registered waiter lost notification"
        );
        assert_eq!(wake.waiters.load(Ordering::SeqCst), 0);
    }
}

#[cfg(test)]
mod residency_guard_tests {
    use super::*;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    #[test]
    fn residency_guard_busy_has_no_self_notification() {
        let value = WakeMutex::new(7usize);
        let wake = Arc::new(Wake::default());
        assert!(value.install_wake(wake.clone()).is_ok());
        let held = value.lock();
        let seen = wake.snapshot();
        assert!(value.try_lock().is_none());
        assert_eq!(wake.snapshot(), seen, "Busy notified its own retry");
        drop(held);
        assert_eq!(wake.snapshot(), seen.wrapping_add(1));
    }

    #[test]
    fn residency_guard_release_unlocks_before_single_notification() {
        let value = WakeMutex::new(1usize);
        let wake = Arc::new(Wake::default());
        assert!(value.install_wake(wake.clone()).is_ok());
        let seen = wake.snapshot();
        let mut held = value.try_lock().unwrap();
        *held = 9;
        assert_eq!(wake.snapshot(), seen);
        drop(held);
        assert_eq!(wake.snapshot(), seen.wrapping_add(1));
        assert_eq!(*value.value.try_lock().unwrap(), 9, "not unlocked");
        // Actual Drop shares this static ordering helper. Check the mutex at
        // NOTIFICATION instant, not merely after the complete Drop returned.
        let held = value.value.lock();
        let seen = wake.snapshot();
        unlock_then_notify(held, || {
            assert!(
                value.value.try_lock().is_some(),
                "notification preceded membership unlock"
            );
            wake.notify();
        });
        assert_eq!(wake.snapshot(), seen.wrapping_add(1));
    }

    #[test]
    fn residency_guard_acquired_before_install_reads_wake_at_release() {
        let value = WakeMutex::new(());
        let wake = Arc::new(Wake::default());
        let held = value.lock();
        let seen = wake.snapshot();
        assert!(value.install_wake(wake.clone()).is_ok());
        assert_eq!(wake.snapshot(), seen);
        drop(held);
        assert_eq!(
            wake.snapshot(),
            seen.wrapping_add(1),
            "captured absent wake"
        );
    }

    #[test]
    fn residency_guard_wake_is_same_owner_idempotent_and_foreign_preserved() {
        let value = WakeMutex::new(());
        let wake = Arc::new(Wake::default());
        let foreign = Arc::new(Wake::default());
        assert!(value.install_wake(wake.clone()).is_ok());
        assert!(value.install_wake(wake.clone()).is_ok());
        let rejected = value.install_wake(foreign.clone()).err().unwrap();
        assert!(Arc::ptr_eq(&rejected, &foreign));
        let seen = wake.snapshot();
        let foreign_seen = foreign.snapshot();
        drop(value.lock());
        assert_eq!(wake.snapshot(), seen.wrapping_add(1));
        assert_eq!(foreign.snapshot(), foreign_seen, "foreign wake installed");
    }

    #[test]
    fn residency_guard_unwind_unlocks_and_notifies() {
        let value = WakeMutex::new(());
        let wake = Arc::new(Wake::default());
        assert!(value.install_wake(wake.clone()).is_ok());
        let seen = wake.snapshot();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _held = value.lock();
            panic!("injected residency guard unwind");
        }));
        assert!(panic.is_err());
        assert_eq!(wake.snapshot(), seen.wrapping_add(1));
        assert!(value.value.try_lock().is_some(), "unwind retained lock");
    }
}
