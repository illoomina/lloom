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
        self.epoch.load(Ordering::SeqCst)
    }
    pub(crate) fn notify(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
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
    pub(crate) fn wait(&self, seen: usize) {
        self.park(seen, None);
    }
    pub(crate) fn park(
        &self,
        seen: usize,
        deadline: Option<std::time::Instant>,
    ) -> parking_lot_core::ParkResult {
        self.waiters.fetch_add(1, Ordering::SeqCst);
        // SAFETY: same owned key as notify; validation only reads an atomic,
        // cannot panic or reenter parking_lot. The caller keeps self alive.
        let result = unsafe {
            parking_lot_core::park(
                self as *const Self as usize,
                || self.snapshot() == seen,
                || {},
                |_, _| {},
                parking_lot_core::DEFAULT_PARK_TOKEN,
                deadline,
            )
        };
        self.waiters.fetch_sub(1, Ordering::SeqCst);
        result
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
