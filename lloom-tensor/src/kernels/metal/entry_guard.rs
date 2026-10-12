use std::ops::{Deref, DerefMut};

/// Custody outlives the lock, including early return and unwinding. Payload
/// destructors may reenter the protected state, so field order alone is not
/// the unlock boundary.
pub(super) struct DeferredGuard<G, C> {
    guard: Option<G>,
    custody: C,
}

impl<G, C> DeferredGuard<G, C> {
    pub(super) fn new(guard: G, custody: C) -> Self {
        Self {
            guard: Some(guard),
            custody,
        }
    }

    pub(super) fn parts_mut(&mut self) -> (&mut G, &mut C) {
        (
            self.guard.as_mut().expect("live deferred guard"),
            &mut self.custody,
        )
    }
}

impl<G: Deref, C> Deref for DeferredGuard<G, C> {
    type Target = G::Target;
    fn deref(&self) -> &Self::Target {
        self.guard.as_ref().expect("live deferred guard").deref()
    }
}

impl<G: DerefMut, C> DerefMut for DeferredGuard<G, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard
            .as_mut()
            .expect("live deferred guard")
            .deref_mut()
    }
}

impl<G, C> Drop for DeferredGuard<G, C> {
    fn drop(&mut self) {
        drop(self.guard.take());
    }
}

/// Each bound is derived from one acquisition's caller census. Exhaustion
/// refuses before close or ownership transfer; no existing owner is replaced.
pub(super) struct DeferredSlots<T, const N: usize>([Option<T>; N]);

impl<T, const N: usize> Default for DeferredSlots<T, N> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| None))
    }
}

impl<T, const N: usize> DeferredSlots<T, N> {
    pub(super) fn reserve(&mut self) -> Option<&mut Option<T>> {
        self.0.iter_mut().find(|slot| slot.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::{DeferredGuard, DeferredSlots};
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    struct Reenter<'a> {
        lock: &'a parking_lot::Mutex<()>,
        drops: &'a RefCell<Vec<&'static str>>,
        name: &'static str,
    }

    impl Drop for Reenter<'_> {
        fn drop(&mut self) {
            assert!(
                self.lock.try_lock().is_some(),
                "payload destructor ran under EntryState lock"
            );
            self.drops.borrow_mut().push(self.name);
        }
    }

    #[test]
    fn deferred_payload_reenters_after_normal_unlock() {
        let lock = parking_lot::Mutex::new(());
        let drops = RefCell::new(Vec::new());
        let payload = Reenter {
            lock: &lock,
            drops: &drops,
            name: "normal",
        };
        let guard = DeferredGuard::new(lock.lock(), payload);
        assert!(lock.try_lock().is_none());
        drop(guard);
        assert_eq!(*drops.borrow(), ["normal"]);
    }

    #[test]
    fn deferred_payload_reenters_after_error_unlock() {
        fn refuse(
            lock: &parking_lot::Mutex<()>,
            drops: &RefCell<Vec<&'static str>>,
        ) -> Result<(), ()> {
            let payload = Reenter {
                lock,
                drops,
                name: "error",
            };
            let _guard = DeferredGuard::new(lock.lock(), payload);
            Err(())
        }
        let lock = parking_lot::Mutex::new(());
        let drops = RefCell::new(Vec::new());
        assert_eq!(refuse(&lock, &drops), Err(()));
        assert_eq!(*drops.borrow(), ["error"]);
    }

    #[test]
    fn deferred_payload_reenters_after_unwind_unlock() {
        let lock = parking_lot::Mutex::new(());
        let drops = RefCell::new(Vec::new());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let payload = Reenter {
                lock: &lock,
                drops: &drops,
                name: "unwind",
            };
            let _guard = DeferredGuard::new(lock.lock(), payload);
            panic!("controlled unwind");
        }))
        .is_err());
        assert_eq!(*drops.borrow(), ["unwind"]);
    }

    #[test]
    fn full_slots_refuse_without_replacing_or_moving_any_owner() {
        let lock = parking_lot::Mutex::new(());
        let drops = RefCell::new(Vec::new());
        let mut source = Some(Reenter {
            lock: &lock,
            drops: &drops,
            name: "unmoved",
        });
        let mut guard = DeferredGuard::new(lock.lock(), DeferredSlots::<_, 2>::default());
        for name in ["predecessor", "own"] {
            let (_, custody) = guard.parts_mut();
            *custody.reserve().expect("two caller-bound slots") = Some(Reenter {
                lock: &lock,
                drops: &drops,
                name,
            });
        }
        let (_, custody) = guard.parts_mut();
        assert!(
            custody.reserve().is_none(),
            "exhaustion returned occupied slot"
        );
        if let Some(destination) = custody.reserve() {
            *destination = source.take();
        }
        assert!(source.is_some(), "exhaustion moved caller ownership");
        assert!(
            drops.borrow().is_empty(),
            "exhaustion replaced retained ownership"
        );
        drop(guard);
        assert_eq!(*drops.borrow(), ["predecessor", "own"]);
        drop(source);
        assert_eq!(*drops.borrow(), ["predecessor", "own", "unmoved"]);
    }
}
