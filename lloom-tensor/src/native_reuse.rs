//! Dual-sided native reuse gate, separate from logical retirement and cleanup.
//!
//! Native feedback proves completion, not return of the submitting native call.
//! A Slot becomes reusable only after ordered callbacks AND host handoff return.
//! Whichever publishes second extracts once; neither side waits for the other.
//! Thus a capture destructor on the cleanup worker can submit multiple native
//! successors without depending on that same worker to return their slots.
//!
//! Unknown handoff never publishes RETURNED. Failed/panicking callbacks never
//! publish CALLBACKS_DONE. Those owners remain rooted in the queue registry;
//! queue poison is neither terminality nor permission to destroy this cell.
use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicU8, Ordering},
};

const INSTALLED: u8 = 1;
const RETURNED: u8 = 2;
const CALLBACKS_DONE: u8 = 4;
const CLAIMED: u8 = 8;
const WRITING: u8 = 16;
const RESTORED: u8 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReuseError {
    Uninstalled,
    Duplicate,
    AlreadyInstalled,
    PoolRejected,
    Claimed,
}
#[derive(Clone, Copy)]
pub(crate) enum ReuseSide {
    HandoffReturned,
    CallbacksDone,
}

/// Embedded in the existing immutable logical Submission, not an extra Arc.
/// The registry retains that Submission independently of native block disposal.
pub(crate) struct NativeReuse<T> {
    state: AtomicU8,
    value: UnsafeCell<Option<T>>,
}
// SAFETY: installation is exclusive under WRITING; no reader exists then.
// Only the unique CLAIMED winner accesses value after publication. Restoration
// is owned by the unique extraction token and permits no second extraction. Final Drop
// requires exclusive ownership of this cell, as enforced by Rust lifetimes.
unsafe impl<T: Send> Send for NativeReuse<T> {}
unsafe impl<T: Send> Sync for NativeReuse<T> {}
impl<T> NativeReuse<T> {
    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
            value: UnsafeCell::new(None),
        }
    }
    pub(crate) fn install(&self, value: T) -> Result<(), (ReuseError, T)> {
        if self
            .state
            .compare_exchange(0, WRITING, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err((ReuseError::AlreadyInstalled, value));
        }
        // SAFETY: exclusively claimed unpublished cell. Assignment drops None.
        unsafe {
            *self.value.get() = Some(value);
        }
        self.state.store(INSTALLED, Ordering::Release);
        Ok(())
    }
    /// # Safety
    /// Adapter has independent proof native handoff never occurred. No
    /// publication flag by itself establishes cancellation of native work.
    pub(crate) unsafe fn take_cancelled(&self) -> Option<T> {
        if self
            .state
            .compare_exchange(
                INSTALLED,
                INSTALLED | CLAIMED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return None;
        }
        // SAFETY: exclusive CLAIMED winner, caller supplies native cancellation.
        unsafe { (&mut *self.value.get()).take() }
    }
    /// No callbacks, allocation or Drop. A returned owner must be immediately
    /// moved to the queue's available pool, or restored on pool rejection.
    /// The adapter must not run arbitrary/panicking code with it on its stack.
    pub(crate) fn publish(
        &self,
        side: ReuseSide,
    ) -> Result<Option<ReuseRelease<'_, T>>, ReuseError> {
        let bit = match side {
            ReuseSide::HandoffReturned => RETURNED,
            ReuseSide::CallbacksDone => CALLBACKS_DONE,
        };
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & CLAIMED != 0 {
                return Err(ReuseError::Claimed);
            }
            if state & INSTALLED == 0 || state & WRITING != 0 {
                return Err(ReuseError::Uninstalled);
            }
            if state & bit != 0 {
                return Err(ReuseError::Duplicate);
            }
            match self.state.compare_exchange_weak(
                state,
                state | bit,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    state |= bit;
                    break;
                }
                Err(found) => state = found,
            }
        }
        if state & (RETURNED | CALLBACKS_DONE) != (RETURNED | CALLBACKS_DONE) {
            return Ok(None);
        }
        if self
            .state
            .compare_exchange(state, state | CLAIMED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(None);
        }
        // SAFETY: one CLAIMED CAS winner, after both independent publications.
        Ok(
            unsafe { (&mut *self.value.get()).take() }.map(|value| ReuseRelease {
                gate: self,
                value: Some(value),
            }),
        )
    }
}

/// Unique extraction token. Safe callers cannot restore a different cell or
/// race restoration with extraction: construction follows the completed take.
pub(crate) struct ReuseRelease<'a, T> {
    gate: &'a NativeReuse<T>,
    value: Option<T>,
}
impl<T> ReuseRelease<'_, T> {
    /// Adapter sink must be its non-panicking available-pool push only, not an
    /// arbitrary callback. A panic inside sink can destroy T on that stack;
    /// this helper does not promise unwind custody across arbitrary consuming code.
    pub(crate) fn release(
        mut self,
        sink: impl FnOnce(T) -> Result<(), T>,
    ) -> Result<(), ReuseError> {
        match sink(self.value.take().expect("unique extraction token")) {
            Ok(()) => Ok(()),
            Err(value) => {
                self.value = Some(value); // Drop restores rejected ownership.
                Err(ReuseError::PoolRejected) // rejected pool, owner retained
            }
        }
    }
}

impl<T> Drop for ReuseRelease<'_, T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            // SAFETY: unique token follows completed extraction; both side
            // publications are final. Its live borrow excludes final gate Drop.
            unsafe {
                *self.gate.value.get() = Some(value);
            }
            self.gate.state.fetch_or(RESTORED, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::AtomicUsize, Arc};
    struct Counted(Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn both_orders_return_once_and_unknown_handoff_retains() {
        for first in [ReuseSide::HandoffReturned, ReuseSide::CallbacksDone] {
            let count = Arc::new(AtomicUsize::new(0));
            let gate = NativeReuse::new();
            assert!(gate.install(Counted(count.clone())).is_ok());
            assert!(gate.publish(first).unwrap().is_none());
            assert_eq!(count.load(Ordering::SeqCst), 0);
            let second = match first {
                ReuseSide::HandoffReturned => ReuseSide::CallbacksDone,
                ReuseSide::CallbacksDone => ReuseSide::HandoffReturned,
            };
            let value = gate.publish(second).unwrap().expect("both sides release");
            assert!(matches!(
                gate.publish(first),
                Err(ReuseError::Duplicate | ReuseError::Claimed)
            ));
            assert_eq!(count.load(Ordering::SeqCst), 0);
            assert!(value
                .release(|owner| {
                    drop(owner);
                    Ok(())
                })
                .is_ok());
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
        let count = Arc::new(AtomicUsize::new(0));
        let rooted = NativeReuse::new();
        assert!(rooted.install(Counted(count.clone())).is_ok());
        assert!(rooted.publish(ReuseSide::CallbacksDone).unwrap().is_none());
        assert_eq!(count.load(Ordering::SeqCst), 0); // no RETURNED: retained
                                                     // Test owns no native work: dropping fixture is safe, not queue recovery.
    }
    #[test]
    fn concurrent_sides_and_rejected_pool_preserve_exact_owner() {
        for _ in 0..32 {
            let gate = Arc::new(NativeReuse::new());
            let count = Arc::new(AtomicUsize::new(0));
            assert!(gate.install(Counted(count.clone())).is_ok());
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| gate.publish(ReuseSide::CallbacksDone).unwrap());
                let producer = gate.publish(ReuseSide::HandoffReturned).unwrap();
                let callback = worker.join().unwrap();
                assert_ne!(producer.is_some(), callback.is_some());
                let value = producer.or(callback).unwrap();
                assert_eq!(value.release(Err), Err(ReuseError::PoolRejected));
            });
            assert!(matches!(
                gate.publish(ReuseSide::CallbacksDone),
                Err(ReuseError::Duplicate | ReuseError::Claimed)
            ));
            assert_eq!(count.load(Ordering::SeqCst), 0);
            drop(gate);
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
    }
    #[test]
    fn cancellation_claim_is_terminal_for_safe_publication() {
        // Deterministic causal oracle before scheduler-dependent interleavings.
        let gate = NativeReuse::new();
        let count = Arc::new(AtomicUsize::new(0));
        assert!(gate.install(Counted(count.clone())).is_ok());
        let cancelled = unsafe { gate.take_cancelled() }.expect("cancellation won");
        assert!(
            matches!(
                gate.publish(ReuseSide::CallbacksDone),
                Err(ReuseError::Claimed)
            ),
            "safe callback publication accepted a cancellation-claimed cell"
        );
        assert!(
            matches!(
                gate.publish(ReuseSide::HandoffReturned),
                Err(ReuseError::Claimed)
            ),
            "safe handoff publication accepted a cancellation-claimed cell"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        drop(cancelled);
        drop(gate);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        for _ in 0..32 {
            let gate = Arc::new(NativeReuse::new());
            let count = Arc::new(AtomicUsize::new(0));
            assert!(gate.install(Counted(count.clone())).is_ok());
            // No native work exists in this fixture. Race the production phase
            // claim against one safe publication, never fake native cancellation.
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| gate.publish(ReuseSide::CallbacksDone));
                let cancelled = unsafe { gate.take_cancelled() };
                let published = worker.join().unwrap();
                if let Some(owner) = cancelled {
                    assert!(matches!(published, Err(ReuseError::Claimed)));
                    assert!(matches!(
                        gate.publish(ReuseSide::HandoffReturned),
                        Err(ReuseError::Claimed)
                    ));
                    assert_eq!(count.load(Ordering::SeqCst), 0);
                    drop(owner);
                } else {
                    assert!(published.unwrap().is_none());
                    gate.publish(ReuseSide::HandoffReturned)
                        .unwrap()
                        .unwrap()
                        .release(|owner| {
                            drop(owner);
                            Ok(())
                        })
                        .unwrap();
                }
            });
            assert_eq!(count.load(Ordering::SeqCst), 1);
            drop(gate);
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
    }
}
