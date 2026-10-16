//! What every command buffer a [`Commands`](super::Commands) commits runs when
//! it completes: raise the failure latch on a native error, mark the buffer's
//! epoch complete on the queue's clock, run the completion subscribers.
//!
//! Built once with the queue and installed in its completion adapter (the
//! Metal 3 `Completions`, the Metal 4 `Queue`). A commit only flags its record
//! for the hook, so it registers no closure, clones no `Arc`, captures no epoch
//! and takes no lock (`docs/tuning.md` §13.5x).
//!
//! The epoch is a count, not a capture. Commits flag their records and take
//! their epochs in one order, under the queue's commit exclusion; the adapter
//! runs flagged records strictly in that order, one at a time, whatever order
//! Metal reports completions in. So the n-th run completes epoch n. A run that
//! would complete an epoch not yet committed is that argument failing, and it
//! latches a failure rather than mark it.
//!
//! Queue observers are configured before an irreversible setup seal. The seal
//! excludes the entire snapshot publication interval atomically; submission
//! refuses setup-busy before native preparation rather than locking against a
//! publisher. Completions retain their one acquire load of immutable storage.

use super::{commands::CommandBufferFailure, metal4::Feedback, GpuClock};
use crate::retire::Epoch;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;

pub(super) type Subscriber = Arc<dyn Fn() + Send + Sync>;
type Snapshot = Vec<Subscriber>;

pub(super) struct CompletionHook {
    clock: Arc<GpuClock>,
    failure: Arc<CommandBufferFailure>,
    /// Runs so far; the epoch the next run completes.
    runs: AtomicU64,
    subscribers: Subscribers,
}

impl CompletionHook {
    pub(super) fn new(clock: Arc<GpuClock>, failure: Arc<CommandBufferFailure>) -> Self {
        Self {
            clock,
            failure,
            runs: AtomicU64::new(0),
            subscribers: Subscribers::new(),
        }
    }

    /// One flagged record completed, in commit order (module docs).
    pub(super) fn complete(&self, feedback: &Feedback) {
        if let Some(error) = &feedback.error {
            self.failure.record(error.clone());
        }
        let epoch = Epoch(self.runs.fetch_add(1, Ordering::Relaxed));
        if epoch.0 >= self.clock.current_epoch().0 {
            self.failure
                .record_static("a completion ran ahead of the commits it reports");
            return;
        }
        self.clock.mark_completed(epoch);
        self.subscribers.run();
    }

    pub(super) fn subscribe(
        &self,
        f: Subscriber,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        self.subscribers.subscribe(f)
    }

    pub(super) fn seal(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.subscribers.setup.seal().map_err(setup_error)
    }

    /// Hold the subscribers' lock on the calling thread while `f` runs, so a
    /// test can show a commit and a completion never take it.
    #[cfg(test)]
    pub(super) fn while_subscribers_locked<R>(&self, f: impl FnOnce() -> R) -> R {
        let _published = self.subscribers.published.lock();
        f()
    }

    #[cfg(test)]
    pub(super) fn while_registration_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let _permit = self.subscribers.setup.register().expect("open setup");
        f()
    }

    /// Final teardown, once no completion can run again: every published
    /// subscriber snapshot, for the caller to drop off this core. Locks first
    /// and transfers on [`SubscriberTake::take`], so a caller can take all
    /// its locks before its first transfer.
    pub(super) fn try_take_subscribers(&self) -> Option<SubscriberTake<'_>> {
        self.subscribers
            .published
            .try_lock()
            .map(|published| SubscriberTake {
                published,
                current: &self.subscribers.current,
            })
    }
}

/// The subscribers, published as an immutable snapshot behind one pointer: a
/// completion loads it once and takes no lock. Every snapshot ever published
/// stays owned here, the first (empty) one for the life of the hook, so the
/// pointer a completion loaded is valid however late it reads it.
/// Registration is setup-time and rare (`Commands::on_command_buffer_complete`).
// The `Box` is load-bearing (`clippy::vec_box` below): it keeps each
// snapshot's `Vec` header at an address `current` can point at while
// `published` itself reallocates.
#[allow(clippy::vec_box)]
struct Subscribers {
    setup: crate::completion_setup::Setup,
    current: AtomicPtr<Snapshot>,
    published: parking_lot::Mutex<Vec<Box<Snapshot>>>,
    /// Force the reserve/header refusal paths without replacing their handling.
    #[cfg(test)]
    refuse_reserve: std::sync::atomic::AtomicU8,
}

impl Subscribers {
    fn new() -> Self {
        let empty = Box::new(Snapshot::new());
        Self {
            setup: crate::completion_setup::Setup::new(),
            current: AtomicPtr::new(&*empty as *const Snapshot as *mut Snapshot),
            published: parking_lot::Mutex::new(vec![empty]),
            #[cfg(test)]
            refuse_reserve: std::sync::atomic::AtomicU8::new(0),
        }
    }

    fn subscribe(
        &self,
        f: Subscriber,
    ) -> Result<(), crate::kernels::err::CompletionRegistrationError> {
        let _permit = match self.setup.register() {
            Ok(permit) => permit,
            Err(reason) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    setup_error(reason),
                    Some(f),
                ))
            }
        };
        let mut published = self.published.lock();
        let previous = published.last().expect("the empty snapshot");
        let mut next = Snapshot::new();
        let snapshot_additional = previous.len() + 1;
        #[cfg(test)]
        let snapshot_additional = if self
            .refuse_reserve
            .compare_exchange(1, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            usize::MAX
        } else {
            snapshot_additional
        };
        if next.try_reserve_exact(snapshot_additional).is_err() {
            return Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "subscriber snapshot allocation refused",
                ),
                Some(f),
            ));
        }
        next.extend(previous.iter().cloned());
        let history_additional = 1;
        #[cfg(test)]
        let history_additional = if self
            .refuse_reserve
            .compare_exchange(2, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            usize::MAX
        } else {
            history_additional
        };
        if published.try_reserve(history_additional).is_err() {
            return Err(crate::kernels::err::CompletionRegistrationError::new(
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "subscriber history allocation refused",
                ),
                Some(f),
            ));
        }
        next.push(f);
        #[cfg(test)]
        let refuse_header = self
            .refuse_reserve
            .compare_exchange(3, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok();
        #[cfg(not(test))]
        let refuse_header = false;
        let next = match if refuse_header {
            Err(next)
        } else {
            crate::cold_box::try_new(next)
        } {
            Ok(next) => next,
            Err(mut rejected) => {
                let callback = rejected.pop().expect("new callback is last");
                // Every earlier clone still has an original in published.
                // The rejected arbitrary capture itself escapes in the error.
                drop(rejected);
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    crate::kernels::MetalKernelError::QueueUnavailable(
                        "subscriber snapshot header allocation refused",
                    ),
                    Some(callback),
                ));
            }
        };
        self.current.store(
            &*next as *const Snapshot as *mut Snapshot,
            Ordering::Release,
        );
        published.push(next);
        Ok(())
    }

    fn run(&self) {
        // SAFETY: `current` always points at a snapshot `published` owns; a
        // snapshot is never mutated after publication, and is dropped only by
        // `SubscriberTake::take`, which runs once no completion can.
        let snapshot = unsafe { &*self.current.load(Ordering::Acquire) };
        for f in snapshot {
            f();
        }
    }
}

fn setup_error(reason: crate::completion_setup::Refusal) -> crate::kernels::MetalKernelError {
    match reason {
        crate::completion_setup::Refusal::Busy => {
            crate::kernels::MetalKernelError::SubscriberSetupBusy
        }
        crate::completion_setup::Refusal::Sealed => {
            crate::kernels::MetalKernelError::SubscriberConfigurationSealed
        }
    }
}

#[allow(clippy::vec_box)] // as `Subscribers`
pub(super) struct SubscriberTake<'a> {
    published: parking_lot::MutexGuard<'a, Vec<Box<Snapshot>>>,
    current: &'a AtomicPtr<Snapshot>,
}

impl SubscriberTake<'_> {
    #[allow(clippy::vec_box)] // as `Subscribers`
    pub(super) fn take(mut self) -> Vec<Box<Snapshot>> {
        let empty = &*self.published[0] as *const Snapshot as *mut Snapshot;
        self.current.store(empty, Ordering::Release);
        self.published.drain(1..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn hook() -> (Arc<GpuClock>, Arc<CommandBufferFailure>, CompletionHook) {
        let clock = Arc::new(GpuClock::new());
        let failure = Arc::new(CommandBufferFailure::default());
        let hook = CompletionHook::new(Arc::clone(&clock), Arc::clone(&failure));
        (clock, failure, hook)
    }

    fn clean() -> Feedback {
        Feedback {
            error: None,
            start: 0.,
            end: 1.,
        }
    }

    #[test]
    fn reserve_refusal_keeps_publication_captures_and_permit_reusable() {
        struct Capture {
            producer: Arc<parking_lot::Mutex<()>>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for Capture {
            fn drop(&mut self) {
                let _guard = self
                    .producer
                    .try_lock()
                    .expect("refused callback dropped under producer lock");
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }
        for stage in [1, 2, 3] {
            let (clock, _, hook) = hook();
            let runs = Arc::new(AtomicUsize::new(0));
            let observed = runs.clone();
            let survivor: Subscriber = Arc::new(move || {
                observed.fetch_add(1, Ordering::Relaxed);
            });
            hook.subscribe(survivor.clone()).unwrap();
            let pointer = hook.subscribers.current.load(Ordering::Acquire);
            let history_len = hook.subscribers.published.lock().len();
            let survivor_refs = Arc::strong_count(&survivor);
            let producer = Arc::new(parking_lot::Mutex::new(()));
            let drops = Arc::new(AtomicUsize::new(0));
            let capture = Capture {
                producer: producer.clone(),
                drops: drops.clone(),
            };
            hook.subscribers
                .refuse_reserve
                .store(stage, Ordering::Relaxed);
            let guard = producer.lock();
            let error = hook
                .subscribe(Arc::new(move || {
                    let _ = &capture;
                    panic!("refused observer installed");
                }))
                .unwrap_err();
            let expected = match stage {
                1 => "subscriber snapshot allocation refused",
                2 => "subscriber history allocation refused",
                3 => "subscriber snapshot header allocation refused",
                _ => unreachable!(),
            };
            assert!(matches!(
                error.cause(),
                crate::kernels::MetalKernelError::QueueUnavailable(message) if *message == expected
            ));
            assert_eq!(hook.subscribers.current.load(Ordering::Acquire), pointer);
            assert_eq!(hook.subscribers.published.lock().len(), history_len);
            assert_eq!(Arc::strong_count(&survivor), survivor_refs);
            assert_eq!(drops.load(Ordering::Relaxed), 0);
            drop(guard);
            drop(error);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            // Failure released REGISTERING without publishing or sealing.
            hook.subscribe(Arc::new(|| {})).unwrap();
            hook.seal().unwrap();
            clock.commit_epoch();
            hook.complete(&clean());
            assert_eq!(runs.load(Ordering::Relaxed), 1);
            let snapshots = hook.try_take_subscribers().unwrap().take();
            drop(snapshots);
            assert_eq!(Arc::strong_count(&survivor), 1);
        }
    }

    #[test]
    fn completion_borrows_snapshot_without_cloning_subscribers() {
        let (clock, _, hook) = hook();
        let observed = Arc::new(AtomicUsize::new(0));
        let weak_slot =
            Arc::new(std::sync::OnceLock::<std::sync::Weak<dyn Fn() + Send + Sync>>::new());
        let record = observed.clone();
        let self_weak = weak_slot.clone();
        let subscriber: Subscriber = Arc::new(move || {
            record.store(self_weak.get().unwrap().strong_count(), Ordering::Relaxed);
        });
        let weak = Arc::downgrade(&subscriber);
        weak_slot.set(weak.clone()).unwrap();
        hook.subscribe(subscriber).unwrap();
        hook.seal().unwrap();
        let before = weak.strong_count();
        clock.commit_epoch();
        hook.complete(&clean());
        assert_eq!(
            observed.load(Ordering::Relaxed),
            before,
            "completion cloned subscriber ownership"
        );
    }

    #[test]
    fn snapshot_history_lives_until_take_and_teardown_never_reopens_setup() {
        let (_, _, hook) = hook();
        let subscriber: Subscriber = Arc::new(|| {});
        hook.subscribe(subscriber.clone()).unwrap();
        hook.subscribe(Arc::new(|| {})).unwrap();
        hook.seal().unwrap();
        assert_eq!(
            hook.subscribers.published.lock().len(),
            3,
            "published snapshot retired before teardown"
        );
        assert_eq!(Arc::strong_count(&subscriber), 3);
        let snapshots = hook.try_take_subscribers().unwrap().take();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(Arc::strong_count(&subscriber), 3);
        drop(snapshots);
        assert_eq!(Arc::strong_count(&subscriber), 1);
        // Teardown is not a new model/setup phase; no reset operation exists.
        hook.seal().unwrap();
        let rejected = hook.subscribe(Arc::new(|| {}));
        assert!(
            matches!(
                rejected,
                Err(ref error) if matches!(error.cause(), crate::kernels::MetalKernelError::SubscriberConfigurationSealed)
            ),
            "teardown or repeated seal reopened setup"
        );
    }

    #[test]
    fn seal_refuses_while_registration_owns_publication() {
        let (clock, _, hook) = hook();
        let permit = hook.subscribers.setup.register().unwrap();
        assert!(matches!(
            hook.seal(),
            Err(crate::kernels::MetalKernelError::SubscriberSetupBusy)
        ));
        assert_eq!(clock.current_epoch().0, 0);
        let pointer = hook.subscribers.current.load(Ordering::Acquire);
        let rejected = hook
            .subscribe(Arc::new(|| panic!("busy observer installed")))
            .unwrap_err();
        assert!(matches!(
            rejected.cause(),
            crate::kernels::MetalKernelError::SubscriberSetupBusy
        ));
        assert_eq!(hook.subscribers.current.load(Ordering::Acquire), pointer);
        drop(permit);
        drop(rejected);
        hook.subscribe(Arc::new(|| {})).unwrap();
        hook.seal().unwrap();
    }

    #[test]
    fn late_capture_is_returned_until_the_callers_locks_are_released() {
        struct Probe {
            lock: Arc<parking_lot::Mutex<()>>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                let _guard = self
                    .lock
                    .try_lock()
                    .expect("callback capture dropped under producer lock");
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }
        let (_, _, hook) = hook();
        hook.seal().unwrap();
        let lock = Arc::new(parking_lot::Mutex::new(()));
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = Probe {
            lock: lock.clone(),
            drops: drops.clone(),
        };
        let guard = lock.lock();
        let rejected = hook
            .subscribe(Arc::new(move || {
                let _ = &probe;
            }))
            .unwrap_err();
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(guard);
        drop(rejected);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(hook.subscribers.published.lock().len(), 1);
        hook.seal().unwrap();
        assert!(
            hook.subscribe(Arc::new(|| {})).is_err(),
            "model reset must not reopen setup"
        );
    }

    /// The n-th run completes epoch n: the clock reads exactly the runs, and
    /// a run beyond the commits latches rather than marks.
    #[test]
    fn each_run_completes_the_next_committed_epoch_and_no_further() {
        let (clock, failure, hook) = hook();
        for _ in 0..3 {
            clock.commit_epoch();
        }
        for n in 1..=3 {
            hook.complete(&clean());
            assert_eq!(clock.completed_count().0, n);
            assert!(failure.check().is_ok());
        }
        hook.complete(&clean());
        assert_eq!(
            clock.completed_count().0,
            3,
            "marked an epoch never committed"
        );
        assert!(failure.check().is_err(), "the overrun was not latched");
    }

    /// A native error latches and still completes its epoch, as the old
    /// per-commit closure did.
    #[test]
    fn an_error_latches_and_its_epoch_still_completes() {
        let (clock, failure, hook) = hook();
        clock.commit_epoch();
        hook.complete(&Feedback {
            error: Some("Invalid Resource".into()),
            start: 0.,
            end: 1.,
        });
        assert_eq!(clock.completed_count().0, 1);
        let message = failure.check().unwrap_err().to_string();
        assert!(message.contains("Invalid Resource"), "{message}");
    }

    /// A setup subscriber runs on every completion after sealing, the
    /// teardown take leaves none to run, and every subscriber is handed over
    /// exactly once.
    #[test]
    fn subscribers_run_after_sealing_and_teardown_takes_them_all() {
        let (clock, _, hook) = hook();
        let runs = Arc::new(AtomicUsize::new(0));
        for _ in 0..4 {
            clock.commit_epoch();
        }
        for _ in 0..2 {
            let r = Arc::clone(&runs);
            hook.subscribe(Arc::new(move || {
                r.fetch_add(1, Ordering::Relaxed);
            }))
            .unwrap();
        }
        hook.seal().unwrap();
        let rejected = hook
            .subscribe(Arc::new(|| panic!("late subscriber ran")))
            .unwrap_err();
        assert!(matches!(
            rejected.cause(),
            crate::kernels::MetalKernelError::SubscriberConfigurationSealed
        ));
        drop(rejected);
        hook.complete(&clean());
        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "both subscribers, once each"
        );
        let taken = hook.try_take_subscribers().expect("uncontended").take();
        assert_eq!(taken.len(), 2, "one snapshot per registration");
        assert_eq!(taken.last().unwrap().len(), 2);
        hook.complete(&clean());
        assert_eq!(runs.load(Ordering::Relaxed), 2, "a taken subscriber ran");
        assert_eq!(Arc::strong_count(&runs), 1 + 2, "one clone per subscriber");
        drop(taken);
        assert_eq!(
            Arc::strong_count(&runs),
            1,
            "a subscriber outlived the take"
        );
    }
}
