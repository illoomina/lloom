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
//!
//! A final owner whose current recording a cancellation abandoned submits
//! nothing (`docs/invariants.md` §15.1q). The epochs still owed then complete
//! by accounting: once every record that carried hook runs has retired in its
//! adapter, the hook advances its run count over them and marks them, with no
//! native feedback and no subscriber run, since no command buffer completed.

use super::{commands::CommandBufferFailure, metal4::Feedback, GpuClock};
use crate::completion_wake::Wake;
use crate::retire::Epoch;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

pub(super) type Subscriber = Arc<dyn Fn() + Send + Sync>;
type Snapshot = Vec<Subscriber>;

pub(super) struct CompletionHook {
    clock: Arc<GpuClock>,
    failure: Arc<CommandBufferFailure>,
    /// Runs so far; the epoch the next run completes.
    runs: AtomicU64,
    /// Runs whose record has retired in its adapter: callbacks, these runs and
    /// everything the adapter does for the record after them have returned.
    retired: AtomicU64,
    /// An abandoned final's owed epochs `[from, through)`, published once.
    abandoned: AtomicBool,
    abandoned_from: AtomicU64,
    abandoned_through: AtomicU64,
    /// The abandoned epochs are marked and every run below them retired.
    settled: AtomicBool,
    /// The host retirement service's Wake, notified once `settled` is set.
    final_wake: OnceLock<Arc<Wake>>,
    subscribers: Subscribers,
}

impl CompletionHook {
    pub(super) fn new(clock: Arc<GpuClock>, failure: Arc<CommandBufferFailure>) -> Self {
        Self {
            clock,
            failure,
            runs: AtomicU64::new(0),
            retired: AtomicU64::new(0),
            abandoned: AtomicBool::new(false),
            abandoned_from: AtomicU64::new(0),
            abandoned_through: AtomicU64::new(0),
            settled: AtomicBool::new(false),
            final_wake: OnceLock::new(),
            subscribers: Subscribers::new(),
        }
    }

    /// Cold, once: the Wake a settled abandonment notifies.
    pub(super) fn install_final_wake(&self, wake: Arc<Wake>) -> Result<(), Arc<Wake>> {
        match self.final_wake.get() {
            Some(installed) if Arc::ptr_eq(installed, &wake) => Ok(()),
            Some(_) => Err(wake),
            None => self.final_wake.set(wake),
        }
    }

    /// A record that carried `runs` hook runs has retired in its adapter.
    /// Called once per record, after the adapter's last step for it; records
    /// retire one at a time, in commit order.
    pub(super) fn retire(&self, runs: u32) {
        if runs == 0 {
            return;
        }
        self.retired.fetch_add(u64::from(runs), Ordering::SeqCst);
        self.settle_abandoned();
    }

    /// The final owner abandoned its current recording with epochs
    /// `[from, through)` owed, `from` being every hook run accepted
    /// submissions carried and `through` the clock's reserved epochs. They
    /// complete once every run below `from` has retired: at once when none
    /// is pending, otherwise on the retire of the last record still running.
    /// Published once.
    pub(super) fn abandon(
        &self,
        from: u64,
        through: u64,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        if through < from || through != self.clock.current_epoch().0 {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "abandoned epochs disagree with the clock",
            ));
        }
        if self.abandoned.load(Ordering::Relaxed) {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "a final abandonment was published twice",
            ));
        }
        self.abandoned_from.store(from, Ordering::Relaxed);
        self.abandoned_through.store(through, Ordering::Relaxed);
        self.abandoned.store(true, Ordering::SeqCst);
        self.settle_abandoned();
        Ok(())
    }

    /// A final abandonment was published.
    pub(super) fn abandonment_published(&self) -> bool {
        self.abandoned.load(Ordering::Acquire)
    }

    /// The published abandonment has settled: its epochs are marked and
    /// every record carrying a run below them has retired.
    pub(super) fn abandonment_settled(&self) -> bool {
        self.settled.load(Ordering::Acquire)
    }

    /// Mark the abandoned epochs once every run below them has retired.
    /// Both the final owner (after publishing) and every retire (after
    /// counting) look: each writes, then reads the other's write, so one of
    /// the two sees both. The run count moves from `from` to `through` in one
    /// exchange, so exactly one of them marks; the other does nothing.
    fn settle_abandoned(&self) {
        if !self.abandoned.load(Ordering::SeqCst) {
            return;
        }
        let from = self.abandoned_from.load(Ordering::Relaxed);
        let through = self.abandoned_through.load(Ordering::Relaxed);
        let retired = self.retired.load(Ordering::SeqCst);
        if retired < from {
            return;
        }
        if retired > from {
            self.failure
                .record_static("more hook runs retired than accepted submissions carried");
            self.notify_final();
            return;
        }
        match self
            .runs
            .compare_exchange(from, through, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                if through > from {
                    self.clock.mark_completed(Epoch(through - 1));
                }
                self.settled.store(true, Ordering::Release);
            }
            Err(seen) if seen == through => return,
            Err(_) => self
                .failure
                .record_static("a hook run ran past an abandoned final's epochs"),
        }
        self.notify_final();
    }

    fn notify_final(&self) {
        if let Some(wake) = self.final_wake.get() {
            wake.notify();
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

    /// Opaque test guard so a fixture can release before joining a producer
    /// even when the producer is deliberately blocked by a mutex mutant.
    #[cfg(test)]
    pub(super) fn lock_subscribers_for_test(&self) -> impl Drop + '_ {
        self.subscribers.published.lock()
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

    /// The native feedback value is plain data; no Device/native work is
    /// constructed. The actual hook publishes error before any observer runs.
    #[test]
    fn native_error_is_latched_before_the_subscriber_observes_completion() {
        let (clock, failure, hook) = hook();
        let runs = Arc::new(AtomicUsize::new(0));
        let observed_runs = runs.clone();
        let observed_clock = clock.clone();
        let observed_failure = failure.clone();
        hook.subscribe(Arc::new(move || {
            assert!(
                observed_failure.check().is_err(),
                "native error was not latched before subscriber completion"
            );
            assert_eq!(observed_clock.completed_count().0, 1);
            observed_runs.fetch_add(1, Ordering::Relaxed);
        }))
        .unwrap();
        hook.seal().unwrap();
        clock.commit_epoch();
        hook.complete(&Feedback {
            error: Some("controlled native error feedback".into()),
            start: 0.,
            end: 1.,
        });
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(clock.current_epoch().0, 1);
        assert_eq!(clock.completed_count().0, 1);
        for _ in 0..2 {
            assert!(failure
                .check()
                .unwrap_err()
                .to_string()
                .contains("controlled native error feedback"));
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

    /// `n` committed epochs, each completed and retired as its record would
    /// be (one run per record).
    fn commit_and_retire(clock: &GpuClock, hook: &CompletionHook, n: u64) {
        for _ in 0..n {
            clock.commit_epoch();
            hook.complete(&clean());
            hook.retire(1);
        }
    }

    fn with_wake(hook: &CompletionHook) -> Arc<Wake> {
        let wake = Arc::new(Wake::default());
        assert!(hook.install_final_wake(wake.clone()).is_ok());
        assert!(hook.install_final_wake(wake.clone()).is_ok());
        assert!(hook.install_final_wake(Arc::new(Wake::default())).is_err());
        wake
    }

    /// Nothing earlier pending: the owed epochs complete as the final owner
    /// publishes them, and the run count moves over them.
    #[test]
    fn an_abandonment_with_nothing_pending_completes_at_once() {
        let (clock, failure, hook) = hook();
        let wake = with_wake(&hook);
        commit_and_retire(&clock, &hook, 2);
        // Two epochs held at a cut and cancelled: reserved, never committed.
        clock.commit_epoch();
        clock.commit_epoch();
        let seen = wake.snapshot();
        hook.abandon(2, 4).unwrap();
        assert!(hook.abandonment_settled());
        assert_eq!(clock.completed_count().0, 4);
        assert_eq!(hook.runs.load(Ordering::Relaxed), 4);
        assert_ne!(wake.snapshot(), seen, "the settle did not wake the service");
        assert!(failure.check().is_ok());
    }

    /// A record still running holds the owed epochs: its run alone (counted,
    /// not retired) settles nothing, and no epoch above its own is marked;
    /// its retire settles them, on its own thread.
    #[test]
    fn an_abandonment_rides_the_retire_of_the_last_record() {
        let (clock, failure, hook) = hook();
        let wake = with_wake(&hook);
        commit_and_retire(&clock, &hook, 1);
        clock.commit_epoch(); // the record still in flight
        clock.commit_epoch(); // a cancelled one
        hook.abandon(2, 3).unwrap();
        assert!(!hook.abandonment_settled());
        assert_eq!(clock.completed_count().0, 1, "marked ahead of the prefix");
        hook.complete(&clean());
        assert!(
            !hook.abandonment_settled(),
            "settled on a run whose record had not retired"
        );
        assert_eq!(
            clock.completed_count().0,
            2,
            "the cancelled epoch was marked before the record retired"
        );
        let seen = wake.snapshot();
        hook.retire(1);
        assert!(hook.abandonment_settled());
        assert_eq!(clock.completed_count().0, 3);
        assert_ne!(wake.snapshot(), seen);
        assert!(failure.check().is_ok());
    }

    /// With nothing owed the abandonment marks nothing, and still settles
    /// only once the prefix has retired: terminality reads it.
    #[test]
    fn nothing_owed_still_waits_for_the_prefix() {
        let (clock, _, hook) = hook();
        commit_and_retire(&clock, &hook, 1);
        clock.commit_epoch();
        hook.abandon(2, 2).unwrap();
        assert!(!hook.abandonment_settled());
        hook.complete(&clean());
        hook.retire(1);
        assert!(hook.abandonment_settled());
        assert_eq!(clock.completed_count().0, 2);
    }

    /// The range must end at the clock's reserved epochs and is published
    /// once; a run after the settle is a run past it, and latches.
    #[test]
    fn an_abandonment_is_checked_against_the_clock_and_never_overrun() {
        let (clock, failure, hook) = hook();
        commit_and_retire(&clock, &hook, 1);
        clock.commit_epoch();
        assert!(hook.abandon(1, 3).is_err(), "past the clock");
        assert!(hook.abandon(1, 1).is_err(), "short of the clock");
        assert!(hook.abandon(3, 2).is_err(), "inverted");
        hook.abandon(1, 2).unwrap();
        assert!(hook.abandonment_settled());
        assert!(hook.abandon(2, 2).is_err(), "published twice");
        hook.complete(&clean());
        assert_eq!(clock.completed_count().0, 2);
        assert!(
            failure.check().is_err(),
            "a run past the settle was not latched"
        );
    }

    /// The final owner's publication and the last record's retire race: in
    /// every interleaving the epochs are marked exactly once, the run count
    /// lands on the reserved epochs and nothing latches.
    #[test]
    fn a_racing_publication_and_retire_settle_exactly_once() {
        for _ in 0..2000 {
            let (clock, failure, hook) = hook();
            let hook = Arc::new(hook);
            commit_and_retire(&clock, &hook, 1);
            clock.commit_epoch();
            clock.commit_epoch();
            hook.complete(&clean());
            let start = Arc::new(std::sync::Barrier::new(2));
            let retiring = {
                let (hook, start) = (hook.clone(), start.clone());
                std::thread::spawn(move || {
                    start.wait();
                    hook.retire(1);
                })
            };
            start.wait();
            hook.abandon(2, 3).unwrap();
            retiring.join().unwrap();
            assert!(hook.abandonment_settled());
            assert_eq!(clock.completed_count().0, 3);
            assert_eq!(hook.runs.load(Ordering::Relaxed), 3);
            assert!(failure.check().is_ok());
        }
    }
}
