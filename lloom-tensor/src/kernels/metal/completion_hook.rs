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

    pub(super) fn subscribe(&self, f: Subscriber) {
        self.subscribers.subscribe(f);
    }

    /// Hold the subscribers' lock on the calling thread while `f` runs, so a
    /// test can show a commit and a completion never take it.
    #[cfg(test)]
    pub(super) fn while_subscribers_locked<R>(&self, f: impl FnOnce() -> R) -> R {
        let _published = self.subscribers.published.lock();
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
struct Subscribers {
    current: AtomicPtr<Snapshot>,
    published: parking_lot::Mutex<Vec<Box<Snapshot>>>,
}

impl Subscribers {
    fn new() -> Self {
        let empty = Box::new(Snapshot::new());
        Self {
            current: AtomicPtr::new(&*empty as *const Snapshot as *mut Snapshot),
            published: parking_lot::Mutex::new(vec![empty]),
        }
    }

    fn subscribe(&self, f: Subscriber) {
        let mut published = self.published.lock();
        let mut next = Snapshot::clone(published.last().expect("the empty snapshot"));
        next.push(f);
        let next = Box::new(next);
        self.current.store(
            &*next as *const Snapshot as *mut Snapshot,
            Ordering::Release,
        );
        published.push(next);
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

pub(super) struct SubscriberTake<'a> {
    published: parking_lot::MutexGuard<'a, Vec<Box<Snapshot>>>,
    current: &'a AtomicPtr<Snapshot>,
}

impl SubscriberTake<'_> {
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

    /// A subscriber runs on every completion after its registration, the
    /// teardown take leaves none to run, and every subscriber is handed over
    /// exactly once.
    #[test]
    fn subscribers_run_from_registration_and_teardown_takes_them_all() {
        let (clock, _, hook) = hook();
        let runs = Arc::new(AtomicUsize::new(0));
        for _ in 0..4 {
            clock.commit_epoch();
        }
        hook.complete(&clean());
        assert_eq!(runs.load(Ordering::Relaxed), 0);
        for _ in 0..2 {
            let r = Arc::clone(&runs);
            hook.subscribe(Arc::new(move || {
                r.fetch_add(1, Ordering::Relaxed);
            }));
        }
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
