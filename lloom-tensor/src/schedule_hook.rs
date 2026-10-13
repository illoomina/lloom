//! Schedule points: the named transitions of the host completion/retirement
//! protocols where a test scheduler may observe a thread or switch away from
//! it (`docs/invariants.md` §15.1l).
//!
//! Each point is written where the transition already is, as
//! `schedule_point!(Site)`. Without the `schedule-hooks` feature the macro
//! expands to an empty block, so a build without it carries no call, load or
//! branch.
//! With it, a point loads one installed function pointer and calls it with
//! the site when one is installed; nothing is installed by default. The
//! feature is enabled only from `lloom-fuzz` (its own workspace) and from
//! test builds that ask for it, so a release build cannot acquire it through
//! feature unification.
//!
//! No point runs inside a lock its own function holds, or inside a parking
//! queue's validation callback. The `Wake` points run on behalf of every
//! caller of `Wake`, and a caller may hold its own lock across the call or be
//! a Metal completion thread: a hook that blocks there blocks that caller. A
//! hook must synchronise its own state: the points add no ordering between
//! the threads that reach them.

/// One transition. The discriminant is the site id a recorded schedule
/// carries; [`SCHEMA`] changes whenever the list does.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Site {
    /// A waiter read the wake epoch; its predicate check follows.
    WakeSnapshot = 1,
    /// A notifier published the next epoch; it reads the waiter count next.
    WakeNotified = 2,
    /// A waiter counted itself; it has not yet entered the parking queue.
    WakeRegistered = 3,
    /// A waiter left the parking queue: woken, invalidated or timed out.
    WakeParkReturned = 4,
    /// A residency guard released its lock; the service wake follows.
    GuardUnlocked = 5,
    /// The retirement worker finished an inbox pass; it may park next.
    RetirePassed = 6,
    /// The retirement worker published a failure after its loop unwound.
    RetireFailed = 7,
    /// A listener block was registered on a shared event; its waiter locks next.
    ListenerRegistered = 8,
    /// A listener block published its value and unlocked; it notifies next.
    ListenerFired = 9,
    /// A completion wait found its buffer unfinished and is choosing a route.
    CompletionWaiting = 10,
    /// A completion wait's event route timed out; the blocking wait follows.
    CompletionFallback = 11,
}

/// Every site, in id order.
pub const SITES: [Site; 11] = [
    Site::WakeSnapshot,
    Site::WakeNotified,
    Site::WakeRegistered,
    Site::WakeParkReturned,
    Site::GuardUnlocked,
    Site::RetirePassed,
    Site::RetireFailed,
    Site::ListenerRegistered,
    Site::ListenerFired,
    Site::CompletionWaiting,
    Site::CompletionFallback,
];

/// Raise when a point moves without the list changing: a recorded schedule
/// names transitions by id, and a moved point is a different transition.
const PLACEMENT_REVISION: u64 = 1;

/// The schema a recorded schedule must match: a digest of every site's id and
/// name, and of [`PLACEMENT_REVISION`].
pub const SCHEMA: u64 = schema();

const fn schema() -> u64 {
    const NAMES: [&str; 11] = [
        "WakeSnapshot",
        "WakeNotified",
        "WakeRegistered",
        "WakeParkReturned",
        "GuardUnlocked",
        "RetirePassed",
        "RetireFailed",
        "ListenerRegistered",
        "ListenerFired",
        "CompletionWaiting",
        "CompletionFallback",
    ];
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ PLACEMENT_REVISION;
    let mut i = 0;
    while i < SITES.len() {
        hash = (hash ^ SITES[i] as u64).wrapping_mul(0x0100_0000_01b3);
        let name = NAMES[i].as_bytes();
        let mut j = 0;
        while j < name.len() {
            hash = (hash ^ name[j] as u64).wrapping_mul(0x0100_0000_01b3);
            j += 1;
        }
        i += 1;
    }
    hash
}

/// A schedule point; nothing without the `schedule-hooks` feature.
#[cfg(not(feature = "schedule-hooks"))]
#[allow(unused_macros)]
macro_rules! schedule_point {
    ($site:ident) => {{}};
}

/// A schedule point: calls the installed hook, if any, with the site.
#[cfg(feature = "schedule-hooks")]
#[allow(unused_macros)]
macro_rules! schedule_point {
    ($site:ident) => {
        $crate::schedule_hook::point($crate::schedule_hook::Site::$site)
    };
}

#[allow(unused_imports)]
pub(crate) use schedule_point;

#[cfg(feature = "schedule-hooks")]
pub use hook::*;

#[cfg(feature = "schedule-hooks")]
mod hook {
    use super::Site;
    use std::sync::atomic::{AtomicPtr, Ordering};

    /// What a scheduler runs at each point.
    pub type Hook = fn(Site);

    static HOOK: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

    /// Install `hook`, or remove the installed one with `None`; returns the
    /// previous. A thread already inside the previous hook finishes it.
    pub fn install(hook: Option<Hook>) -> Option<Hook> {
        let new = hook.map_or(std::ptr::null_mut(), |h| h as *mut ());
        let old = HOOK.swap(new, Ordering::AcqRel);
        // SAFETY: every non-null value stored is a `Hook` cast above.
        (!old.is_null()).then(|| unsafe { std::mem::transmute::<*mut (), Hook>(old) })
    }

    /// The points' target.
    #[inline]
    pub fn point(site: Site) {
        let hook = HOOK.load(Ordering::Acquire);
        if !hook.is_null() {
            // SAFETY: every non-null value stored is a `Hook` cast in `install`.
            let hook = unsafe { std::mem::transmute::<*mut (), Hook>(hook) };
            hook(site);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_ids_are_dense_from_one_and_fit_a_schedule_trace() {
        for (i, site) in SITES.iter().enumerate() {
            assert_eq!(*site as usize, i + 1, "{site:?}");
        }
        assert!(
            SITES.len() < 64,
            "a recorded schedule declares sites 1..=63"
        );
    }

    /// The digest of today's list. A failure here means the list or a point's
    /// placement changed: recorded schedules from before are incompatible.
    #[test]
    fn schema_is_pinned() {
        assert_eq!(SCHEMA, 0x0575_0e05_bb55_030d, "{SCHEMA:#018x}");
    }

    /// One uncontended pass through four points: a snapshot, a notification,
    /// and a residency guard's release with its notification.
    pub(super) fn round_trip(
        wake: &crate::completion_wake::Wake,
        guarded: &crate::completion_wake::WakeMutex<u64>,
    ) -> usize {
        let seen = wake.snapshot();
        wake.notify();
        *guarded.lock() += 1;
        seen
    }

    /// The points' cost with no hook installed, against the same loop built
    /// without the feature. A timing arm, run alone in a release build.
    #[test]
    #[ignore = "timing arm: run alone, release, under lloom-arb measure"]
    fn round_trip_cost() {
        use std::sync::Arc;
        const ITERS: u64 = 20_000_000;
        let wake = Arc::new(crate::completion_wake::Wake::default());
        let guarded = crate::completion_wake::WakeMutex::new(0u64);
        assert!(guarded.install_wake(wake.clone()).is_ok());
        // The instrument's positive control: a no-op hook makes every point an
        // indirect call, a cost the timing must resolve.
        #[cfg(feature = "schedule-hooks")]
        let installed = std::env::var_os("LLOOM_SCHEDULE_BENCH_HOOK").is_some()
            && super::install(Some(|_| {})).is_none();
        #[cfg(not(feature = "schedule-hooks"))]
        let installed = false;
        let mut sink = 0usize;
        for _ in 0..ITERS / 100 {
            sink = sink.wrapping_add(round_trip(&wake, &guarded));
        }
        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            sink = sink.wrapping_add(round_trip(std::hint::black_box(&wake), &guarded));
        }
        let elapsed = started.elapsed();
        std::hint::black_box(sink);
        println!(
            "ROUND_TRIP hooks={} iters={ITERS} ns_per_iter={:.4}",
            match (cfg!(feature = "schedule-hooks"), installed) {
                (_, true) => "installed",
                (true, false) => "compiled",
                (false, false) => "absent",
            },
            elapsed.as_nanos() as f64 / ITERS as f64
        );
    }
}

#[cfg(all(test, feature = "schedule-hooks"))]
mod hooked {
    use super::*;
    use crate::completion_wake::{Wake, WakeMutex};
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, LazyLock};
    use std::time::{Duration, Instant};

    thread_local! {
        static LOG: RefCell<Vec<(Site, bool)>> = const { RefCell::new(Vec::new()) };
    }
    static GUARDED: LazyLock<WakeMutex<()>> = LazyLock::new(|| WakeMutex::new(()));
    /// The hook is process-wide; these tests take turns installing it.
    static TURN: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    static WORKER: [AtomicUsize; SITES.len() + 1] =
        [const { AtomicUsize::new(0) }; SITES.len() + 1];

    /// Logs this thread's points; at a guard release, also whether the guard's
    /// lock is free at that instant. Counts the retirement worker's points.
    fn record(site: Site) {
        if std::thread::current().name() == Some("lloom-metal-retire") {
            WORKER[site as usize].fetch_add(1, Ordering::Relaxed);
            return;
        }
        let free = site == Site::GuardUnlocked && !GUARDED.test_is_locked();
        LOG.with(|log| log.borrow_mut().push((site, free)));
    }

    fn take() -> Vec<(Site, bool)> {
        LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }

    /// One wake round trip, in protocol order. Parking-queue validation must
    /// reach no point: it runs under the queue lock, where a hook may not block.
    #[test]
    fn wake_points_fire_in_protocol_order_and_never_under_the_queue_lock() {
        let _turn = TURN.lock();
        install(Some(record));
        take();
        let wake = Wake::default();
        let seen = wake.snapshot();
        wake.notify();
        let parked = wake.park(seen, Some(Instant::now() + Duration::from_secs(1)));
        let log = take();
        install(None);
        assert_eq!(parked, parking_lot_core::ParkResult::Invalid);
        assert_eq!(
            log.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            [
                Site::WakeSnapshot,
                Site::WakeNotified,
                Site::WakeRegistered,
                Site::WakeParkReturned
            ]
        );
    }

    /// A residency guard's release reaches its point with the lock already
    /// free, and the service wake after it.
    #[test]
    fn guard_point_fires_after_unlock_and_before_the_wake() {
        let _turn = TURN.lock();
        install(Some(record));
        let wake = Arc::new(Wake::default());
        assert!(GUARDED.install_wake(wake.clone()).is_ok());
        let held = GUARDED.lock();
        take();
        drop(held);
        let log = take();
        install(None);
        assert_eq!(
            log,
            [(Site::GuardUnlocked, true), (Site::WakeNotified, false)]
        );
    }

    /// The timing arm's loop reaches exactly four points an iteration, so its
    /// cost is the points' cost and not a loop that skipped them.
    #[test]
    fn round_trip_reaches_four_points_an_iteration() {
        let _turn = TURN.lock();
        let wake = Arc::new(Wake::default());
        let guarded = WakeMutex::new(0u64);
        assert!(guarded.install_wake(wake.clone()).is_ok());
        install(Some(record));
        take();
        for _ in 0..1000 {
            super::tests::round_trip(&wake, &guarded);
        }
        let log = take();
        install(None);
        assert_eq!(log.len(), 4000);
        assert_eq!(
            log[..4].iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            [
                Site::WakeSnapshot,
                Site::WakeNotified,
                Site::GuardUnlocked,
                Site::WakeNotified
            ]
        );
    }

    #[test]
    fn no_point_reaches_a_removed_hook() {
        let _turn = TURN.lock();
        install(Some(record));
        install(None);
        take();
        let wake = Wake::default();
        wake.notify();
        let _ = wake.snapshot();
        assert!(take().is_empty());
    }

    /// The shared retirement worker reaches its points through its own loop:
    /// a snapshot, a pass, a park, and a return once notified.
    #[test]
    fn retirement_worker_reaches_its_points() {
        let _turn = TURN.lock();
        install(Some(record));
        let service = crate::retirement_service::HostRetirementService::shared().unwrap();
        let count = |site: Site| WORKER[site as usize].load(Ordering::Relaxed);
        // Notify until the worker has parked, been woken and passed again:
        // it may have parked before this hook was installed, or not started.
        let deadline = Instant::now() + Duration::from_secs(10);
        while [
            Site::RetirePassed,
            Site::WakeRegistered,
            Site::WakeParkReturned,
        ]
        .iter()
        .any(|s| count(*s) == 0)
        {
            assert!(Instant::now() < deadline, "the worker never cycled");
            service.wake().notify();
            std::thread::sleep(Duration::from_millis(1));
        }
        install(None);
        let counts: Vec<_> = SITES.iter().map(|s| (*s, count(*s))).collect();
        println!("WORKER_POINTS {counts:?}");
        for site in [
            Site::WakeSnapshot,
            Site::RetirePassed,
            Site::WakeRegistered,
            Site::WakeParkReturned,
        ] {
            assert!(count(site) >= 1, "{site:?} not reached");
        }
        assert_eq!(count(Site::RetireFailed), 0);
    }
}
