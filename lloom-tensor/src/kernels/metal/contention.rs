//! Contended acquisitions of the command-submission locks, counted per lock.
//!
//! The cross-encoder fence map is read by the encoding thread once per buffer
//! per session; the map, the fence book beside it and the encoder-state lock
//! are written at encoder closes and per dispatch. Whether any acquisition
//! waits is a measurement, not an assumption (`DESIGN.md` §6.4d): under CPU
//! oversubscription a completion thread descheduled while holding the map made
//! the encoding thread wait milliseconds, which is why completion threads no
//! longer take these locks and why the census stays to show they do not.
//!
//! [`Censused`] wraps a `parking_lot` lock. An acquisition tries without
//! blocking first and only on failure blocks, timed; the uncontended path is
//! the lock's own fast path and records nothing, so the census is always on
//! and costs nothing where there is nothing to count. Counters live with the
//! lock they describe, per side and access, so a reader attributes contention
//! to one lock of one `Commands` rather than to a process-wide total.

use crate::kernels::metal::cache_pad::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The thread an acquisition was made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The thread encoding and committing work.
    Host,
    /// A Metal completion-handler thread.
    Driver,
}

/// Shared (a read lock) or exclusive (a write lock or a mutex).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Shared,
    Exclusive,
}

const SLOTS: usize = 4;

#[inline]
fn slot(side: Side, access: Access) -> usize {
    (side as usize) * 2 + access as usize
}

/// Acquisitions that found the lock unavailable, and the time they spent
/// blocked on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Contention {
    pub waits: u64,
    pub blocked_ns: u64,
}

impl Contention {
    /// The change from `earlier` to `self`, for a window between two reads.
    pub fn since(self, earlier: Contention) -> Contention {
        Contention {
            waits: self.waits.saturating_sub(earlier.waits),
            blocked_ns: self.blocked_ns.saturating_sub(earlier.blocked_ns),
        }
    }
}

#[derive(Default)]
struct Slot {
    waits: AtomicU64,
    blocked_ns: AtomicU64,
}

/// A lock and the census of its contended acquisitions.
///
/// The counters sit on their own cache line so that a contended acquisition
/// recording its wait does not invalidate the line an uncontended one on the
/// other thread is spinning on.
pub struct Censused<L> {
    lock: L,
    census: CachePadded<[Slot; SLOTS]>,
}

impl<L> Censused<L> {
    pub fn new(lock: L) -> Self {
        Self {
            lock,
            census: CachePadded::new(Default::default()),
        }
    }

    /// Contended acquisitions from `side` with `access` since construction.
    pub fn contention(&self, side: Side, access: Access) -> Contention {
        let s = &self.census[slot(side, access)];
        Contention {
            waits: s.waits.load(Ordering::Relaxed),
            blocked_ns: s.blocked_ns.load(Ordering::Relaxed),
        }
    }

    #[cold]
    fn record(&self, side: Side, access: Access, since: Instant) {
        let s = &self.census[slot(side, access)];
        s.waits.fetch_add(1, Ordering::Relaxed);
        s.blocked_ns
            .fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

impl<T> Censused<parking_lot::RwLock<T>> {
    #[inline]
    pub fn read(&self, side: Side) -> parking_lot::RwLockReadGuard<'_, T> {
        if let Some(g) = self.lock.try_read() {
            return g;
        }
        let t = Instant::now();
        let g = self.lock.read();
        self.record(side, Access::Shared, t);
        g
    }

    #[inline]
    pub fn write(&self, side: Side) -> parking_lot::RwLockWriteGuard<'_, T> {
        if let Some(g) = self.lock.try_write() {
            return g;
        }
        let t = Instant::now();
        let g = self.lock.write();
        self.record(side, Access::Exclusive, t);
        g
    }
}

impl<T> Censused<parking_lot::Mutex<T>> {
    #[inline]
    pub fn lock(&self, side: Side) -> parking_lot::MutexGuard<'_, T> {
        if let Some(g) = self.lock.try_lock() {
            return g;
        }
        let t = Instant::now();
        let g = self.lock.lock();
        self.record(side, Access::Exclusive, t);
        g
    }
}

/// The submission locks' contention for one `Commands`
/// ([`crate::kernels::metal::Commands::submission_contention`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmissionContention {
    /// The fence map, read by the encoding thread once per buffer per session.
    pub map_host_read: Contention,
    /// The fence map, written by the encoding thread at each encoder close.
    pub map_host_write: Contention,
    /// The fence book (live fences, retiring sessions), from the encoding
    /// thread.
    pub live_fences_host: Contention,
    /// The encoder-state lock, from the encoding thread.
    pub state_host: Contention,
}

impl SubmissionContention {
    /// Per-field change from `earlier` to `self`.
    pub fn since(&self, earlier: &SubmissionContention) -> SubmissionContention {
        SubmissionContention {
            map_host_read: self.map_host_read.since(earlier.map_host_read),
            map_host_write: self.map_host_write.since(earlier.map_host_write),
            live_fences_host: self.live_fences_host.since(earlier.live_fences_host),
            state_host: self.state_host.since(earlier.state_host),
        }
    }

    /// Waits summed over every lock and side.
    pub fn total_waits(&self) -> u64 {
        self.fields().iter().map(|(_, c)| c.waits).sum()
    }

    /// Blocked time summed over every field: every submission lock is the
    /// encoding thread's, so this is the time the host spent waiting on one.
    pub fn host_blocked_ns(&self) -> u64 {
        self.fields().iter().map(|(_, c)| c.blocked_ns).sum()
    }

    /// `name=waits/blocked_us` for each field, space-separated.
    pub fn render(&self) -> String {
        self.fields()
            .iter()
            .map(|(name, c)| format!("{name}={}/{:.1}us", c.waits, c.blocked_ns as f64 / 1e3))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn fields(&self) -> [(&'static str, Contention); 4] {
        [
            ("map_host_read", self.map_host_read),
            ("map_host_write", self.map_host_write),
            ("live_fences_host", self.live_fences_host),
            ("state_host", self.state_host),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// Long enough that a test thread descheduled under load still arrives
    /// while the holder has the lock.
    const HOLD: Duration = Duration::from_millis(100);

    /// Run `acquire` (take the lock, sleep `HOLD`, release) on another thread.
    fn hold_elsewhere<L: Send + Sync + 'static>(
        lock: &Arc<L>,
        acquire: fn(&L),
    ) -> std::thread::JoinHandle<()> {
        let lock = Arc::clone(lock);
        std::thread::spawn(move || acquire(&lock))
    }

    /// Spin until the holder has observably taken the lock.
    fn wait_until(mut taken: impl FnMut() -> bool) {
        let start = Instant::now();
        while !taken() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the holder never took the lock"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn an_uncontended_acquisition_records_nothing() {
        let rw = Censused::new(parking_lot::RwLock::new(0u32));
        let m = Censused::new(parking_lot::Mutex::new(0u32));
        for _ in 0..1000 {
            drop(rw.read(Side::Host));
            drop(rw.write(Side::Driver));
            drop(m.lock(Side::Host));
        }
        for side in [Side::Host, Side::Driver] {
            for access in [Access::Shared, Access::Exclusive] {
                assert_eq!(rw.contention(side, access), Contention::default());
                assert_eq!(m.contention(side, access), Contention::default());
            }
        }
    }

    /// A read made while another thread holds the write lock is counted once,
    /// on the side and access it was made with, and its blocked time covers
    /// the rest of the hold.
    #[test]
    fn a_read_behind_a_held_write_is_counted_once_with_its_wait() {
        let rw = Arc::new(Censused::new(parking_lot::RwLock::new(0u32)));
        let holder = hold_elsewhere(&rw, |l| {
            let _g = l.lock.write();
            std::thread::sleep(HOLD);
        });
        wait_until(|| rw.lock.is_locked_exclusive());
        let t = Instant::now();
        drop(rw.read(Side::Host));
        let waited = t.elapsed();
        holder.join().unwrap();

        let c = rw.contention(Side::Host, Access::Shared);
        assert_eq!(c.waits, 1, "one blocked read");
        assert!(
            c.blocked_ns > 0 && c.blocked_ns <= waited.as_nanos() as u64,
            "the recorded wait {} ns must be positive and within the measured {} ns",
            c.blocked_ns,
            waited.as_nanos()
        );
        assert!(
            waited >= HOLD / 4,
            "the read returned after {waited:?}: it did not wait for the holder"
        );
        assert_eq!(
            rw.contention(Side::Driver, Access::Shared),
            Contention::default()
        );
        assert_eq!(
            rw.contention(Side::Host, Access::Exclusive),
            Contention::default()
        );
        assert_eq!(
            rw.contention(Side::Driver, Access::Exclusive),
            Contention::default()
        );
    }

    /// A write made while another thread holds a read lock is counted on its
    /// own slot -- the completion handler's side of the fence map.
    #[test]
    fn a_write_behind_a_held_read_is_counted_on_the_writers_side() {
        let rw = Arc::new(Censused::new(parking_lot::RwLock::new(0u32)));
        let holder = hold_elsewhere(&rw, |l| {
            let _g = l.lock.read();
            std::thread::sleep(HOLD);
        });
        wait_until(|| rw.lock.is_locked() && !rw.lock.is_locked_exclusive());
        drop(rw.write(Side::Driver));
        holder.join().unwrap();

        assert_eq!(rw.contention(Side::Driver, Access::Exclusive).waits, 1);
        assert!(rw.contention(Side::Driver, Access::Exclusive).blocked_ns > 0);
        assert_eq!(rw.contention(Side::Host, Access::Shared).waits, 0);
        assert_eq!(rw.contention(Side::Host, Access::Exclusive).waits, 0);
    }

    #[test]
    fn a_mutex_behind_a_holder_is_counted_once() {
        let m = Arc::new(Censused::new(parking_lot::Mutex::new(0u32)));
        let holder = hold_elsewhere(&m, |l| {
            let _g = l.lock.lock();
            std::thread::sleep(HOLD);
        });
        wait_until(|| m.lock.is_locked());
        drop(m.lock(Side::Host));
        holder.join().unwrap();

        assert_eq!(m.contention(Side::Host, Access::Exclusive).waits, 1);
        assert!(m.contention(Side::Host, Access::Exclusive).blocked_ns > 0);
        assert_eq!(m.contention(Side::Driver, Access::Exclusive).waits, 0);
    }

    /// The price of one uncontended acquire-and-release, the raw lock against
    /// the census wrapper -- the number an acquisition count multiplies into a
    /// per-token cost (`DESIGN.md` §6.4d; §6.4a's counts x prices). A timing,
    /// so ignored by default and asserted on nothing: run it alone, in
    /// release, under `lloom-arb measure`:
    /// `cargo test --release -p lloom-tensor --features metal --lib
    /// price_of_an_uncontended_acquisition -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn price_of_an_uncontended_acquisition() {
        use std::hint::black_box;
        const OPS: u32 = 20_000_000;
        const REPEATS: usize = 7;
        fn ns_per_op(mut f: impl FnMut()) -> f64 {
            let t = Instant::now();
            for _ in 0..OPS {
                f();
            }
            t.elapsed().as_nanos() as f64 / OPS as f64
        }
        let m = Censused::new(parking_lot::Mutex::new(0u64));
        let rw = Censused::new(parking_lot::RwLock::new(0u64));
        let arms: [(&str, &dyn Fn()); 6] = [
            ("mutex lock, raw", &|| *black_box(&m.lock).lock() += 1),
            ("mutex lock, censused", &|| {
                *black_box(&m).lock(Side::Host) += 1
            }),
            ("rwlock read, raw", &|| {
                black_box(*black_box(&rw.lock).read());
            }),
            ("rwlock read, censused", &|| {
                black_box(*black_box(&rw).read(Side::Host));
            }),
            ("rwlock write, raw", &|| *black_box(&rw.lock).write() += 1),
            ("rwlock write, censused", &|| {
                *black_box(&rw).write(Side::Host) += 1
            }),
        ];
        println!("# {OPS} acquisitions per repeat, {REPEATS} repeats, arms interleaved per repeat");
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for _ in 0..REPEATS {
            for (i, (_, f)) in arms.iter().enumerate() {
                samples[i].push(ns_per_op(f));
            }
        }
        for (i, (name, _)) in arms.iter().enumerate() {
            let mut v = samples[i].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{name:<24} ns/op min {:.3} median {:.3} max {:.3}",
                v[0],
                v[REPEATS / 2],
                v[REPEATS - 1]
            );
        }
        for side in [Side::Host, Side::Driver] {
            for access in [Access::Shared, Access::Exclusive] {
                assert_eq!(
                    m.contention(side, access).waits + rw.contention(side, access).waits,
                    0,
                    "a single thread cannot contend"
                );
            }
        }
    }

    #[test]
    fn a_window_is_the_difference_of_two_reads() {
        let a = SubmissionContention {
            map_host_read: Contention {
                waits: 3,
                blocked_ns: 1_000,
            },
            state_host: Contention {
                waits: 1,
                blocked_ns: 50,
            },
            ..Default::default()
        };
        let b = SubmissionContention {
            map_host_read: Contention {
                waits: 5,
                blocked_ns: 4_000,
            },
            live_fences_host: Contention {
                waits: 2,
                blocked_ns: 700,
            },
            state_host: Contention {
                waits: 1,
                blocked_ns: 50,
            },
            ..Default::default()
        };
        let d = b.since(&a);
        assert_eq!(
            d.map_host_read,
            Contention {
                waits: 2,
                blocked_ns: 3_000
            }
        );
        assert_eq!(d.live_fences_host.waits, 2);
        assert_eq!(d.state_host, Contention::default());
        assert_eq!(d.total_waits(), 4);
        assert_eq!(
            d.host_blocked_ns(),
            3_700,
            "every field's blocked time, the unchanged one contributing none"
        );
        assert!(d.render().contains("map_host_read=2/3.0us"));
    }
}
