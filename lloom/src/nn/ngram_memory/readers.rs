//! A fixed set of threads that read a request's cold rows by `pread`, for
//! [`super::MmapRows`] under [`super::ReadPool::Workers`]. Each thread is
//! handed its rows through a job of its own, every row of a request is counted
//! on one `done` counter, and the thread whose row completes the request wakes
//! its caller. The threads park between jobs and never spin: a spinning core
//! moves the GPU's clock (`docs/expert-streaming.md` §20.14a). A request
//! allocates nothing and takes no lock.
//!
//! A row still in flight a set time after its request was handed out may be
//! read again on an idle thread (the hedge). Each row is claimed once, by the
//! first read to land, under the request's generation, so a read that loses
//! the race — or lands after its request returned — writes nothing its caller
//! reads, and the caller never waits for it.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::thread::Thread;
use std::time::{Duration, Instant};

/// Rows one request reads at most: a token's 16, a batch of 16 rows'. A
/// larger request is refused and read on rayon's pool, counted
/// ([`ReaderStats::refused`]).
pub const MAX_ROWS: usize = 256;
/// Bytes one row may take (an IQ4_NL row of 160 is 90; f32 of 160 is 640):
/// a wider table's request is refused as [`MAX_ROWS`] says.
pub const MAX_ROW_BYTES: usize = 1024;
/// Threads the set holds at most: a configured count past it is clamped, and
/// the set reports the count it runs ([`ReaderStats::threads`]).
pub const MAX_THREADS: usize = 64;

/// The longest a caller parks before it looks at its request again. The read
/// that completes the request wakes it; this only bounds a lost wake.
const PARK_BOUND: Duration = Duration::from_millis(10);

/// A row's claim: the request's generation, and whether the row is still in
/// flight, landed, or failed.
const PENDING: u64 = 0;
const LANDED: u64 = 1;
const FAILED: u64 = 2;

fn claim(round: u64, state: u64) -> u64 {
    (round << 2) | state
}

/// A value on a cache line of its own (128 bytes on Apple silicon).
#[repr(align(128))]
#[derive(Default)]
struct Line<T>(T);

#[derive(Clone, Copy, Default)]
struct Item {
    /// The row's place in the request (and in its destination).
    row: u32,
    /// Its offset in the file.
    off: u64,
}

/// One thread's job. The caller writes it only while the thread is idle
/// (`finished == seq`) and hands it over by advancing `seq`.
struct Job {
    round: u64,
    fd: i32,
    row_bytes: usize,
    /// Rows in the whole request: the `done` count that completes it.
    rows: usize,
    dst: *mut u8,
    hedge: bool,
    count: usize,
    items: [Item; MAX_ROWS],
    caller: Option<Thread>,
}

struct Worker {
    /// Advanced by the caller to hand the job over.
    seq: Line<AtomicU64>,
    /// The `seq` this thread last finished, its wake of the caller included.
    finished: Line<AtomicU64>,
    job: UnsafeCell<Job>,
    /// The thread's own landing buffer: a read that loses its row's claim
    /// writes only here.
    buf: UnsafeCell<[u8; MAX_ROW_BYTES]>,
    thread: OnceLock<Thread>,
}

/// What every request's rows are claimed and counted on.
struct Request {
    claims: [AtomicU64; MAX_ROWS],
    done: Line<AtomicU32>,
    errno: AtomicI32,
}

#[derive(Default)]
struct Counters {
    requests: AtomicU64,
    rows: AtomicU64,
    hedged: AtomicU64,
    hedge_wins: AtomicU64,
    refused: AtomicU64,
}

/// The set's counts since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReaderStats {
    pub threads: usize,
    /// Requests the set read.
    pub requests: u64,
    /// Rows those requests read.
    pub rows: u64,
    /// Rows read a second time by the hedge.
    pub hedged: u64,
    /// Rows whose hedged read landed first.
    pub hedge_wins: u64,
    /// Requests the set could not take (held by another caller, no thread
    /// idle, too large): read on the rayon pool instead.
    pub refused: u64,
}

/// The process's read threads ([`readers`]).
pub struct Readers {
    workers: Box<[Worker]>,
    req: Request,
    /// Held by the one caller whose request is in flight; another caller is
    /// refused and reads another way.
    busy: AtomicBool,
    round: AtomicU64,
    counters: Counters,
}

// SAFETY: a `Job` and a landing buffer are touched by one thread at a time —
// the caller while its worker is idle, the worker between `seq` moving and its
// `finished` store, both ordered by those two words — and the destination
// pointer is written only under a row's claim, while the caller that owns the
// destination waits for that row to be counted.
unsafe impl Sync for Readers {}
// SAFETY: as above; nothing in a `Readers` is tied to the thread that made it.
unsafe impl Send for Readers {}

static SET: OnceLock<Option<&'static Readers>> = OnceLock::new();

/// The process's set, started on first use with `threads` threads (later
/// calls get the first set whatever they ask); `None` where a thread could
/// not be spawned.
pub fn readers(threads: usize) -> Option<&'static Readers> {
    *SET.get_or_init(|| start(threads.clamp(1, MAX_THREADS)))
}

/// The set, if one has started.
pub fn started() -> Option<&'static Readers> {
    SET.get().copied().flatten()
}

fn start(threads: usize) -> Option<&'static Readers> {
    let workers: Box<[Worker]> = (0..threads)
        .map(|_| Worker {
            seq: Line::default(),
            finished: Line::default(),
            job: UnsafeCell::new(Job {
                round: 0,
                fd: -1,
                row_bytes: 0,
                rows: 0,
                dst: std::ptr::null_mut(),
                hedge: false,
                count: 0,
                items: [Item::default(); MAX_ROWS],
                caller: None,
            }),
            buf: UnsafeCell::new([0; MAX_ROW_BYTES]),
            thread: OnceLock::new(),
        })
        .collect();
    let set: &'static Readers = Box::leak(Box::new(Readers {
        workers,
        req: Request {
            claims: std::array::from_fn(|_| AtomicU64::new(0)),
            done: Line::default(),
            errno: AtomicI32::new(0),
        },
        busy: AtomicBool::new(false),
        round: AtomicU64::new(0),
        counters: Counters::default(),
    }));
    for w in 0..threads {
        std::thread::Builder::new()
            .name(format!("ngram-row-{w}"))
            .spawn(move || set.run(w))
            .ok()?;
    }
    Some(set)
}

impl Readers {
    /// Threads in the set.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    pub fn stats(&self) -> ReaderStats {
        let c = &self.counters;
        ReaderStats {
            threads: self.workers.len(),
            requests: c.requests.load(Ordering::Relaxed),
            rows: c.rows.load(Ordering::Relaxed),
            hedged: c.hedged.load(Ordering::Relaxed),
            hedge_wins: c.hedge_wins.load(Ordering::Relaxed),
            refused: c.refused.load(Ordering::Relaxed),
        }
    }

    /// Read `rows` rows of `row_bytes` from `fd` — row `i` at `offset(i)` —
    /// into `dst` (`rows · row_bytes`), concurrently; a row still in flight
    /// `hedge` after the request is read again on an idle thread. `None`
    /// when the set cannot take the request: another caller's request is in
    /// flight, no thread is idle, or it does not fit.
    pub fn read(
        &self,
        fd: i32,
        rows: usize,
        row_bytes: usize,
        offset: impl Fn(usize) -> u64,
        dst: &mut [u8],
        hedge: Option<Duration>,
    ) -> Option<std::io::Result<()>> {
        if rows == 0 {
            return Some(Ok(()));
        }
        let fits = rows <= MAX_ROWS && row_bytes <= MAX_ROW_BYTES && dst.len() == rows * row_bytes;
        let out = if fits
            && self
                .busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            let out = self.read_held(fd, rows, row_bytes, &offset, dst, hedge);
            self.busy.store(false, Ordering::Release);
            out
        } else {
            None
        };
        if out.is_none() {
            self.counters.refused.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn read_held(
        &self,
        fd: i32,
        rows: usize,
        row_bytes: usize,
        offset: &impl Fn(usize) -> u64,
        dst: &mut [u8],
        hedge: Option<Duration>,
    ) -> Option<std::io::Result<()>> {
        let mut idle = [0u8; MAX_THREADS];
        let n_idle = self.idle(&mut idle);
        if n_idle == 0 {
            return None;
        }
        let round = self.round.fetch_add(1, Ordering::Relaxed) + 1;
        for c in &self.req.claims[..rows] {
            c.store(claim(round, PENDING), Ordering::Relaxed);
        }
        self.req.done.0.store(0, Ordering::Relaxed);
        self.req.errno.store(0, Ordering::Relaxed);
        let me = std::thread::current();
        let dst = dst.as_mut_ptr();
        let k = n_idle.min(rows);
        for (j, &w) in idle[..k].iter().enumerate() {
            let items = (j..rows).step_by(k).map(|i| Item {
                row: i as u32,
                off: offset(i),
            });
            self.hand(
                w as usize, &me, round, fd, rows, row_bytes, dst, false, items,
            );
        }
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        self.counters.rows.fetch_add(rows as u64, Ordering::Relaxed);
        // The clock is read only for a hedge, whose deadline it is.
        let mut deadline = hedge.map(|h| (Instant::now(), h));
        while (self.req.done.0.load(Ordering::Acquire) as usize) < rows {
            match deadline {
                Some((t0, h)) => {
                    let left = h.saturating_sub(t0.elapsed());
                    if left.is_zero() {
                        self.hedge(round, &me, fd, rows, row_bytes, dst, offset);
                        deadline = None;
                    } else {
                        std::thread::park_timeout(left);
                    }
                }
                None => std::thread::park_timeout(PARK_BOUND),
            }
        }
        let e = self.req.errno.load(Ordering::Relaxed);
        Some(if e == 0 {
            Ok(())
        } else {
            Err(std::io::Error::from_raw_os_error(e))
        })
    }

    /// The threads with no job in hand, into `out`; how many.
    fn idle(&self, out: &mut [u8; MAX_THREADS]) -> usize {
        let mut n = 0;
        for (w, wk) in self.workers.iter().enumerate() {
            if wk.finished.0.load(Ordering::Acquire) == wk.seq.0.load(Ordering::Relaxed) {
                out[n] = w as u8;
                n += 1;
            }
        }
        n
    }

    /// Read the rows of request `round` still in flight again, one on each idle
    /// thread, as far as the idle threads go.
    #[allow(clippy::too_many_arguments)]
    fn hedge(
        &self,
        round: u64,
        me: &Thread,
        fd: i32,
        rows: usize,
        row_bytes: usize,
        dst: *mut u8,
        offset: &impl Fn(usize) -> u64,
    ) {
        let mut idle = [0u8; MAX_THREADS];
        let n_idle = self.idle(&mut idle);
        let pending = claim(round, PENDING);
        let mut j = 0;
        for (i, c) in self.req.claims[..rows].iter().enumerate() {
            if j == n_idle {
                break;
            }
            if c.load(Ordering::Acquire) != pending {
                continue;
            }
            let item = Item {
                row: i as u32,
                off: offset(i),
            };
            let w = idle[j] as usize;
            self.hand(
                w,
                me,
                round,
                fd,
                rows,
                row_bytes,
                dst,
                true,
                std::iter::once(item),
            );
            self.counters.hedged.fetch_add(1, Ordering::Relaxed);
            j += 1;
        }
    }

    /// Hand worker `w` (idle) a job of `items`.
    #[allow(clippy::too_many_arguments)]
    fn hand(
        &self,
        w: usize,
        me: &Thread,
        round: u64,
        fd: i32,
        rows: usize,
        row_bytes: usize,
        dst: *mut u8,
        hedge: bool,
        items: impl Iterator<Item = Item>,
    ) {
        let wk = &self.workers[w];
        // SAFETY: the worker is idle (`finished == seq`, read with Acquire by
        // `idle`), and reads its job again only once `seq` advances below.
        let job = unsafe { &mut *wk.job.get() };
        job.round = round;
        job.fd = fd;
        job.row_bytes = row_bytes;
        job.rows = rows;
        job.dst = dst;
        job.hedge = hedge;
        job.count = 0;
        for it in items {
            job.items[job.count] = it;
            job.count += 1;
        }
        if job.caller.as_ref().map(Thread::id) != Some(me.id()) {
            job.caller = Some(me.clone());
        }
        let s = wk.seq.0.load(Ordering::Relaxed) + 1;
        wk.seq.0.store(s, Ordering::Release);
        if let Some(t) = wk.thread.get() {
            t.unpark();
        }
    }

    fn run(&'static self, w: usize) {
        let (class, _) = crate::generation::hot_thread::claim_for_spawned("default");
        super::READ_POOL_QOS.store(class as u8, Ordering::Relaxed);
        let me = &self.workers[w];
        let _ = me.thread.set(std::thread::current());
        let mut seen = 0u64;
        loop {
            let s = me.seq.0.load(Ordering::Acquire);
            if s == seen {
                std::thread::park();
                continue;
            }
            seen = s;
            // SAFETY: the caller wrote the job before advancing `seq` (read
            // above with Acquire) and writes it again only once `finished`
            // reaches `s`; the landing buffer is this thread's alone.
            let (job, buf) = unsafe { (&*me.job.get(), &mut *me.buf.get()) };
            self.serve(job, &mut buf[..job.row_bytes]);
            me.finished.0.store(s, Ordering::Release);
        }
    }

    /// Read each row of `job` still in flight and claim it; the claim's
    /// winner lands it in the caller's destination and counts it.
    fn serve(&self, job: &Job, buf: &mut [u8]) {
        let pending = claim(job.round, PENDING);
        for it in &job.items[..job.count] {
            let c = &self.req.claims[it.row as usize];
            // Landed by its twin, or its request has ended: nothing to read.
            if c.load(Ordering::Acquire) != pending {
                continue;
            }
            let errno = pread_exact(job.fd, buf, it.off);
            let state = if errno == 0 { LANDED } else { FAILED };
            if c.compare_exchange(
                pending,
                claim(job.round, state),
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_err()
            {
                continue;
            }
            if errno == 0 {
                // SAFETY: the claim makes row `it.row` this thread's alone;
                // it lies inside the caller's destination (`rows · row_bytes`
                // bytes), which stays live until `done` counts the row.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.as_ptr(),
                        job.dst.add(it.row as usize * job.row_bytes),
                        job.row_bytes,
                    )
                };
            } else {
                self.req.errno.store(errno, Ordering::Relaxed);
            }
            if job.hedge {
                self.counters.hedge_wins.fetch_add(1, Ordering::Relaxed);
            }
            if self.req.done.0.fetch_add(1, Ordering::AcqRel) as usize + 1 == job.rows {
                if let Some(t) = &job.caller {
                    t.unpark();
                }
            }
        }
    }
}

/// `pread` all of `buf` at `off`; 0, or the errno that stopped it.
fn pread_exact(fd: i32, buf: &mut [u8], mut off: u64) -> i32 {
    let mut at = 0;
    while at < buf.len() {
        // SAFETY: `buf[at..]` is writable for `buf.len() - at` bytes.
        let n = unsafe {
            libc::pread(
                fd,
                buf[at..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - at,
                off as libc::off_t,
            )
        };
        if n > 0 {
            at += n as usize;
            off += n as u64;
            continue;
        }
        if n == 0 {
            return libc::EIO;
        }
        let e = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(libc::EIO);
        if e != libc::EINTR {
            return e;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    /// A file of `rows` rows of `rb` bytes, row `i` filled with its own
    /// pattern.
    fn rows_file(tag: &str, rows: usize, rb: usize) -> (std::path::PathBuf, std::fs::File) {
        let path =
            std::env::temp_dir().join(format!("lloom-ngram-readers-{}-{tag}", std::process::id()));
        let bytes: Vec<u8> = (0..rows * rb)
            .map(|i| ((i / rb) * 31 + i % rb * 7) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        (path.clone(), std::fs::File::open(&path).unwrap())
    }

    /// `read`, retried while another test's request holds the set.
    fn read_retry(
        set: &Readers,
        file: &std::fs::File,
        offs: &[u64],
        rb: usize,
        dst: &mut [u8],
        hedge: Option<Duration>,
    ) -> std::io::Result<()> {
        loop {
            if let Some(r) = set.read(file.as_raw_fd(), offs.len(), rb, |i| offs[i], dst, hedge) {
                return r;
            }
            std::thread::yield_now();
        }
    }

    /// The set lands every row of a request where the request says, at every
    /// hedge (none, every row in flight hedged at once, a late one), and a
    /// request that cannot fit is refused rather than read wrong. The
    /// reference is the file's own bytes, row by row.
    #[test]
    fn every_row_lands_in_its_place_whatever_the_hedge() {
        let (rb, n_rows) = (90usize, 400usize);
        let (path, file) = rows_file("land", n_rows, rb);
        let all = std::fs::read(&path).unwrap();
        let set = readers(8).expect("the read threads");
        let ids: Vec<usize> = (0..40).map(|i| (i * 97 + 13) % n_rows).collect();
        let offs: Vec<u64> = ids.iter().map(|&id| (id * rb) as u64).collect();
        for hedge in [None, Some(Duration::ZERO), Some(Duration::from_micros(300))] {
            let mut dst = vec![0u8; ids.len() * rb];
            read_retry(set, &file, &offs, rb, &mut dst, hedge).unwrap();
            for (i, &id) in ids.iter().enumerate() {
                assert_eq!(
                    &dst[i * rb..(i + 1) * rb],
                    &all[id * rb..(id + 1) * rb],
                    "row {i} (file row {id}) at hedge {hedge:?}"
                );
            }
        }
        let mut big = vec![0u8; (MAX_ROWS + 1) * rb];
        assert!(set
            .read(
                file.as_raw_fd(),
                MAX_ROWS + 1,
                rb,
                |i| (i * rb) as u64,
                &mut big,
                None
            )
            .is_none());
        std::fs::remove_file(&path).ok();
    }

    /// A zero hedge re-reads, on the idle threads, the rows still in flight
    /// when the request is handed out, and the rows land right whichever
    /// read wins: the counters show the second reads, the bytes are the
    /// file's.
    #[test]
    fn a_hedged_row_lands_once_whichever_read_wins() {
        let rb = 90usize;
        let (path, file) = rows_file("hedge", 64, rb);
        let all = std::fs::read(&path).unwrap();
        let set = readers(8).expect("the read threads");
        let before = set.stats();
        for round in 0..50usize {
            let ids = [round % 64, (round * 7 + 3) % 64, (round * 13 + 5) % 64];
            let offs: Vec<u64> = ids.iter().map(|&id| (id * rb) as u64).collect();
            let mut dst = vec![0u8; ids.len() * rb];
            read_retry(set, &file, &offs, rb, &mut dst, Some(Duration::ZERO)).unwrap();
            for (i, &id) in ids.iter().enumerate() {
                assert_eq!(&dst[i * rb..(i + 1) * rb], &all[id * rb..(id + 1) * rb]);
            }
        }
        let after = set.stats();
        assert!(
            after.hedged > before.hedged,
            "50 requests hedged at zero re-read nothing: {after:?}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A read past the file's end fails the request with its error instead
    /// of landing a short row.
    #[test]
    fn a_failed_read_fails_its_request() {
        let rb = 90usize;
        let (path, file) = rows_file("fail", 4, rb);
        let set = readers(8).expect("the read threads");
        let mut dst = vec![0u8; 2 * rb];
        let offs = [10 * rb as u64, 11 * rb as u64];
        assert!(
            read_retry(set, &file, &offs, rb, &mut dst, None).is_err(),
            "a read past the end landed"
        );
        std::fs::remove_file(&path).ok();
    }
}
