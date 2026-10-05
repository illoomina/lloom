//! The expert prefetcher — reader threads that read experts' planes from the
//! checkpoint into the streamed tier's staging ring AHEAD of the layer that
//! needs them, so the read overlaps the GPU's work on the layers before it
//! (`design/docs/expert-streaming.md` §20.3 P1 / §20.4: "the GPU must never
//! block on an SSD read"; DS4 hides a prefill layer's loads behind the
//! previous layer's compute).
//!
//! # What the thread does and does not touch
//!
//! The threads `pread` into ring buffers and nothing else. They never write a
//! slot and never encodes: the copy from a staged buffer into the slot is
//! encoded by the thread that runs the layer (`Streamed::resolve`, through
//! `QTensor::write_staged`), on the compute encoder, so the write sits in
//! the encoder's hazard chain after every earlier reader of that slot —
//! exactly the ordering the synchronous path rests on (`expert_stream`,
//! `QMetalStorage::write_with`). A prefetch is a HINT: the residency map,
//! the router and the dispatch never depend on what was staged; an expert
//! the layer needs that is not staged is loaded as it always was, and a
//! staged expert the layer did not need is handed back to the ring and
//! counted (`wasted`). The stream is byte-identical with the prefetcher on
//! or off by construction.
//!
//! # Bounds
//!
//! Requests carry the requesting tier's generation; a new hint for the same
//! tier supersedes its pending requests (the thread skips them, `stale`).
//! Staged bytes are capped (`max_staged_bytes`): the thread parks when the
//! cap is reached and resumes as the tiers take or release what was staged,
//! so a prefetch burst never turns the ring's one-shots into growth.
//!
//! # Pieces
//!
//! One `pread` of a plane is latency-bound (the SSD reads 2.2 GB/s at one
//! in flight, 6.6 at four, 8.6 at eight), so a request is a piece of a plane
//! rather than an expert (`LLOOM_PREFETCH_SPLIT` = pieces per plane; 0 is the
//! expert as one request, its planes read one after another): a single miss
//! is read by several threads at once. An expert's pieces land in ring
//! entries acquired when its first piece is taken, and it is staged once the
//! last one has landed.
//!
//! Metal only: the ring and the copy are the Metal path's; the CPU tier
//! keeps its synchronous loads.

use lloom_tensor::kernels::metal::staging::SharedStagingRing;
use lloom_tensor::kernels::metal::BufferHandle;
use lloom_tensor::kernels::metal::Epoch;
use lloom_tensor::{MetalDevice, Result};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Where one expert's plane is read from: the file, the tensor's data start
/// and the plane size (`ExpertSource`'s addressing, cloned for the thread).
#[derive(Clone)]
pub struct PlaneSource {
    pub file: Arc<File>,
    pub data_start: u64,
    pub plane_bytes: usize,
}

impl PlaneSource {
    fn read(&self, expert: u32, dst: &mut [u8]) -> std::io::Result<()> {
        self.read_at(expert, 0, dst)
    }

    /// `dst.len()` bytes of `expert`'s plane from byte `at` of it.
    fn read_at(&self, expert: u32, at: usize, dst: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let off = self.data_start + expert as u64 * self.plane_bytes as u64 + at as u64;
        self.file.read_exact_at(dst, off)
    }
}

/// One expert's three planes, read and waiting in the ring for the layer
/// to land them.
pub struct Staged {
    pub expert: u32,
    /// Per projection: the ring entry (`None` = a one-shot) and its buffer.
    pub planes: [(Option<usize>, BufferHandle); 3],
}

impl Staged {
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(|(_, b)| b.size()).sum()
    }
}

/// What one request reads: the expert's three planes one after another, or
/// bytes `lo..hi` of one plane — a piece of assembly `asm`, which has
/// `pieces` in all.
#[derive(Clone, Copy, Debug)]
enum Part {
    Whole,
    Piece {
        asm: u64,
        plane: u8,
        lo: usize,
        hi: usize,
        pieces: usize,
    },
}

struct Req {
    key: usize,
    generation: u64,
    expert: u32,
    planes: [PlaneSource; 3],
    part: Part,
    /// A direct read: into these slots, once `epoch` is complete; the
    /// batch it belongs to (`direct_pending`) is told when it lands.
    direct: Option<(u64, [DirectDst; 3], Epoch)>,
    /// A `load_now` request: the layer is waiting for it, so it is read
    /// past the staged-bytes cap — the cap bounds what the HINTS hold, and
    /// a hint's leftovers holding the cap against a needed read is a
    /// deadlock the host cannot break (it is the one who would take them:
    /// measured 125 s in one `load_now` under the paired prefill, #749
    /// P26).
    now: bool,
}

/// Relaxed atomics, read off the hot path.
#[derive(Debug, Default)]
pub struct PrefetchCounters {
    /// Direct reads (into the slot, no ring), and the time their threads
    /// waited for the slots' last readers' epochs (drained by `read_direct`).
    pub direct: AtomicU64,
    pub direct_wait_ns: AtomicU64,
    /// Reader-thread time inside `pread`, and bytes, per arm: the ring's
    /// staging reads and the direct reads.
    pub stage_read_ns: AtomicU64,
    pub stage_read_bytes: AtomicU64,
    pub direct_read_ns: AtomicU64,
    pub direct_read_bytes: AtomicU64,
    /// Experts requested by the tiers.
    pub issued: AtomicU64,
    /// Experts the thread read into the ring.
    pub staged: AtomicU64,
    /// Staged experts a layer landed instead of loading.
    pub used: AtomicU64,
    /// Staged experts handed back unused.
    pub wasted: AtomicU64,
    /// Requests superseded before the thread reached them.
    pub stale: AtomicU64,
    /// Bytes the thread read.
    pub bytes: AtomicU64,
    /// Times the thread parked on the staged-bytes cap.
    pub parked: AtomicU64,
    /// Experts a layer read through the threads at the layer (`load_now`).
    pub now: AtomicU64,
    /// `submit_now` tickets not yet waited for, and the most there were at
    /// once: above one, several layers' reads were in flight together.
    pub now_outstanding: AtomicU64,
    pub now_peak: AtomicU64,
}

/// A snapshot of [`PrefetchCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    pub issued: u64,
    pub staged: u64,
    pub used: u64,
    pub wasted: u64,
    pub stale: u64,
    pub bytes: u64,
    pub parked: u64,
    pub now: u64,
    pub stage_read_ns: u64,
    pub stage_read_bytes: u64,
    pub direct_read_ns: u64,
    pub direct_read_bytes: u64,
    pub now_peak: u64,
}

/// An expert read in pieces, into the ring entries its first piece
/// acquired; staged when `left` — its pieces not yet landed — reaches zero.
struct Assembly {
    key: usize,
    generation: u64,
    expert: u32,
    bufs: [(Option<usize>, BufferHandle); 3],
    left: usize,
    /// A piece failed to read, or was cancelled before it ran.
    failed: bool,
}

impl Assembly {
    fn bytes(&self) -> usize {
        self.bufs.iter().map(|(_, b)| b.size()).sum()
    }
}

struct State {
    /// Direct-read batches issued, and the requests still to land per batch.
    direct_batches: u64,
    direct_pending: HashMap<u64, usize>,
    /// Experts being read in pieces, by assembly id, and the next id.
    assembling: HashMap<u64, Assembly>,
    next_asm: u64,
    reqs: VecDeque<Req>,
    done: HashMap<usize, Vec<Staged>>,
    generation: HashMap<usize, u64>,
    staged_bytes: usize,
    /// `take_needed` calls so far, and per tier the count at its last one:
    /// a tier's leftovers survive the takes of the NEXT tier (the pass may
    /// still be resolving this one in groups) and are released at the one
    /// after — the pass has moved on, and held leftovers across a whole
    /// pass exceed the cap many times over (measured: the threads parked
    /// 744 times and staged 1 847 of 27 061, issue #749 P19).
    takes: u64,
    last_take: HashMap<usize, u64>,
}

struct Shared {
    state: parking_lot::Mutex<State>,
    cv: parking_lot::Condvar,
    ring: SharedStagingRing,
    device: MetalDevice,
    counters: PrefetchCounters,
    max_staged_bytes: usize,
    stop: AtomicBool,
    /// The GPU clock a direct read gates on: the slot's last reader's epoch
    /// must be complete before the bytes land in place.
    clock: Arc<lloom_tensor::kernels::metal::GpuClock>,
    /// Pieces per plane (0: an expert is one request).
    split: usize,
}

/// A slot's bytes, for a direct read: a raw pointer into a Shared slot
/// pool. Distinct slots never overlap; the pool outlives the prefetcher's
/// jobs (the tier waits for them).
#[derive(Clone, Copy)]
pub struct DirectDst(pub *mut u8);
// SAFETY: a pointer into a Shared `MTLBuffer` the tier keeps alive; the
// reader writes its own plane's range only.
unsafe impl Send for DirectDst {}
unsafe impl Sync for DirectDst {}

/// One expert read straight into its slot's three planes (`read_direct`),
/// once `epoch` — the slot's last GPU reader's command buffer — is complete.
pub struct DirectJob {
    pub expert: u32,
    pub planes: [PlaneSource; 3],
    pub dsts: [DirectDst; 3],
    pub epoch: Epoch,
}

/// A [`Prefetcher::submit_now`] in flight: what was already staged, and
/// what the threads are reading.
pub struct NowTicket {
    key: usize,
    generation: u64,
    missing: Vec<u32>,
    have: Vec<Staged>,
}

/// A [`Prefetcher::submit_direct`] in flight (`None`: nothing submitted).
pub struct DirectTicket {
    batch: Option<u64>,
}

/// Bytes the prefetcher keeps staged at most: ~2 layers of a cold pool's
/// misses at Qwen4-exp's geometry (170 experts × 1.75 MB a layer), under
/// the ring's own 1 GB cap so the thread never forces one-shots.
pub const DEFAULT_MAX_STAGED_BYTES: usize = 768 << 20;

/// The prefetcher: one per streamed tier, shared by its layers, with its
/// thread. Dropped with the model; the thread is joined then.
pub struct Prefetcher {
    shared: Arc<Shared>,
    workers: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
}

/// Reader threads by default: one `pread` of a 0.58 MB plane is latency-bound
/// (the SSD reads 2.2 GB/s at one in flight, 6.6 at four, 8.6 at eight, 9.9
/// at thirty-two — `p25-ssd-random-0.58MB.txt`); eight took the warmed
/// prefill from 6.2 to 5.6 s and the hotlist warm from 5.1 to 4.1, sixteen
/// no further (P25). `LLOOM_PREFETCH_THREADS` overrides.
pub const DEFAULT_THREADS: usize = 8;

fn threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().stream.prefetch_threads.max(1))
}

/// Pieces per plane by default (`LLOOM_PREFETCH_SPLIT`, read once).
pub fn default_split() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().stream.prefetch_split)
}

/// A piece's bytes are a multiple of this (the last piece of a plane takes
/// the remainder).
const PIECE_ALIGN: usize = 16 << 10;

/// `plane_bytes` cut into at most `split` pieces, `(lo, hi)` each.
fn cut(plane_bytes: usize, split: usize) -> impl Iterator<Item = (usize, usize)> {
    let step = plane_bytes
        .div_ceil(split.max(1))
        .next_multiple_of(PIECE_ALIGN)
        .max(PIECE_ALIGN);
    (0..plane_bytes)
        .step_by(step)
        .map(move |lo| (lo, (lo + step).min(plane_bytes)))
}

impl std::fmt::Debug for Prefetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prefetcher {{ {:?} }}", self.stats())
    }
}

impl Prefetcher {
    pub fn new(device: &MetalDevice, ring: SharedStagingRing) -> Self {
        Self::with_cap(device, ring, DEFAULT_MAX_STAGED_BYTES)
    }

    pub fn with_cap(
        device: &MetalDevice,
        ring: SharedStagingRing,
        max_staged_bytes: usize,
    ) -> Self {
        Self::with_split(device, ring, max_staged_bytes, default_split())
    }

    /// [`Self::with_cap`] cutting each plane into `split` pieces (0: an
    /// expert is one request).
    pub fn with_split(
        device: &MetalDevice,
        ring: SharedStagingRing,
        max_staged_bytes: usize,
        split: usize,
    ) -> Self {
        let shared = Arc::new(Shared {
            state: parking_lot::Mutex::new(State {
                direct_batches: 0,
                direct_pending: HashMap::new(),
                assembling: HashMap::new(),
                next_asm: 0,
                reqs: VecDeque::new(),
                done: HashMap::new(),
                generation: HashMap::new(),
                staged_bytes: 0,
                takes: 0,
                last_take: HashMap::new(),
            }),
            cv: parking_lot::Condvar::new(),
            ring,
            device: device.clone(),
            counters: PrefetchCounters::default(),
            max_staged_bytes,
            stop: AtomicBool::new(false),
            clock: device.gpu_clock(),
            split,
        });
        let workers = (0..threads())
            .map(|i| {
                let w = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("expert-prefetch-{i}"))
                    .spawn(move || worker(w))
                    .expect("spawn expert-prefetch")
            })
            .collect();
        Self {
            shared,
            workers: parking_lot::Mutex::new(workers),
        }
    }

    /// The requests reading `expert` of tier `key` — one, or three planes
    /// × the split's pieces sharing an assembly id — appended to `out`.
    #[allow(clippy::too_many_arguments)]
    fn reqs_for(
        &self,
        g: &mut State,
        key: usize,
        generation: u64,
        expert: u32,
        planes: &[PlaneSource; 3],
        now: bool,
        direct: Option<(u64, [DirectDst; 3], Epoch)>,
        out: &mut Vec<Req>,
    ) {
        let split = self.shared.split;
        if split == 0 {
            out.push(Req {
                key,
                generation,
                expert,
                planes: planes.clone(),
                part: Part::Whole,
                direct,
                now,
            });
            return;
        }
        let asm = g.next_asm;
        g.next_asm += 1;
        let pieces: usize = planes
            .iter()
            .map(|p| cut(p.plane_bytes, split).count())
            .sum();
        for (i, src) in planes.iter().enumerate() {
            for (lo, hi) in cut(src.plane_bytes, split) {
                out.push(Req {
                    key,
                    generation,
                    expert,
                    planes: planes.clone(),
                    part: Part::Piece {
                        asm,
                        plane: i as u8,
                        lo,
                        hi,
                        pieces,
                    },
                    direct,
                    now,
                });
            }
        }
    }

    /// Put `reqs` behind the requests a layer is already waiting for and
    /// ahead of every hint, in order.
    fn push_now(g: &mut State, reqs: Vec<Req>) {
        let at = g.reqs.iter().take_while(|r| r.now).count();
        for (i, r) in reqs.into_iter().enumerate() {
            g.reqs.insert(at + i, r);
        }
    }

    /// Ask for `experts` of tier `key`, in order, from `planes` (gate, up,
    /// down). Supersedes the tier's pending requests and hands back what it
    /// had staged and not taken (a hint is per pass: what the last pass left
    /// is stale).
    pub fn request(&self, key: usize, experts: &[u32], planes: &[PlaneSource; 3]) {
        let (leftovers, finished) = {
            let mut g = self.shared.state.lock();
            let gen = g.generation.entry(key).or_insert(0);
            *gen += 1;
            let gen = *gen;
            let (_, finished) = cancel_key(&mut g, key);
            let left = g.done.remove(&key).unwrap_or_default();
            for s in &left {
                g.staged_bytes = g.staged_bytes.saturating_sub(s.bytes());
            }
            let mut reqs = Vec::with_capacity(experts.len());
            for &e in experts {
                self.reqs_for(&mut g, key, gen, e, planes, false, None, &mut reqs);
            }
            g.reqs.extend(reqs);
            (left, finished)
        };
        self.shared
            .counters
            .issued
            .fetch_add(experts.len() as u64, Ordering::Relaxed);
        self.give_back(leftovers);
        release_assemblies(&self.shared, finished);
        self.shared.cv.notify_all();
    }

    /// Read `jobs` straight into their slots through the reader threads —
    /// each once its slot's last reader's epoch is complete — and wait for
    /// all of them. The point-of-use read of the `pread` arm without the
    /// ring and the landing copy (`DESIGN.md` §20.4a, #749 P27). Returns
    /// the time waited for the epochs, summed over the jobs.
    pub fn read_direct(&self, key: usize, jobs: Vec<DirectJob>) -> std::time::Duration {
        let t = self.submit_direct(key, jobs);
        self.wait_direct(t)
    }

    /// [`Self::read_direct`]'s submission: the reads are queued ahead of
    /// every hint and run while the caller does other work; the ticket is
    /// waited for with [`Self::wait_direct`].
    pub fn submit_direct(&self, key: usize, jobs: Vec<DirectJob>) -> DirectTicket {
        if jobs.is_empty() {
            return DirectTicket { batch: None };
        }
        let n = jobs.len();
        let batch = {
            let mut g = self.shared.state.lock();
            g.direct_batches += 1;
            let batch = g.direct_batches;
            let gen = *g.generation.entry(key).or_insert(0);
            let mut reqs = Vec::with_capacity(n);
            for j in jobs {
                self.reqs_for(
                    &mut g,
                    key,
                    gen,
                    j.expert,
                    &j.planes,
                    true,
                    Some((batch, j.dsts, j.epoch)),
                    &mut reqs,
                );
            }
            g.direct_pending.insert(batch, reqs.len());
            Self::push_now(&mut g, reqs);
            batch
        };
        self.shared
            .counters
            .direct
            .fetch_add(n as u64, Ordering::Relaxed);
        self.shared.cv.notify_all();
        DirectTicket { batch: Some(batch) }
    }

    /// Wait for a [`Self::submit_direct`]'s reads to land; the time the
    /// threads waited for epochs since the last such wait.
    pub fn wait_direct(&self, t: DirectTicket) -> std::time::Duration {
        let Some(batch) = t.batch else {
            return std::time::Duration::ZERO;
        };
        let mut g = self.shared.state.lock();
        while g.direct_pending.get(&batch).is_some_and(|&left| left > 0) {
            self.shared.cv.wait(&mut g);
        }
        g.direct_pending.remove(&batch);
        std::time::Duration::from_nanos(
            self.shared
                .counters
                .direct_wait_ns
                .swap(0, Ordering::Relaxed),
        )
    }

    /// The staged experts of tier `key` that `needed` names, out of the
    /// prefetcher's books; the caller lands each one. What was staged and
    /// not needed by THIS group stays staged for the pass's next group of
    /// the same layer (a pass whose expert union exceeds the pool resolves
    /// in several groups) and is released by the tier's next hint if no
    /// group took it. The pass has reached the tier, so its requests still
    /// pending are cancelled (`stale`): the threads move on to the next
    /// layer's.
    pub fn take_needed(&self, key: usize, needed: &[u32]) -> Vec<Staged> {
        self.take_needed_with(key, needed, true)
    }

    /// Read `experts` of tier `key` NOW through the reader threads — the
    /// misses a layer has to have before it dispatches, in parallel instead
    /// of one `pread` after another on the encoding thread — and wait for
    /// them. What was already staged for them is taken as is; the rest is
    /// requested ahead of the queue (the layer is waiting) and waited for.
    /// An expert whose read failed (reported by the thread) is left out and
    /// the caller reads it itself.
    pub fn load_now(&self, key: usize, experts: &[u32], planes: &[PlaneSource; 3]) -> Vec<Staged> {
        let t = self.submit_now(key, experts, planes);
        self.wait_now(t)
    }

    /// [`Self::load_now`]'s submission: what is staged is taken, the rest
    /// queued behind the reads already waited for and ahead of every hint.
    /// Several layers' submissions then read concurrently
    /// (`Streamed::settle_batch`); each ticket is waited for with
    /// [`Self::wait_now`].
    pub fn submit_now(&self, key: usize, experts: &[u32], planes: &[PlaneSource; 3]) -> NowTicket {
        let have = self.take_needed_with(key, experts, false);
        let missing: Vec<u32> = experts
            .iter()
            .copied()
            .filter(|e| !have.iter().any(|s| s.expert == *e))
            .collect();
        if missing.is_empty() {
            return NowTicket {
                key,
                generation: 0,
                missing,
                have,
            };
        }
        let generation = {
            let mut g = self.shared.state.lock();
            let gen = *g.generation.entry(key).or_insert(0);
            let mut reqs = Vec::with_capacity(missing.len());
            for &e in &missing {
                self.reqs_for(&mut g, key, gen, e, planes, true, None, &mut reqs);
            }
            Self::push_now(&mut g, reqs);
            gen
        };
        self.shared
            .counters
            .issued
            .fetch_add(missing.len() as u64, Ordering::Relaxed);
        self.shared
            .counters
            .now
            .fetch_add(missing.len() as u64, Ordering::Relaxed);
        let c = &self.shared.counters;
        let open = c.now_outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        c.now_peak.fetch_max(open, Ordering::Relaxed);
        self.shared.cv.notify_all();
        NowTicket {
            key,
            generation,
            missing,
            have,
        }
    }

    /// Wait for a [`Self::submit_now`]'s experts: every one staged, or its
    /// read reported failed (the caller reads what is missing itself).
    pub fn wait_now(&self, t: NowTicket) -> Vec<Staged> {
        let NowTicket {
            key,
            generation: gen,
            missing,
            mut have,
        } = t;
        if missing.is_empty() {
            return have;
        }
        self.shared
            .counters
            .now_outstanding
            .fetch_sub(1, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            have.extend(self.take_needed_with(key, &missing, false));
            if missing.iter().all(|e| have.iter().any(|s| s.expert == *e)) {
                return have;
            }
            let mut g = self.shared.state.lock();
            let pending = g
                .reqs
                .iter()
                .any(|r| r.key == key && r.generation == gen && missing.contains(&r.expert))
                || g.assembling
                    .values()
                    .any(|a| a.key == key && a.generation == gen && missing.contains(&a.expert));
            let in_flight = g.staged_bytes > 0;
            let staged_here = g.done.get(&key).is_some_and(|v| !v.is_empty());
            if (!pending && !in_flight && !staged_here) || std::time::Instant::now() > deadline {
                return have;
            }
            self.shared
                .cv
                .wait_for(&mut g, std::time::Duration::from_millis(20));
        }
    }

    fn take_needed_with(&self, key: usize, needed: &[u32], cancel: bool) -> Vec<Staged> {
        let (out, expired, finished) = {
            let mut g = self.shared.state.lock();
            let mut finished = Vec::new();
            if cancel {
                let (dropped, fin) = cancel_key(&mut g, key);
                finished = fin;
                if dropped > 0 {
                    self.shared
                        .counters
                        .stale
                        .fetch_add(dropped, Ordering::Relaxed);
                }
                *g.generation.entry(key).or_insert(0) += 1;
            }
            // A different tier than the last take: the pass moved on; every
            // tier taken before the previous one is behind the pass now.
            let same_as_last = g.last_take.get(&key).copied() == Some(g.takes);
            if !same_as_last {
                g.takes += 1;
            }
            let now = g.takes;
            g.last_take.insert(key, now);
            let mut expired: Vec<Staged> = Vec::new();
            let old: Vec<usize> = g
                .last_take
                .iter()
                .filter(|(k, &t)| **k != key && t + 1 < now)
                .map(|(k, _)| *k)
                .collect();
            for k in old {
                if let Some(v) = g.done.remove(&k) {
                    expired.extend(v);
                }
                g.last_take.remove(&k);
            }
            for s in &expired {
                g.staged_bytes = g.staged_bytes.saturating_sub(s.bytes());
            }
            let mut out = Vec::new();
            if let Some(done) = g.done.get_mut(&key) {
                let mut keep = Vec::with_capacity(done.len());
                for s in done.drain(..) {
                    if needed.contains(&s.expert) {
                        out.push(s);
                    } else {
                        keep.push(s);
                    }
                }
                *done = keep;
            }
            for s in &out {
                g.staged_bytes = g.staged_bytes.saturating_sub(s.bytes());
            }
            (out, expired, finished)
        };
        self.give_back(expired);
        release_assemblies(&self.shared, finished);
        self.shared.cv.notify_all();
        out
    }

    /// Everything staged for tier `key` (tests, teardown).
    pub fn take(&self, key: usize) -> Vec<Staged> {
        let all: Vec<u32> = {
            let g = self.shared.state.lock();
            g.done
                .get(&key)
                .map(|v| v.iter().map(|s| s.expert).collect())
                .unwrap_or_default()
        };
        self.take_needed(key, &all)
    }

    /// A taken expert the layer landed (counted; its ring entries were
    /// stamped by the copy).
    pub fn used(&self, n: u64) {
        self.shared.counters.used.fetch_add(n, Ordering::Relaxed);
    }

    /// Taken experts the layer did not need: their buffers back to the ring.
    pub fn release(&self, staged: Vec<Staged>) {
        self.give_back(staged);
    }

    fn give_back(&self, staged: Vec<Staged>) {
        if staged.is_empty() {
            return;
        }
        let n = staged.len() as u64;
        let mut ring = self.shared.ring.lock();
        for s in staged {
            for (entry, buf) in &s.planes {
                ring.release(*entry, buf);
            }
        }
        drop(ring);
        self.shared.counters.wasted.fetch_add(n, Ordering::Relaxed);
    }

    pub fn stats(&self) -> PrefetchStats {
        let c = &self.shared.counters;
        PrefetchStats {
            issued: c.issued.load(Ordering::Relaxed),
            staged: c.staged.load(Ordering::Relaxed),
            used: c.used.load(Ordering::Relaxed),
            wasted: c.wasted.load(Ordering::Relaxed),
            stale: c.stale.load(Ordering::Relaxed),
            bytes: c.bytes.load(Ordering::Relaxed),
            parked: c.parked.load(Ordering::Relaxed),
            now: c.now.load(Ordering::Relaxed),
            stage_read_ns: c.stage_read_ns.load(Ordering::Relaxed),
            stage_read_bytes: c.stage_read_bytes.load(Ordering::Relaxed),
            direct_read_ns: c.direct_read_ns.load(Ordering::Relaxed),
            direct_read_bytes: c.direct_read_bytes.load(Ordering::Relaxed),
            now_peak: c.now_peak.load(Ordering::Relaxed),
        }
    }

    /// Experts staged and not yet taken, over every tier.
    pub fn pending(&self) -> usize {
        self.shared
            .state
            .lock()
            .done
            .values()
            .map(|v| v.len())
            .sum()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        for h in self.workers.lock().drain(..) {
            let _ = h.join();
        }
        // Whatever was staged and never taken goes back to the ring, and
        // so does every assembly whose pieces never all ran.
        let (left, partial): (Vec<Staged>, Vec<Assembly>) = {
            let mut g = self.shared.state.lock();
            g.staged_bytes = 0;
            g.reqs.clear();
            (
                g.done.drain().flat_map(|(_, v)| v).collect(),
                g.assembling.drain().map(|(_, a)| a).collect(),
            )
        };
        let mut ring = self.shared.ring.lock();
        for s in left {
            for (entry, buf) in &s.planes {
                ring.release(*entry, buf);
            }
        }
        for a in partial {
            for (entry, buf) in &a.bufs {
                ring.release(*entry, buf);
            }
        }
    }
}

/// Remove tier `key`'s pending requests. An expert with pieces already
/// taken is marked failed (its bytes will never be whole) and handed back
/// once the pieces in flight land; one whose last pieces were all pending
/// is returned here for the caller to release. Returns the experts dropped
/// before any of their reads ran, and the assemblies finished.
fn cancel_key(g: &mut State, key: usize) -> (u64, Vec<Assembly>) {
    let mut whole = 0u64;
    let mut pieces: HashMap<u64, usize> = HashMap::new();
    g.reqs.retain(|r| {
        if r.key != key {
            return true;
        }
        match r.part {
            Part::Whole => whole += 1,
            Part::Piece { asm, .. } => *pieces.entry(asm).or_default() += 1,
        }
        false
    });
    let mut dropped = whole;
    let mut finished = Vec::new();
    for (asm, n) in pieces {
        match g.assembling.get_mut(&asm) {
            Some(a) => {
                a.failed = true;
                a.left -= n;
                if a.left == 0 {
                    let a = g.assembling.remove(&asm).expect("present");
                    g.staged_bytes = g.staged_bytes.saturating_sub(a.bytes());
                    finished.push(a);
                }
            }
            None => dropped += 1,
        }
    }
    (dropped, finished)
}

/// Hand finished-but-unusable assemblies' entries back to the ring.
fn release_assemblies(w: &Shared, finished: Vec<Assembly>) {
    if finished.is_empty() {
        return;
    }
    let mut ring = w.ring.lock();
    for a in finished {
        for (entry, buf) in &a.bufs {
            ring.release(*entry, buf);
        }
    }
}

/// The ring entries for one expert's three planes, or none (every entry
/// taken is handed back on failure).
fn acquire_three(
    w: &Shared,
    planes: &[PlaneSource; 3],
) -> Result<[(Option<usize>, BufferHandle); 3]> {
    let mut ring = w.ring.lock();
    let mut got: Vec<(Option<usize>, BufferHandle)> = Vec::with_capacity(3);
    for src in planes {
        match ring.acquire(&w.device, src.plane_bytes) {
            Ok((entry, buf)) if !buf.contents().is_null() => got.push((entry, buf)),
            other => {
                if let Ok((entry, buf)) = &other {
                    ring.release(*entry, buf);
                }
                for (e, b) in &got {
                    ring.release(*e, b);
                }
                return match other {
                    Err(e) => Err(e),
                    Ok(_) => lloom_tensor::bail!(
                        "the staging buffer has no CPU mapping ({} bytes)",
                        src.plane_bytes
                    ),
                };
            }
        }
    }
    let mut it = got.into_iter();
    Ok([
        it.next().expect("three"),
        it.next().expect("three"),
        it.next().expect("three"),
    ])
}

/// What a popped request reads into.
enum Target {
    /// Straight into a slot (`read_direct`).
    Direct(u64, [DirectDst; 3], Epoch),
    /// Into ring entries of its own (`Part::Whole`, `stage`).
    Stage,
    /// A piece of an assembly, into this plane's ring entry.
    Piece(u64, BufferHandle),
}

/// The reader threads' scheduling class as the first of them read it after
/// its claim (`LLOOM_PREFETCH_QOS`), for a run's report: the host-planned pass
/// waits on these reads, so their class is stated, not inherited unseen
/// (`docs/invariants.md` §15.2 #10b).
static WORKER_QOS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The reader threads' class (`Unknown` before any started, or off macOS).
pub fn worker_qos() -> crate::generation::QosClass {
    crate::generation::QosClass::from_index(WORKER_QOS.load(Ordering::Relaxed))
}

fn worker(w: Arc<Shared>) {
    let (class, _) = crate::generation::hot_thread::claim_for_spawned(
        &crate::tuning::tuning().stream.prefetch_qos,
    );
    WORKER_QOS.store(class as u8, Ordering::Relaxed);
    loop {
        let (req, target) = {
            let mut g = w.state.lock();
            loop {
                if w.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(front) = g.reqs.front() {
                    let need: usize = front.planes.iter().map(|p| p.plane_bytes).sum();
                    let started = matches!(front.part, Part::Piece { asm, .. } if g.assembling.contains_key(&asm));
                    if front.now
                        || started
                        || g.staged_bytes + need <= w.max_staged_bytes
                        || g.staged_bytes == 0
                    {
                        break;
                    }
                    w.counters.parked.fetch_add(1, Ordering::Relaxed);
                }
                w.cv.wait(&mut g);
            }
            let req = g.reqs.pop_front().expect("front checked");
            // A direct read is never stale (a take's cancel is for the
            // hints) and takes no ring: it lands where it is told.
            let target = match (req.direct, req.part) {
                (Some((batch, dsts, epoch)), _) => Target::Direct(batch, dsts, epoch),
                (None, Part::Whole) => {
                    if g.generation.get(&req.key).copied().unwrap_or(0) != req.generation {
                        w.counters.stale.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    // Reserve the bytes now so a burst of requests cannot all
                    // pass the cap check before any of them lands.
                    g.staged_bytes += req.planes.iter().map(|p| p.plane_bytes).sum::<usize>();
                    Target::Stage
                }
                (
                    None,
                    Part::Piece {
                        asm, plane, pieces, ..
                    },
                ) => {
                    if !g.assembling.contains_key(&asm) {
                        // The expert's first piece: stale now means every
                        // piece of it is.
                        if g.generation.get(&req.key).copied().unwrap_or(0) != req.generation {
                            g.reqs.retain(
                                |r| !matches!(r.part, Part::Piece { asm: a, .. } if a == asm),
                            );
                            w.counters.stale.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        match acquire_three(&w, &req.planes) {
                            Ok(bufs) => {
                                let a = Assembly {
                                    key: req.key,
                                    generation: req.generation,
                                    expert: req.expert,
                                    bufs,
                                    left: pieces,
                                    failed: false,
                                };
                                g.staged_bytes += a.bytes();
                                g.assembling.insert(asm, a);
                            }
                            Err(e) => {
                                // A miss the layer will take synchronously.
                                eprintln!(
                                    "expert-prefetch: expert {} of tier {}: {e}",
                                    req.expert, req.key
                                );
                                g.reqs.retain(
                                    |r| !matches!(r.part, Part::Piece { asm: a, .. } if a == asm),
                                );
                                drop(g);
                                w.cv.notify_all();
                                continue;
                            }
                        }
                    }
                    let buf = BufferHandle::clone(&g.assembling[&asm].bufs[plane as usize].1);
                    Target::Piece(asm, buf)
                }
            };
            (req, target)
        };
        match target {
            Target::Direct(batch, dsts, epoch) => {
                // The slot's last reader must have run: the pool's clock says
                // when its command buffer completed.
                let t0 = std::time::Instant::now();
                while !w.clock.is_complete(epoch) {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                w.counters
                    .direct_wait_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let mut bytes = 0u64;
                let t_read = std::time::Instant::now();
                let spans: Vec<(usize, usize, usize)> = match req.part {
                    Part::Whole => (0..3).map(|i| (i, 0, req.planes[i].plane_bytes)).collect(),
                    Part::Piece { plane, lo, hi, .. } => vec![(plane as usize, lo, hi)],
                };
                for (i, lo, hi) in spans {
                    // SAFETY: `dsts[i]` addresses `plane_bytes` of a Shared
                    // slot the tier keeps alive, given to this job alone;
                    // `lo..hi` lies inside it and no other piece writes it.
                    let slice =
                        unsafe { std::slice::from_raw_parts_mut(dsts[i].0.add(lo), hi - lo) };
                    if let Err(e) = req.planes[i].read_at(req.expert, lo, slice) {
                        eprintln!(
                            "expert-prefetch: direct read of expert {} of tier {}: {e}",
                            req.expert, req.key
                        );
                    }
                    bytes += (hi - lo) as u64;
                }
                w.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                w.counters
                    .direct_read_ns
                    .fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
                w.counters
                    .direct_read_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
                let mut g = w.state.lock();
                if let Some(left) = g.direct_pending.get_mut(&batch) {
                    *left = left.saturating_sub(1);
                }
                drop(g);
                w.cv.notify_all();
            }
            Target::Stage => stage_whole(&w, &req),
            Target::Piece(asm, buf) => {
                let Part::Piece { plane, lo, hi, .. } = req.part else {
                    unreachable!("a piece target is a piece")
                };
                let t_read = std::time::Instant::now();
                // SAFETY: a Shared ring entry of at least the plane's bytes,
                // acquired for this assembly alone; `lo..hi` lies inside the
                // plane and no other piece writes it.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(buf.contents().add(lo), hi - lo) };
                let r = req.planes[plane as usize].read_at(req.expert, lo, dst);
                w.counters
                    .stage_read_ns
                    .fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
                w.counters
                    .stage_read_bytes
                    .fetch_add((hi - lo) as u64, Ordering::Relaxed);
                if let Err(e) = &r {
                    eprintln!(
                        "expert-prefetch: expert {} of tier {}: {e}",
                        req.expert, req.key
                    );
                }
                piece_landed(&w, asm, r.is_err());
            }
        }
    }
}

/// One piece of assembly `asm` has landed (or failed): the expert is staged
/// once its last piece is in — or handed back, if a piece failed or the
/// tier's generation moved on while it was read.
fn piece_landed(w: &Shared, asm: u64, failed: bool) {
    let mut g = w.state.lock();
    let Some(a) = g.assembling.get_mut(&asm) else {
        return;
    };
    a.failed |= failed;
    a.left -= 1;
    if a.left > 0 {
        return;
    }
    let a = g.assembling.remove(&asm).expect("present");
    let current = g.generation.get(&a.key).copied().unwrap_or(0) == a.generation;
    if !a.failed && current {
        let bytes = a.bytes() as u64;
        g.done.entry(a.key).or_default().push(Staged {
            expert: a.expert,
            planes: a.bufs,
        });
        drop(g);
        w.counters.staged.fetch_add(1, Ordering::Relaxed);
        w.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
        w.cv.notify_all();
        return;
    }
    g.staged_bytes = g.staged_bytes.saturating_sub(a.bytes());
    drop(g);
    if !a.failed {
        w.counters.stale.fetch_add(1, Ordering::Relaxed);
    }
    release_assemblies(w, vec![a]);
    w.cv.notify_all();
}

/// A whole-expert request: its three planes read one after another into
/// ring entries, then staged (or handed back if superseded meanwhile).
fn stage_whole(w: &Shared, req: &Req) {
    match stage(w, req) {
        Ok(staged) => {
            let bytes = staged.bytes() as u64;
            let mut g = w.state.lock();
            if g.generation.get(&req.key).copied().unwrap_or(0) == req.generation {
                g.done.entry(req.key).or_default().push(staged);
                drop(g);
                w.counters.staged.fetch_add(1, Ordering::Relaxed);
                w.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                w.cv.notify_all();
            } else {
                // Superseded while reading: back to the ring, unbooked.
                g.staged_bytes = g.staged_bytes.saturating_sub(staged.bytes());
                drop(g);
                let mut ring = w.ring.lock();
                for (entry, buf) in &staged.planes {
                    ring.release(*entry, buf);
                }
                w.counters.stale.fetch_add(1, Ordering::Relaxed);
            }
        }
        Err(e) => {
            // A read that failed is a miss the layer will take
            // synchronously (and report, if it fails there too).
            eprintln!(
                "expert-prefetch: expert {} of tier {}: {e}",
                req.expert, req.key
            );
            let mut g = w.state.lock();
            g.staged_bytes = g
                .staged_bytes
                .saturating_sub(req.planes.iter().map(|p| p.plane_bytes).sum::<usize>());
            drop(g);
            w.cv.notify_all();
        }
    }
}

/// Read the three planes of `req.expert` into ring buffers.
fn stage(w: &Shared, req: &Req) -> Result<Staged> {
    let mut planes: Vec<(Option<usize>, BufferHandle)> = Vec::with_capacity(3);
    for src in &req.planes {
        let (entry, buf) = {
            let mut ring = w.ring.lock();
            ring.acquire(&w.device, src.plane_bytes)?
        };
        let ptr = buf.contents();
        if ptr.is_null() {
            let mut ring = w.ring.lock();
            ring.release(entry, &buf);
            for (e, b) in &planes {
                ring.release(*e, b);
            }
            lloom_tensor::bail!(
                "the staging buffer has no CPU mapping ({} bytes)",
                src.plane_bytes
            )
        }
        // SAFETY: a Shared buffer of at least `plane_bytes` bytes acquired
        // from the ring — its last copy complete, or fresh — mapped at
        // `ptr`; nothing else references it until it is staged.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, src.plane_bytes) };
        let t_read = std::time::Instant::now();
        let r = src.read(req.expert, dst);
        w.counters
            .stage_read_ns
            .fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
        w.counters
            .stage_read_bytes
            .fetch_add(src.plane_bytes as u64, Ordering::Relaxed);
        if let Err(e) = r {
            let mut ring = w.ring.lock();
            ring.release(entry, &buf);
            for (e, b) in &planes {
                ring.release(*e, b);
            }
            return Err(lloom_tensor::Error::wrap(e));
        }
        planes.push((entry, buf));
    }
    let mut it = planes.into_iter();
    Ok(Staged {
        expert: req.expert,
        planes: [
            it.next().expect("three"),
            it.next().expect("three"),
            it.next().expect("three"),
        ],
    })
}
