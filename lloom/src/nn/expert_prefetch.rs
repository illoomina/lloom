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
//! Metal only: the ring and the copy are the Metal path's; the CPU tier
//! keeps its synchronous loads.

use lloom_tensor::kernels::metal::staging::SharedStagingRing;
use lloom_tensor::kernels::metal::PooledBuffer;
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
        use std::os::unix::fs::FileExt;
        let off = self.data_start + expert as u64 * self.plane_bytes as u64;
        self.file.read_exact_at(dst, off)
    }
}

/// One expert's three planes, read and waiting in the ring for the layer
/// to land them.
pub struct Staged {
    pub expert: u32,
    /// Per projection: the ring entry (`None` = a one-shot) and its buffer.
    pub planes: [(Option<usize>, Arc<PooledBuffer>); 3],
}

impl Staged {
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(|(_, b)| b.size()).sum()
    }
}

struct Req {
    key: usize,
    generation: u64,
    expert: u32,
    planes: [PlaneSource; 3],
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
}

struct State {
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
    *N.get_or_init(|| {
        std::env::var("LLOOM_PREFETCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(DEFAULT_THREADS)
    })
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
        let shared = Arc::new(Shared {
            state: parking_lot::Mutex::new(State {
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

    /// Ask for `experts` of tier `key`, in order, from `planes` (gate, up,
    /// down). Supersedes the tier's pending requests and hands back what it
    /// had staged and not taken (a hint is per pass: what the last pass left
    /// is stale).
    pub fn request(&self, key: usize, experts: &[u32], planes: &[PlaneSource; 3]) {
        let leftovers = {
            let mut g = self.shared.state.lock();
            let gen = g.generation.entry(key).or_insert(0);
            *gen += 1;
            let gen = *gen;
            g.reqs.retain(|r| r.key != key);
            let left = g.done.remove(&key).unwrap_or_default();
            for s in &left {
                g.staged_bytes = g.staged_bytes.saturating_sub(s.bytes());
            }
            for &e in experts {
                g.reqs.push_back(Req {
                    key,
                    generation: gen,
                    expert: e,
                    planes: planes.clone(),
                    now: false,
                });
            }
            left
        };
        self.shared
            .counters
            .issued
            .fetch_add(experts.len() as u64, Ordering::Relaxed);
        self.give_back(leftovers);
        self.shared.cv.notify_all();
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
        let mut have = self.take_needed_with(key, experts, false);
        let missing: Vec<u32> = experts
            .iter()
            .copied()
            .filter(|e| !have.iter().any(|s| s.expert == *e))
            .collect();
        if missing.is_empty() {
            return have;
        }
        let gen = {
            let mut g = self.shared.state.lock();
            let gen = *g.generation.entry(key).or_insert(0);
            for &e in missing.iter().rev() {
                g.reqs.push_front(Req {
                    key,
                    generation: gen,
                    expert: e,
                    planes: planes.clone(),
                    now: true,
                });
            }
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
        self.shared.cv.notify_all();
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
                .any(|r| r.key == key && r.generation == gen && missing.contains(&r.expert));
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
        let (out, expired) = {
            let mut g = self.shared.state.lock();
            if cancel {
                let before = g.reqs.len();
                g.reqs.retain(|r| r.key != key);
                let dropped = (before - g.reqs.len()) as u64;
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
            (out, expired)
        };
        self.give_back(expired);
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
        // Whatever was staged and never taken goes back to the ring.
        let left: Vec<Staged> = {
            let mut g = self.shared.state.lock();
            g.staged_bytes = 0;
            g.done.drain().flat_map(|(_, v)| v).collect()
        };
        let mut ring = self.shared.ring.lock();
        for s in left {
            for (entry, buf) in &s.planes {
                ring.release(*entry, buf);
            }
        }
    }
}

fn worker(w: Arc<Shared>) {
    loop {
        let req = {
            let mut g = w.state.lock();
            loop {
                if w.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(front) = g.reqs.front() {
                    let need: usize = front.planes.iter().map(|p| p.plane_bytes).sum();
                    if front.now
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
            if g.generation.get(&req.key).copied().unwrap_or(0) != req.generation {
                w.counters.stale.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            // Reserve the bytes now so a burst of requests cannot all pass
            // the cap check before any of them lands.
            g.staged_bytes += req.planes.iter().map(|p| p.plane_bytes).sum::<usize>();
            req
        };
        match stage(&w, &req) {
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
}

/// Read the three planes of `req.expert` into ring buffers.
fn stage(w: &Shared, req: &Req) -> Result<Staged> {
    let mut planes: Vec<(Option<usize>, Arc<PooledBuffer>)> = Vec::with_capacity(3);
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
        if let Err(e) = src.read(req.expert, dst) {
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
