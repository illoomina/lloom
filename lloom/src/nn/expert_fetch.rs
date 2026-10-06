//! The fetch inside the token: a device-planned decode step whose router
//! chose experts the pool does not hold waits ON THE DEVICE, at that layer,
//! for a worker to read them beside the pass — no replay of the token, no
//! host plan (`design/docs/expert-streaming.md` §20.14c).
//!
//! The encoding thread registers one job per layer as it encodes the
//! layer's join point and encodes a wait for the job's value on a
//! host-signalled event (`HostEvent`, the PLE fill's shape). The worker
//! takes the jobs in order: it watches the layer's routing ids in the
//! tier's host-read slot (the router writes them there; the slot was
//! filled with a sentinel before the pass), hands the experts the pool
//! lacks to the tier (`Streamed::fetch_beside`: slots chosen, the planes
//! read straight into them by the prefetcher's reader threads, the
//! device's residency table told), then raises the event. The layer's slot
//! map and routed experts are encoded after the wait, so they read the
//! table and the bytes the worker left.
//!
//! The worker never spins: it sleeps between looks at the slot
//! (`docs/invariants.md` §15.2 #10b). A GPU→host event would tell it of the
//! routing ~1 ms late (§20.14b).

use crate::nn::expert_stream::{FetchStats, Streamed};
use lloom_tensor::metal_backend::{HostEvent, HostSlot};
use lloom_tensor::retire::Epoch;
use lloom_tensor::{MetalDevice, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long the worker waits for a layer's routing before it gives the
/// layer up (raises its event with nothing loaded): the GPU then maps the
/// pool as it stands, a missing expert is flagged, and the step is redone.
/// A fail-safe against a wait that would otherwise never be satisfied.
const ROUTING_TIMEOUT: Duration = Duration::from_secs(2);

/// One layer of one pass: the tier, where its routing lands, the value the
/// device waits for, and the epoch before which every command is known
/// complete (the pass's floor, `FetchLane::begin_pass`).
struct Job {
    tier: TierRef,
    ids: Arc<HostSlot>,
    topk: usize,
    n_experts: usize,
    value: u64,
    floor: Epoch,
}

/// A tier of the pass that registered the job.
///
/// SAFETY of the pointer: the pass that registers a job does not return
/// before the job is done — on success its settle waits for the device,
/// which waits for every job's value, and the worker marks a job done before
/// it raises the value; on any other exit [`FetchPass`]'s drop cancels the
/// jobs left and waits for the worker. The model, and so the tier, outlives
/// the pass.
struct TierRef(*const Streamed);
// SAFETY: see the type; `Streamed` is shared across threads by `&` already
// (its state is behind its own mutex and atomics).
unsafe impl Send for TierRef {}

#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    /// The last value handed out; the event starts at the lane's base.
    next: u64,
    registered: u64,
    done: u64,
    floor: Epoch,
}

/// Counters, relaxed atomics (an instrument read off the hot path).
#[derive(Debug, Default)]
pub struct FetchCounters {
    /// Jobs taken, layers that missed, experts loaded, experts that found
    /// no slot (the step is redone), routings never seen in time.
    pub jobs: AtomicU64,
    pub missed_layers: AtomicU64,
    pub loads: AtomicU64,
    pub short: AtomicU64,
    pub timeouts: AtomicU64,
    /// Host time from a job's turn to its routing seen, and from the routing
    /// seen to the event raised (the loads, when there were any).
    pub watch_ns: AtomicU64,
    pub serve_ns: AtomicU64,
}


struct Lane {
    event: HostEvent,
    state: parking_lot::Mutex<State>,
    cv: parking_lot::Condvar,
    cancel: AtomicBool,
    stop: AtomicBool,
    poll: Duration,
    counters: FetchCounters,
}

/// The worker and its queue: one per model, made on the first fetching
/// step and kept for the model's life.
pub struct FetchLane {
    lane: Arc<Lane>,
    worker: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for FetchLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FetchLane {{ {:?} }}", self.stats())
    }
}

impl FetchLane {
    /// A lane on `device`'s command stream, its worker sleeping `poll`
    /// between looks at a routing and claiming the class `qos`.
    pub fn new(device: &MetalDevice, poll: Duration, qos: &str) -> Result<Arc<Self>> {
        let event = HostEvent::new(device)?;
        let base = event.signaled();
        let lane = Arc::new(Lane {
            event,
            state: parking_lot::Mutex::new(State {
                next: base,
                ..State::default()
            }),
            cv: parking_lot::Condvar::new(),
            cancel: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            poll,
            counters: FetchCounters::default(),
        });
        let w = Arc::clone(&lane);
        let qos = qos.to_string();
        let worker = std::thread::Builder::new()
            .name("expert-fetch".into())
            .spawn(move || {
                crate::generation::hot_thread::claim_for_spawned(&qos);
                work(&w)
            })
            .map_err(lloom_tensor::Error::wrap)?;
        Ok(Arc::new(Self {
            lane,
            worker: parking_lot::Mutex::new(Some(worker)),
        }))
    }

    /// Open a pass: every command buffer before `floor` is known complete,
    /// so a slot whose last reader ran before it may be written at once.
    /// The guard ends the pass (see [`FetchPass`]).
    pub fn begin_pass(self: &Arc<Self>, floor: Epoch) -> FetchPass {
        self.lane.state.lock().floor = floor;
        FetchPass {
            lane: Arc::clone(self),
        }
    }

    /// Register `tier`'s layer of the open pass and encode the device's
    /// wait for it: the commands encoded after this run once the worker has
    /// served the routing the router writes into `ids`.
    pub(crate) fn register(
        &self,
        tier: &Streamed,
        ids: Arc<HostSlot>,
        topk: usize,
        n_experts: usize,
    ) -> Result<()> {
        let value = {
            let mut g = self.lane.state.lock();
            g.next += 1;
            g.registered += 1;
            let value = g.next;
            let floor = g.floor;
            g.jobs.push_back(Job {
                tier: TierRef(tier as *const Streamed),
                ids,
                topk,
                n_experts,
                value,
                floor,
            });
            value
        };
        self.lane.cv.notify_all();
        self.lane.event.encode_wait(value)
    }

    pub fn stats(&self) -> FetchStats {
        let c = &self.lane.counters;
        FetchStats {
            jobs: c.jobs.load(Ordering::Relaxed),
            missed_layers: c.missed_layers.load(Ordering::Relaxed),
            loads: c.loads.load(Ordering::Relaxed),
            short: c.short.load(Ordering::Relaxed),
            timeouts: c.timeouts.load(Ordering::Relaxed),
            watch_ns: c.watch_ns.load(Ordering::Relaxed),
            serve_ns: c.serve_ns.load(Ordering::Relaxed),
        }
    }

    /// Wait until every registered job is done, cancelling the ones not yet
    /// served when `cancel` (their values raised with nothing loaded).
    fn drain(&self, cancel: bool) {
        if cancel {
            self.lane.cancel.store(true, Ordering::Release);
            self.lane.cv.notify_all();
        }
        let mut g = self.lane.state.lock();
        while g.done < g.registered {
            self.lane.cv.wait(&mut g);
        }
        drop(g);
        self.lane.cancel.store(false, Ordering::Release);
    }
}

impl Drop for FetchLane {
    fn drop(&mut self) {
        self.drain(true);
        self.lane.stop.store(true, Ordering::Release);
        self.lane.cv.notify_all();
        if let Some(h) = self.worker.lock().take() {
            let _ = h.join();
        }
    }
}

/// A fetching pass in progress. Dropped, it waits for the worker to finish
/// every job the pass registered — at once on success (the device waited
/// for each), cancelling the rest otherwise — so no job outlives the pass
/// whose tiers it points at.
pub struct FetchPass {
    lane: Arc<FetchLane>,
}

impl FetchPass {
    /// End the pass after its device work completed: every job is served.
    pub fn finish(self) {}
}

impl Drop for FetchPass {
    fn drop(&mut self) {
        let pending = {
            let g = self.lane.lane.state.lock();
            g.done < g.registered
        };
        self.lane.drain(pending);
    }
}

fn work(lane: &Lane) {
    let mut ids: Vec<u32> = Vec::new();
    loop {
        let (tier, slot, topk, n_experts, value, floor) = {
            let mut g = lane.state.lock();
            loop {
                if lane.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(j) = g.jobs.front() {
                    break (
                        j.tier.0,
                        Arc::clone(&j.ids),
                        j.topk,
                        j.n_experts,
                        j.value,
                        j.floor,
                    );
                }
                lane.cv.wait(&mut g);
            }
        };
        lane.counters.jobs.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let seen = loop {
            if lane.cancel.load(Ordering::Acquire) {
                break false;
            }
            if slot.peek_u32(topk, &mut ids).is_err() {
                break false;
            }
            if ids.iter().all(|&e| (e as usize) < n_experts) {
                break true;
            }
            if t0.elapsed() > ROUTING_TIMEOUT {
                lane.counters.timeouts.fetch_add(1, Ordering::Relaxed);
                break false;
            }
            std::thread::sleep(lane.poll);
        };
        let t1 = Instant::now();
        lane.counters
            .watch_ns
            .fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        if seen {
            // SAFETY: the job's pass has not returned (`TierRef`).
            let tier = unsafe { &*tier };
            let (loaded, short) = tier.fetch_beside(&ids, floor);
            if loaded + short > 0 {
                lane.counters.missed_layers.fetch_add(1, Ordering::Relaxed);
            }
            lane.counters.loads.fetch_add(loaded, Ordering::Relaxed);
            lane.counters.short.fetch_add(short, Ordering::Relaxed);
        }
        lane.counters
            .serve_ns
            .fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        {
            let mut g = lane.state.lock();
            g.jobs.pop_front();
            g.done += 1;
        }
        lane.cv.notify_all();
        lane.event.signal(value);
    }
}
