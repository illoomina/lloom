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
use lloom_tensor::kernels::metal::GpuClock;
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

/// The reason a wait on work behind a cancelled fetch fails with.
const CANCELLED: &str =
    "expert fetch: the lane was cancelled before the routing showed; the work behind it was cancelled";

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
    /// Passes whose earlier epochs had not all completed when the worker
    /// reached their first job: the wait for them, here rather than on the
    /// decode thread (`Self::floor_settled`).
    pub floor_waits: AtomicU64,
    /// Looks again at a layer that found no slot for an expert: the slots
    /// free up as their readers complete and their loads land.
    pub retries: AtomicU64,
}

struct Lane {
    event: HostEvent,
    state: parking_lot::Mutex<State>,
    cv: parking_lot::Condvar,
    cancel: AtomicBool,
    stop: AtomicBool,
    poll: Duration,
    counters: FetchCounters,
    /// The device's clock: the worker waits here for the epochs before a
    /// pass's floor, so the decode thread does not.
    clock: Arc<GpuClock>,
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
            clock: device.gpu_clock(),
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

    /// Open a pass at `floor`: the worker writes a slot only once every
    /// command buffer before it has completed (`floor_settled`). The guard
    /// ends the pass (see [`FetchPass`]).
    pub fn begin_pass(self: &Arc<Self>, floor: Epoch) -> FetchPass {
        self.lane.state.lock().floor = floor;
        FetchPass {
            lane: Arc::clone(self),
            upto: None,
            done: false,
        }
    }

    /// Wait for every job and stop taking new ones' effects: the model's
    /// drop, before the tiers the jobs point at go.
    pub(crate) fn quiesce(&self) {
        self.drain(true);
    }

    /// Wait until the first `upto` registered jobs are done.
    fn wait_done(&self, upto: u64) {
        let mut g = self.lane.state.lock();
        while g.done < upto {
            self.lane.cv.wait(&mut g);
        }
    }

    fn registered(&self) -> u64 {
        self.lane.state.lock().registered
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
        self.register_job(TierRef(tier as *const Streamed), ids, topk, n_experts)
    }

    fn register_job(
        &self,
        tier: TierRef,
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
                tier,
                ids,
                topk,
                n_experts,
                value,
                floor,
            });
            value
        };
        self.lane.cv.notify_all();
        self.lane.event.commit_when_produced(value)
    }

    /// The counters once every job registered so far is done. The worker
    /// counts a job when it takes it (`jobs`), after the step registered it
    /// and before it signals the job's value; a decode under the ring
    /// readback returns to the host once its last step is encoded, so a raw
    /// snapshot taken right after it can miss the last job (CI gate 3,
    /// 2026-10-15: `jobs = 95` against 96 registered, the test asserting one
    /// wait a layer). This waits for the lane's own `done == registered`
    /// handshake first. Diagnostic callers only; the hot path never reads it.
    pub fn settled_stats(&self) -> FetchStats {
        self.wait_done(self.registered());
        self.stats()
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
            floor_waits: c.floor_waits.load(Ordering::Relaxed),
            retries: c.retries.load(Ordering::Relaxed),
        }
    }

    /// Wait until every registered job is done, cancelling the ones not yet
    /// served when `cancel`: the work held behind them is cancelled, then
    /// their values are raised with nothing loaded.
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

/// A fetching pass in progress. Its jobs are the ones registered up to
/// [`Self::seal`]: a later pass's jobs may be registered behind them while
/// this one is still open (the settle trailing the next step's encode).
/// Finished, it waits for its own jobs only; dropped unfinished (an error
/// path), it cancels what is pending and waits — so no job outlives the
/// model whose tiers it points at (the model's drop quiesces the lane too).
pub struct FetchPass {
    lane: Arc<FetchLane>,
    upto: Option<u64>,
    done: bool,
}

impl FetchPass {
    /// Mark the end of this pass's jobs: everything registered so far.
    pub fn seal(&mut self) {
        self.upto = Some(self.lane.registered());
    }

    /// End the pass after its device work completed: its jobs are served.
    pub fn finish(mut self) {
        let upto = self.upto.unwrap_or_else(|| self.lane.registered());
        self.lane.wait_done(upto);
        self.done = true;
    }
}

impl Drop for FetchPass {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let pending = {
            let g = self.lane.lane.state.lock();
            g.done < g.registered
        };
        self.lane.drain(pending);
    }
}

/// Wait until every epoch before `floor` has completed: the commands a
/// pass's floor stands after, whose slots the worker may then write. The
/// epoch AT the floor is never waited for — it holds this pass's own joins.
/// `false` when the wait outlived the fail-safe or the lane was cancelled.
fn floor_settled(lane: &Lane, floor: Epoch) -> bool {
    if floor.0 == 0 || lane.clock.is_complete(Epoch(floor.0 - 1)) {
        return true;
    }
    lane.counters.floor_waits.fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    while !lane.clock.is_complete(Epoch(floor.0 - 1)) {
        if lane.cancel.load(Ordering::Acquire) || t0.elapsed() > ROUTING_TIMEOUT {
            return false;
        }
        std::thread::sleep(lane.poll);
    }
    true
}

fn work(lane: &Lane) {
    let mut ids: Vec<u32> = Vec::new();
    let mut settled_floor: Option<Epoch> = None;
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
        if seen && settled_floor != Some(floor) {
            if floor_settled(lane, floor) {
                settled_floor = Some(floor);
            } else {
                lane.counters.timeouts.fetch_add(1, Ordering::Relaxed);
            }
        }
        if seen {
            // SAFETY: the job's pass has not returned (`TierRef`).
            let tier = unsafe { &*tier };
            // A layer that found no slot looks again: slots free up as their
            // readers complete and in-flight loads land, so a pass computes
            // on a missing expert only past the fail-safe.
            let (mut loaded, mut short) = tier.fetch_beside(&ids, floor);
            let t_retry = Instant::now();
            while short > 0
                && !lane.cancel.load(Ordering::Acquire)
                && t_retry.elapsed() <= ROUTING_TIMEOUT
            {
                lane.counters.retries.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(lane.poll);
                let (l, s) = tier.fetch_beside(&ids, floor);
                loaded += l;
                short = s;
            }
            if loaded + short > 0 {
                lane.counters.missed_layers.fetch_add(1, Ordering::Relaxed);
            }
            lane.counters.loads.fetch_add(loaded, Ordering::Relaxed);
            lane.counters.short.fetch_add(short, Ordering::Relaxed);
        }
        lane.counters
            .serve_ns
            .fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // A job given up to a cancellation leaves the table unwritten for a
        // pass nobody redoes: the work behind its wait is cancelled before
        // the raise, and never runs. The fail-safe's raise lets it run on the
        // table as it stands, for the settle to flag and redo.
        if !seen && lane.cancel.load(Ordering::Acquire) {
            if let Err(e) = lane.event.cancel_through(value, CANCELLED) {
                eprintln!("[expert-fetch] cancelling the work behind value {value}: {e}");
            }
        }
        {
            let mut g = lane.state.lock();
            g.jobs.pop_front();
            g.done += 1;
        }
        lane.cv.notify_all();
        lane.event.signal(value);
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use lloom_tensor::kernels::metal::HostInputArm;
    use lloom_tensor::metal_backend::CommandMarker;
    use lloom_tensor::{DType, Device, Tensor};
    use objc2_metal::{MTLDevice, MTLSharedEvent};

    /// The device on the commit arm; `None` on the GPU-wait arm, which holds
    /// nothing a cancellation could take.
    fn device() -> Option<MetalDevice> {
        let dev = Device::new_metal(0).ok()?;
        let m = dev.as_metal_device().ok()?.clone();
        if m.commands().host_input_arm() == HostInputArm::GpuWait {
            eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
            return None;
        }
        Some(m)
    }

    /// A routing slot the router has not written: the sentinel the step's
    /// start fills it with, which no expert id reaches.
    fn unwritten_routing(m: &MetalDevice, topk: usize) -> Arc<HostSlot> {
        let slot = HostSlot::new(m, topk, DType::U32).unwrap();
        slot.fill_u32(u32::MAX, topk).unwrap();
        Arc::new(slot)
    }

    /// The layer's work behind its wait, in the slot map's place: a compute,
    /// then `raise_progress` encodes a progress signal behind it on the
    /// stream. Cut, so it is held behind the job's value.
    fn encode_consumer(
        m: &MetalDevice,
        x: &Tensor,
        raise_progress: impl FnOnce(),
    ) -> (Tensor, CommandMarker) {
        let y = x.affine(2.0, 1.0).unwrap();
        raise_progress();
        let marker = m.flush_marker().unwrap();
        (y, marker)
    }

    /// What the stream owes after a fetch was cancelled: nothing behind the
    /// unwritten routing ran, the wait on it says why, the worker's raise
    /// after the cancellation found nothing held, and the stream goes on —
    /// fresh work computes and every epoch the cancelled recordings took
    /// completes with it, once.
    fn assert_cancelled_and_recovered(
        m: &MetalDevice,
        x: &Tensor,
        progress: impl Fn() -> u64,
        marker: &CommandMarker,
        cancelled_before: u64,
        late_before: u64,
    ) {
        let waited = m.wait_marker(marker);
        assert!(
            waited
                .as_ref()
                .is_err_and(|e| e.to_string().contains("expert fetch")),
            "the consumer's wait does not say it was cancelled: {waited:?}"
        );
        m.wait_until_completed().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(progress(), 0, "the work behind an unwritten slot map ran");
        let stats = m.commands().produced_stats();
        assert_eq!(stats.held_now, 0, "{stats:?}");
        assert!(stats.cancelled > cancelled_before, "{stats:?}");
        assert_eq!(
            stats.late_after_cancel,
            late_before + 1,
            "the worker's raise after the cancellation: {stats:?}"
        );
        let z = x.affine(2.0, 1.0).unwrap();
        assert_eq!(z.to_vec1::<f32>().unwrap(), vec![7.0f32; 16]);
        let clock = m.gpu_clock();
        assert_eq!(
            clock.completed_count().0,
            clock.current_epoch().0,
            "an epoch a cancelled recording took never completed"
        );
        m.commands().check_failed().unwrap();
    }

    /// The lane dropped with a layer's routing never written: the worker
    /// gives the job up, and the work encoded behind its wait — the slot
    /// map, the routed experts — is cancelled, never committed over the
    /// table as it stands (`docs/invariants.md` §15.1q).
    #[test]
    fn a_dropped_lane_cancels_the_work_behind_an_unwritten_routing_metal() {
        let Some(m) = device() else { return };
        let x = Tensor::new(&[3.0f32; 16], &Device::Metal(m.clone())).unwrap();
        m.wait_until_completed().unwrap();
        let progress = m.metal_device().as_ref().newSharedEvent().unwrap();
        let before = m.commands().produced_stats();
        let lane = FetchLane::new(&m, Duration::from_millis(1), "inherit").unwrap();
        lane.register_job(TierRef(std::ptr::null()), unwritten_routing(&m, 2), 2, 4)
            .unwrap();
        let (_y, marker) = encode_consumer(&m, &x, || {
            m.commands().encode_signal_event(&progress, 1).unwrap()
        });
        std::thread::sleep(Duration::from_millis(200));
        let held = m.commands().produced_stats();
        assert!(held.held_now > 0, "nothing held behind the job: {held:?}");
        assert_eq!(progress.signaledValue(), 0, "the held work ran early");
        drop(lane);
        assert_cancelled_and_recovered(
            &m,
            &x,
            || progress.signaledValue(),
            &marker,
            before.cancelled,
            before.late_after_cancel,
        );
        eprintln!(
            "FETCH_LANE_DROP api={} stats={:?}",
            m.commands().actual_command_api(),
            m.commands().produced_stats()
        );
    }

    /// The same through a pass dropped unfinished (a decode's error path).
    /// The job's tier is never read: the routing never shows.
    #[test]
    fn a_pass_dropped_unfinished_cancels_the_work_behind_its_unwritten_routing_metal() {
        let Some(m) = device() else { return };
        let x = Tensor::new(&[3.0f32; 16], &Device::Metal(m.clone())).unwrap();
        m.wait_until_completed().unwrap();
        let progress = m.metal_device().as_ref().newSharedEvent().unwrap();
        let before = m.commands().produced_stats();
        let lane = FetchLane::new(&m, Duration::from_millis(1), "inherit").unwrap();
        let mut pass = lane.begin_pass(Epoch(0));
        lane.register_job(TierRef(std::ptr::null()), unwritten_routing(&m, 2), 2, 4)
            .unwrap();
        let (_y, marker) = encode_consumer(&m, &x, || {
            m.commands().encode_signal_event(&progress, 1).unwrap()
        });
        pass.seal();
        std::thread::sleep(Duration::from_millis(200));
        assert!(m.commands().produced_stats().held_now > 0);
        drop(pass);
        assert_cancelled_and_recovered(
            &m,
            &x,
            || progress.signaledValue(),
            &marker,
            before.cancelled,
            before.late_after_cancel,
        );
        eprintln!(
            "FETCH_PASS_DROP api={} stats={:?}",
            m.commands().actual_command_api(),
            m.commands().produced_stats()
        );
    }
}
