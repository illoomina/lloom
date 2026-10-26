use crate::memory_admission::AllocationError;
use crate::{DType, Result};

use crate::kernels::metal::CommandMarker;
#[cfg(feature = "ug")]
use crate::kernels::metal::ComputePipeline;
use crate::kernels::{
    metal::{
        session_region::{RegionCounters, RegionPlan},
        Arena, ArenaCounters, ArenaCursor, ArenaDecline, ArenaLayout, ArenaOffsets, ArenaRecorder,
        Backing, Backings, BlitCommandsGuard, Buffer, BufferHandle, BufferPool, Commands,
        CommandsGuard, Device, FreeSummary, HeapStats, MTLResourceOptions, PassKey, PassKind,
        PassOpener, PassPlanCounters, PoolCounters, PoolOccupancySnapshot, PooledBuffer,
        ResidencySet, SessionRegion, StepPlan,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::MetalError;

/// Unique identifier for metal devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub(crate) use super::device_retirement::DeviceTeardown;

/// Unregisters an evicted buffer from the residency set, so Metal releases the
/// allocation the pool has just destroyed. A set retains what it holds, so an
/// evicted buffer left in it is a retention that reached 48 GB at long context
/// (`DESIGN.md` §6.3e, §6.3f).
///
/// Called on the host, from the pool's drain -- inside `acquire` or a
/// release, usually on the encoding thread, or an explicit drain (§6.4e) --
/// once per eviction round
/// with every buffer the round destroys. It
/// `removeAllocation`s them and leaves the commit to the next command-buffer
/// cut's `commit_pending`, which publishes the round with the cut's own
/// staged allocations: a commit per round there would be a driver
/// round-trip on the hot path (§6.3h's rule for a host-side removal).
pub(crate) struct ResidencyEvictionObserver {
    residency_set: Arc<ResidencySet>,
    /// The transient backings (§6.3k): an evicted heap sub-buffer's range
    /// goes back to its chunk instead of out of the residency set.
    backings: Arc<Backings>,
}

impl ResidencyEvictionObserver {
    pub(crate) fn new(residency_set: Arc<ResidencySet>, backings: Arc<Backings>) -> Self {
        Self {
            residency_set,
            backings,
        }
    }
}

impl crate::kernels::metal::BufferEvictionObserver for ResidencyEvictionObserver {
    fn on_evict(&self, buffers: &[Buffer]) {
        // One membership update per eviction ROUND, never per buffer:
        // `on_evict` is handed everything one round destroys.
        //
        // Both arms stay selectable (§12.3, §7.1a): `retire_batch` is the arm
        // #166's +0.062 ms was measured on, so a re-measurement has something to
        // measure against, and an operator hitting an unforeseen cost can get
        // the old behaviour back without a rebuild. The old arm's failure mode
        // is now documented rather than latent -- it is an unbounded retention,
        // not a safe deferral.
        // A heap-backed buffer was never in the set: its range returns to
        // its chunk, and only the rest go through the set.
        let rest: Vec<&Buffer> = buffers
            .iter()
            .filter(|b| {
                !(self.backings.transient_shared.reclaim(b)
                    || self.backings.transient_private.reclaim(b))
            })
            .collect();
        if rest.is_empty() {
            return;
        }
        if retain_evicted_allocations() {
            self.residency_set.retire_batch(rest);
        } else {
            // Removed now, committed at the next cut: eviction runs at the
            // pool's drain on the host (§6.4e), where a commit per round
            // would be a driver round-trip on the hot path.
            self.residency_set.remove_batch_deferred(rest);
        }
    }
}

/// Whether an evicted buffer's allocation is **retained** by Metal rather than
/// released (`DESIGN.md` §6.3d's behaviour, superseded by §6.3e/#210).
///
/// `false` is what ships: eviction unregisters. Setting
/// `CANDLE_METAL_RETAIN_EVICTED=1` restores the pre-#210 arm, which is the one
/// §6.3d priced and the one that reaches **48 GB of retention** at long context
/// under `KvAppend=Cat` -- so it is a measurement arm and an escape hatch, not a
/// tuning knob.
///
/// Read once: a per-eviction `var()` would put an environment lookup on the
/// decode path, which is the shape §6.4a prices.
pub(crate) fn retain_evicted_allocations() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // The legacy switch is the retain arm spelled the other way round;
        // where it is set it names the arm, so an old script still selects
        // the arm it pins.
        let t = crate::tuning::tuning();
        match t.metal.eager_residency_release {
            Some(eager) => !eager,
            None => t.pool.retain_evicted,
        }
    })
}

/// What [`MetalDevice::admit_bytes`] decided on: the ledger's committed bytes
/// at the decision (after a reclaim, when one ran), what the reclaim freed,
/// and the verdict. The ledger, not `currentAllocatedSize`: every byte the
/// process holds through its backings is a reservation on it — pools' free
/// and pending lists included — so two processes with the same allocation
/// history read the same figure and give the same verdict
/// (`profiles-and-memory-design.md` §4.2, §11), where the device figure
/// also counted Metal's own objects and whatever another thread had just
/// freed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassAdmission {
    pub committed: usize,
    pub reclaimed: usize,
    /// Bytes the ledger's yielding tenants pledged to give back (the
    /// arbiter's third rung, §4.2) when the pass did not fit after the
    /// reclaim: the owner of those tenants applies the pledge (an elastic
    /// expert tier parks slots) and asks again.
    pub pledged: u128,
    /// Bulk tenants whose concurrent grants prevented a bounded yield attempt.
    pub yield_contended: usize,
    /// The second rung ran: the queue waited out once and the pending lists
    /// given back (`--pool-no-admission-drain` turns it off).
    pub drained: bool,
    pub fits: bool,
    /// The machine gate's refusal (`machine.rs`), when the pass fit the
    /// quota but the bytes past what the process reserved on the machine
    /// did not fit the machine: `Refused { domain: Machine }` with the
    /// growth and the room. The arithmetic is the ledger's
    /// (`Ledger::last_machine_refusal`).
    pub machine: Option<crate::memory_admission::AllocationError>,
}

impl PassAdmission {
    pub fn yield_report(&self) -> crate::memory_arbitration::YieldReport {
        crate::memory_arbitration::YieldReport {
            pledged: self.pledged,
            contended: self.yield_contended,
        }
    }

    pub fn needs_yield_retry(&self) -> bool {
        self.yield_report().needs_retry(self.fits)
    }

    pub fn merge_yield_report(
        &mut self,
        previous: crate::memory_arbitration::YieldReport,
    ) -> crate::Result<()> {
        let merged = self
            .yield_report()
            .checked_merge(previous)
            .ok_or_else(|| crate::Error::Msg("yield report diagnostic total overflow".into()))?;
        self.pledged = merged.pledged;
        self.yield_contended = merged.contended;
        Ok(())
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    pub(crate) allocation_namespace: Arc<super::allocation_namespace::AllocationNamespace>,
    /// Argument blocks and small tables keyed by their bytes, so a value a
    /// kernel is handed every step — a params block, a chunk table, a live
    /// count — is uploaded once and bound by the same buffer until it
    /// changes. Its actual namespace/container owners are independently rooted
    /// through native generation and Core terminal/unknown custody.
    pub(crate) constants: Arc<ConstantsByValue>,
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<Commands>,

    /// Shared-storage buffer pool (`RESOURCE_OPTIONS`), for buffers the CPU
    /// also reads.
    ///
    /// Buffers return themselves here when their last handle drops; see
    /// [`crate::kernels::metal::buffer_pool`]. Reclamation is automatic
    /// and requires nothing of callers.
    pub(crate) buffers: BufferPool,

    /// Private-storage pool (`PRIVATE_RESOURCE_OPTIONS`, StorageModePrivate on
    /// macOS). Intermediate compute buffers do not need CPU access, so Private
    /// avoids coherency overhead.
    ///
    /// Despite the name this is the *activation* path, and it is where reuse
    /// matters most: it takes every intermediate a forward pass produces. It
    /// also holds the resident weights, because `to_dtype` on load allocates
    /// through it -- but those are held by live tensors for the process
    /// lifetime, so they are never in the free list and never looked at by a
    /// lookup. Reclaimability separates them, not pool identity.
    pub(crate) private_buffers: BufferPool,

    /// Where bytes come from (`profiles-and-memory-design.md` §4.1, L0):
    /// the transient pair a pool miss takes from — placement heap chunks in
    /// the residency set once (`DESIGN.md` §6.3k) when
    /// `transient_heaps_on()`, else one `MTLBuffer` per miss with its
    /// residency committed at the next cut — one per storage mode (the
    /// Shared pool's, the Private pool's); and the resident backings a
    /// persistent allocation takes from.
    pub(crate) backings: Arc<Backings>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`crate::kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
    /// Residency set registered on the command queue.
    pub(crate) residency_set: Arc<ResidencySet>,

    /// The memory ledger (`ledger.rs`): every persistent allocation's bytes
    /// under a reservation, the tenants' claims, the quota the binary sets.
    pub(crate) ledger: Arc<crate::kernels::metal::ledger::Ledger>,
    /// The installed arena's base, on the ledger for the arena's life.
    pub(crate) arena_reservation: Arc<Mutex<Option<crate::kernels::metal::ledger::Reservation>>>,

    /// The activation arena, when one is installed.
    ///
    /// `None` is the default and is the classical path: every allocation goes
    /// to the pool exactly as before, and the added cost is one `Option` test
    /// per allocation. Installing an arena changes **where an activation buffer
    /// comes from**, not how the pool decides a buffer is free (`DESIGN.md`
    /// §9.2a) -- the pool's `acquire`, `release`, free list and epoch gate are
    /// untouched.
    ///
    /// `RwLock` rather than `OnceLock` because the layout is selectable at run
    /// time: §9.3's parity check needs to run the same model under the packed
    /// layout and under the non-aliasing reference, and keeping both installable
    /// is the same discipline `ParamStyle` follows for binding styles (§11.3b).
    pub(crate) arena: Arc<RwLock<Option<Arena>>>,

    /// Observes one decode step's allocations so a plan can be derived from it.
    ///
    /// `None` except while recording. Candle is eager, so nothing declares the
    /// activation set up front -- but the dispatch sequence is stable
    /// (§11.1a.1), so one observed step describes the rest. This is §11.1a's
    /// record-then-replay, applied to allocation.
    pub(crate) arena_recorder: Arc<Mutex<Option<Arc<Mutex<ArenaRecorder>>>>>,

    /// An arena recording the decode loop drives through
    /// [`MetalDevice::begin_decode_step`] / [`MetalDevice::finish_decode_step`]
    /// rather than a harness driving it by hand. `None` unless armed.
    pub(crate) arena_auto: Arc<Mutex<Option<ArenaAutoRecord>>>,
    /// What the last armed recording installed. See [`ArenaAutoReport`].
    pub(crate) arena_auto_report: Arc<Mutex<Option<ArenaAutoReport>>>,
    /// How to record the arena again after a step drifts from its plan: the
    /// layout and step count the last arming used. `None` until armed.
    pub(crate) arena_rerecord: Arc<Mutex<Option<(ArenaLayout, usize)>>>,
    /// Arenas dropped and re-armed after a drift (`Arena::take_drifted`).
    pub(crate) arena_rerecordings: Arc<std::sync::atomic::AtomicU64>,

    /// The session region: one owned allocation serving the class the arena
    /// declines (#47, `DESIGN.md` §9.2x).
    ///
    /// Beside the arena rather than inside it because the two have different
    /// placement rules and only one of them may reuse a span across a step
    /// boundary — an arena slot's occupant is dead by the end of the step, a
    /// session tenant is read by the *next* one. `None` until installed, which
    /// is every process that has not asked for it.
    pub(crate) session_region: Arc<RwLock<Option<SessionRegion>>>,

    /// The pass-keyed plan cache (`DESIGN.md` §9.2aa): `None` until a
    /// harness enables it, and then the thing `begin_pass`/`finish_pass`
    /// drive. The arena slot above is what it installs per pass.
    pub(crate) pass_plans: Arc<Mutex<Option<PassPlans>>>,
    /// Whether activations may be served from a reusable base (an arena
    /// installed or recording, or the pass cache on). While set, no BLIT
    /// touches memory a compute dispatch may reuse: `copy2d`'s contiguous
    /// case and `allocate_zeros` take compute kernels, which the encoder's
    /// barrier chain orders; a blit encoder is outside that chain and, with
    /// hazard tracking off, its read of a slot is not ordered against the
    /// compute dispatch that next writes the slot (`DESIGN.md` §9.2aa —
    /// measured: a served prefill tile's conv carry read through the blit
    /// was wrong from its second tile, right through the compute kernel).
    /// Recording and serving must take the SAME path, since a blit is not a
    /// dispatch and the plan's positions count dispatches; hence one flag
    /// for both, set before the first recorded pass.
    pub(crate) base_mode: Arc<std::sync::atomic::AtomicBool>,
    /// Under the pass cache: whether a pass is open (`begin_pass` ..
    /// `finish_pass`). Bit 0 = the cache is on, bit 1 = a pass is open, bit 2
    /// = the open pass is served from a cached plan, bit 3 = the open pass is
    /// a model's own (`PassOpener::Model`). The recorder takes an
    /// allocation only while a pass is open: one made BETWEEN passes (a
    /// sampler's ids, a staged input) is nobody's ordinal — recorded, it
    /// would be charged to the pass before it and, read by the pass after
    /// it, counted as that pass's escape (measured: exactly one per batched
    /// step, the argmax's ids).
    pub(crate) pass_state: Arc<std::sync::atomic::AtomicU8>,
    /// Requests that reached a buffer pool while a served pass was open,
    /// `[shared, private]` (`DESIGN.md` §9.2aa, invariant I1). Counted per
    /// request, hit or miss, so the figure is a function of what the pass
    /// asked for and never of how far the GPU clock has let the free lists
    /// refill (§6.3b).
    pub(crate) served_pass_pool_requests: Arc<[std::sync::atomic::AtomicU64; 2]>,
    /// The allocation-sequence variant a model named for the next pass it
    /// prepares (`set_next_pass_variant`), folded into that pass's key by
    /// `begin_pass` and cleared there and at `finish_pass`.
    pub(crate) next_pass_variant: Arc<std::sync::atomic::AtomicU32>,
    /// The emulated arms this device admits: a type its GPU only emulates is
    /// refused unless the operator opted in (`DESIGN.md` §15.2a).
    pub(crate) emulated: Arc<crate::kernels::metal::arms::EmulatedArms>,
    // LAST: registered roots remain until every ordinary field alias releases.
    pub(crate) teardown: Arc<DeviceTeardown>,
}

/// One forward pass's plan, cached by its [`PassKey`]. Every plan is served
/// from ONE base buffer (`base`), sized to the largest cached plan: passes
/// are sequential on the queue, so consecutive passes of different keys reuse
/// the base exactly as consecutive decode steps reuse today's arena — by
/// barrier order, one step ahead of the GPU (§9.2, §9.2z-i).
///
/// A key's plan is built from a recording of its FIRST pass and the pass
/// AFTER it, whatever that pass's key: only the following pass can say which
/// of the first pass's values it read (session state — the conv state, the
/// held logits, a `Cat` KV), and those are excluded (`ArenaRecorder::
/// plan_previous`). Both of those passes run on the pool, under the recorder,
/// so every address the recorder keys on is a pool buffer's — an arena slot's
/// `raw_addr` is its base's and would not attribute.
pub struct PassPlans {
    state: crate::planner_transaction::PlannerState<PassState, PassKey>,
    layout: ArenaLayout,
    plans: HashMap<PassKey, StepPlan>,
    /// Arenas built over the current base, per key; rebuilt lazily after a
    /// base grow (an `Arena` holds views of the base it was built on).
    arenas: HashMap<PassKey, Arena>,
    base: Option<Buffer>,
    /// The base's bytes on the ledger, replaced with the base.
    base_reservation: Option<crate::kernels::metal::ledger::Reservation>,
    arena_generation: HashMap<PassKey, u64>,
    /// The pass in progress took a data-dependent branch its key does not
    /// name (`MetalDevice::pass_diverged`).
    diverged: bool,
    counters: PassPlanCounters,
    /// Per key: ordinals of its first sight the next pass read undeclared,
    /// as `(ordinal, bytes, allocated at dispatch, first read at dispatch of
    /// the following pass)`.
    escapes: HashMap<PassKey, Vec<(usize, usize, usize, usize)>>,
    /// `Ragged` keys in the order their plans were built; the oldest goes
    /// when there are more than `ragged_cap`.
    ragged_order: std::collections::VecDeque<PassKey>,
    ragged_cap: usize,
    /// The fraction of `recommendedMaxWorkingSetSize` the process may hold
    /// once the base is grown (`DESIGN.md` §9.5, the KV grow's gate). A base
    /// grow is an allocation made OUTSIDE every pass and held for the run,
    /// so it is admitted exactly as a KV bucket grow is: predicted, refused
    /// before anything is allocated, and the run stops rather than risking
    /// the machine. 0 turns the check off.
    admit_fraction: f64,
}

#[derive(Debug)]
struct PlannerInvalidated(&'static str);
impl std::fmt::Display for PlannerInvalidated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "planner transaction invalidated: {}; retry requires a new reservation",
            self.0
        )
    }
}
impl std::error::Error for PlannerInvalidated {}

struct ClosingPlan {
    identity: crate::planner_transaction::ClosingIdentity<
        super::allocation_namespace::AllocationNamespace,
        Mutex<ArenaRecorder>,
    >,
    have: usize,
    fraction: f64,
    closes: PassKey,
    state: PassState,
    plan: Option<StepPlan>,
    needed: Option<usize>,
}
impl ClosingPlan {
    fn validate(
        &self,
        pp: Option<&PassPlans>,
        namespace: &Arc<super::allocation_namespace::AllocationNamespace>,
        session: &Arc<()>,
        recorder: Option<&Arc<Mutex<ArenaRecorder>>>,
    ) -> Result<()> {
        if self
            .identity
            .validates(pp.map(|p| &p.state), namespace, session, recorder)
        {
            Ok(())
        } else {
            Err(crate::Error::wrap(PlannerInvalidated(
                "cache, base, session, namespace or recorder generation changed",
            )))
        }
    }
}
struct ClosingGuard<'a> {
    device: &'a MetalDevice,
    cache: Arc<()>,
    token: Arc<()>,
    armed: bool,
}
impl Drop for ClosingGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut g = self.device.pass_plans.lock();
        if let Some(pp) = g.as_mut() {
            if pp.state.abort_close(&self.cache, &self.token) {
                self.device
                    .pass_state
                    .fetch_and(!0b1110, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// The pass-plan cache set aside by [`MetalDevice::pause_pass_plans`];
/// dropping it puts the cache back.
#[must_use = "the cache is back as soon as the guard drops"]
pub struct PassPlansPaused<'a> {
    device: std::borrow::Cow<'a, MetalDevice>,
    saved: Option<PassPlans>,
}

impl Drop for PassPlansPaused<'_> {
    fn drop(&mut self) {
        if let Some(pp) = self.saved.take() {
            let displaced = self.device.pass_plans.lock().replace(pp);
            // A displaced cache may own native bases/charges. Never destroy it
            // while holding the planner lock (e.g. a reentrant paused restore).
            drop(displaced);
            self.device
                .pass_state
                .fetch_or(0b01, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// `LLOOM_SERVED_REQUEST_TRACE` (read once), or [`set_served_request_trace`]:
/// attribute each pool request a served pass makes to its call site in the
/// model crate. 0 = not yet read, 1 = off, 2 = on.
static SERVED_TRACE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn served_request_trace_on() -> bool {
    match SERVED_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let requested = crate::tuning::tuning().instrument.served_request_trace;
            let on = crate::tuning::effective_instrument("served_request_trace", requested);
            SERVED_TRACE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}

/// Turn the served-pass request attribution on or off for the process (a
/// test's switch; the knob sets it for a run).
pub fn set_served_request_trace(on: bool) -> crate::Result<()> {
    crate::tuning::check_profile_request("served_request_trace", on).map_err(crate::Error::wrap)?;
    SERVED_TRACE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// The admission's second rung (`admit_bytes`: wait the queue out once and
/// give the pending lists back before refusing): on unless
/// `--pool-no-admission-drain`. 0 = not yet read, 1 = off, 2 = on.
static ADMISSION_DRAIN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn admission_drain_on() -> bool {
    match ADMISSION_DRAIN.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = !crate::tuning::tuning().pool.no_admission_drain;
            set_admission_drain(on);
            on
        }
    }
}

/// Turn the admission's second rung on or off for the process (a test's
/// way to run the arm without it; the CLI's is the tuning flag).
pub fn set_admission_drain(on: bool) {
    ADMISSION_DRAIN.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(any(debug_assertions, feature = "profile"))]
type SiteCounts = HashMap<(usize, String), u64>;

#[cfg(any(debug_assertions, feature = "profile"))]
fn served_request_sites_map() -> &'static Mutex<SiteCounts> {
    static SITES: std::sync::OnceLock<Mutex<SiteCounts>> = std::sync::OnceLock::new();
    SITES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The instrument's half of [`MetalDevice::note_pool_request`]: the first two
/// frames of the `lloom` crate (the model's call site and its caller) of a
/// backtrace, counted per pool. Process-wide, like the trace recorder.
#[cfg(any(debug_assertions, feature = "profile"))]
fn note_served_request_site(which: usize) {
    let site = call_site();
    *served_request_sites_map()
        .lock()
        .entry((which, site))
        .or_default() += 1;
}

/// The call site of the current backtrace as the instruments name it: the
/// tensor-crate entry the model called, then the model's (`lloom` crate's)
/// two innermost frames. For attribution instruments only: it takes one
/// backtrace a call.
#[cfg(any(debug_assertions, feature = "profile"))]
pub(crate) fn call_site() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let frames: Vec<&str> = bt
        .lines()
        .map(str::trim)
        .filter_map(|l| l.split_once(": ").map(|(_, f)| f))
        .map(|f| match f.rfind("::h") {
            Some(i) if f.len() - i == 19 => &f[..i],
            _ => f,
        })
        .collect();
    let model = |f: &str| f.starts_with("lloom::") || f.starts_with("<lloom::");
    // The tensor-crate entry the model called (the frame just inside the
    // first model frame), then the model's two innermost frames.
    let first_model = frames.iter().position(|f| model(f));
    let api = first_model
        .and_then(|i| i.checked_sub(1))
        .map(|i| frames[i])
        .unwrap_or("?");
    let mut ours = frames.iter().filter(|f| model(f));
    match (ours.next(), ours.next()) {
        (Some(a), Some(b)) => format!("{api} <- {a} <- {b}"),
        (Some(a), None) => format!("{api} <- {a}"),
        _ => "(no lloom frame)".to_string(),
    }
}

/// What `LLOOM_SERVED_REQUEST_TRACE` attributed since the last call:
/// `(pool, site, requests)`, pool 0 shared and 1 private, most requests
/// first. Taking clears it, so a caller reads one window.
#[cfg(any(debug_assertions, feature = "profile"))]
pub fn served_pass_request_sites() -> Vec<(usize, String, u64)> {
    let mut v: Vec<(usize, String, u64)> = std::mem::take(&mut *served_request_sites_map().lock())
        .into_iter()
        .map(|((w, s), n)| (w, s, n))
        .collect();
    v.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    v
}

#[cfg(not(any(debug_assertions, feature = "profile")))]
pub fn served_pass_request_sites() -> Vec<(usize, String, u64)> {
    Vec::new()
}

/// One cached plan as the harness reports it: `(key, ordinals served,
/// ordinals recorded, undeclared escapes)`.
pub type PassPlanEntry = (PassKey, usize, usize, usize);

#[derive(Clone, Copy, Debug)]
enum PassState {
    /// Under a cached plan.
    Served(PassKey),
    /// On the pool, recording this key's first sight (and closing `pending`
    /// if one was waiting).
    Recording {
        key: PassKey,
        closes: Option<PassKey>,
    },
    /// On the pool, only to close `pending`'s liveness (this key is already
    /// cached or is being recorded again).
    Observing { closes: PassKey },
}

impl PassPlans {
    fn new(layout: ArenaLayout) -> Self {
        Self {
            state: crate::planner_transaction::PlannerState::new(),
            layout,
            plans: HashMap::new(),
            arenas: HashMap::new(),
            base: None,
            base_reservation: None,
            arena_generation: HashMap::new(),
            diverged: false,
            counters: PassPlanCounters::default(),
            escapes: HashMap::new(),
            ragged_order: std::collections::VecDeque::new(),
            ragged_cap: 32,
            // The KV grow's default (`Cache::kv_grow_fraction`), so the two
            // admissions speak of one budget rather than two.
            admit_fraction: 0.85,
        }
    }
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);
// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = RESOURCE_OPTIONS;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// An arena recording in progress, driven by the decode-step brackets.
///
/// The harnesses record the arena by hand -- `begin_arena_recording` on one
/// decode step, `next_arena_recording_step` on the next, `finish` + `install`
/// after them -- around a loop they own. A loop that lives in the engine
/// (`Session::decode`) cannot be edited per experiment, so the same sequence is
/// driven from the brackets it already calls, and the recording is armed once
/// before the run begins.
#[derive(Clone, Copy, Debug)]
pub struct ArenaAutoRecord {
    pub layout: ArenaLayout,
    /// Decode steps to let pass before recording. The first token after the
    /// prefill allocates a different set (`begin_arena_recording`), so one.
    pub skip: usize,
    /// Steps to record. Two is the minimum a plan can be built from
    /// (`next_arena_recording_step`).
    pub record_steps: usize,
    /// Decode steps begun since arming.
    step: usize,
}

/// What an armed recording installed, for the harness to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaAutoReport {
    pub layout: ArenaLayout,
    /// Ordinals the plan serves, of those recorded.
    pub covered: usize,
    pub total: usize,
    /// The decode step (since arming) the arena was installed at the end of.
    pub installed_at_step: usize,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

/// Uploads at most this long are served from the buffer pool rather than
/// allocated (`MetalDevice::new_buffer_with_data`): every per-step argument
/// block, chunk table and slot list is well under it; a prompt of thousands of
/// ids or a weight staging buffer is not.
pub const SMALL_UPLOAD_BYTES: usize = 32 * 1024;

/// Device buffers keyed by their contents (`MetalDevice::constant_buffer`).
///
/// A kernel's argument block is a value: the same bytes every step until a
/// shape changes. Uploading it per call — even as a pool hit — is a lookup, a
/// copy and a fresh buffer identity per dispatch; keyed by value, a repeated
/// block is the SAME buffer, which is also what an ICB command needs to bind
/// (§11.3l: a params buffer allocated per call makes ICB coverage zero).
/// The buffers are read-only for the GPU, so one may be bound by any number
/// of in-flight dispatches. Bounded FIFO: an evicted entry's buffer goes back
/// to the pool on the pool's own clock (§6.3b), so an in-flight reader is
/// never disturbed. Keys are small (the cap on an entry's bytes is
/// `CONSTANT_MAX_BYTES`); the miss path is one pooled upload.
pub struct ConstantsByValue {
    inner: Mutex<ConstantsInner>,
}

#[derive(Default)]
struct ConstantsInner {
    map: HashMap<Vec<u8>, BufferHandle>,
    order: std::collections::VecDeque<Vec<u8>>,
    hits: u64,
    misses: u64,
}

/// Entries held at most; past it the oldest is evicted.
const CONSTANT_ENTRIES: usize = 512;
/// The largest value the cache keys — a chunk table of 131072 / 256 = 512
/// entries is 2 KiB; a params block is under 128 B.
const CONSTANT_MAX_BYTES: usize = 4096;

impl ConstantsByValue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(ConstantsInner::default()),
        }
    }

    /// `(hits, misses)` since creation.
    pub fn counters(&self) -> (u64, u64) {
        let g = self.inner.lock();
        (g.hits, g.misses)
    }
}

impl Default for ConstantsByValue {
    fn default() -> Self {
        Self::new()
    }
}

/// The bytes of a `#[repr(C)]` plain-data value, for [`MetalDevice::constant_of`].
fn pod_bytes<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: `T: Copy` and the callers are `#[repr(C)]` parameter blocks
    // that are already uploaded byte for byte by `new_buffer_with_data`; a
    // padding byte that differs between two equal values costs a cache miss,
    // never a wrong binding.
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    fn allocate_backing(
        backing: &dyn Backing,
        bytes: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> std::result::Result<
        crate::kernels::metal::backing::Allocation,
        crate::memory_admission::AllocationError,
    > {
        match context {
            Some(context) => backing.alloc_in(bytes, context),
            None => backing.alloc(bytes),
        }
    }

    /// Source-only domain view constructor. Not exposed to model loaders until
    /// generation-anchor and independent IO custody are wired and reviewed.
    pub(crate) fn allocation_view(
        &self,
        domain: crate::memory_admission::ModelDomain,
    ) -> Result<Self> {
        // Standalone cold route only. Loaded uses allocation_view_in to split
        // its ALREADY promised term instead of reserving that term a second time.
        let bytes = DeviceTeardown::requested_metadata_bytes().map_err(MetalError::from)?;
        let pass = domain.reserve(bytes)?;
        self.allocation_view_in(domain, &pass.context())
    }

    pub(crate) fn allocation_view_in(
        &self,
        domain: crate::memory_admission::ModelDomain,
        metadata: &crate::memory_admission::AllocationContext,
    ) -> Result<Self> {
        if metadata.domain_key() != domain.key() {
            crate::bail!("namespace metadata and physical domain must name the supplied owner");
        }
        metadata.check_open()?;
        let namespace = super::allocation_namespace::AllocationNamespace::new(
            &self.device,
            self.commands.clock(),
            &self.residency_set,
            &self.ledger,
            Some(domain),
            self.allocation_namespace.native_domain.clone(),
            Some((&self.buffers, &self.private_buffers)),
        );
        let mut view = self.clone();
        view.constants = namespace.constants.clone();
        view.buffers = namespace.buffers.clone();
        view.private_buffers = namespace.private_buffers.clone();
        view.backings = namespace.backings.clone();
        view.arena_reservation = namespace.arena_reservation.clone();
        view.arena = namespace.arena.clone();
        view.arena_recorder = namespace.arena_recorder.clone();
        view.arena_auto = namespace.arena_auto.clone();
        view.arena_auto_report = namespace.arena_auto_report.clone();
        view.arena_rerecord = namespace.arena_rerecord.clone();
        view.arena_rerecordings = namespace.arena_rerecordings.clone();
        view.session_region = namespace.session_region.clone();
        view.pass_plans = namespace.pass_plans.clone();
        view.base_mode = namespace.base_mode.clone();
        view.pass_state = namespace.pass_state.clone();
        view.served_pass_pool_requests = namespace.served_pass_pool_requests.clone();
        view.next_pass_variant = namespace.next_pass_variant.clone();
        view.teardown = DeviceTeardown::new(
            namespace.clone(),
            self.seed.clone(),
            self.kernels.clone(),
            self.commands.in_flight_hold().cloned(),
            self.device.clone(),
            self.residency_set.clone(),
            self.ledger.clone(),
            metadata,
        )
        .map_err(MetalError::from)?;
        view.allocation_namespace = namespace;
        Ok(view)
    }

    /// Namespace-only lifetime anchor for the current generation. Resolve before
    /// EntryState, then attach after any cut under the same generation guard.
    /// Never capture ExecutionScope here: it owns Commands and would cycle.
    /// Legacy has an explicit namespace too; missing/cancelled accounted scope
    /// is rejected by active(), never replaced by a different payer.
    pub(crate) fn allocation_namespace_anchor(
        &self,
    ) -> Result<super::device_retirement::NamespaceOwner> {
        let owner = match super::allocation_scope::active(self)? {
            Some(scope) => scope
                .view
                .teardown
                .namespace_owner()
                .map_err(MetalError::from)?,
            None => self.teardown.namespace_owner().map_err(MetalError::from)?,
        };
        Ok(owner)
    }

    /// Stamp a newly allocated output with its actual payer namespace. Aliases
    /// retain their original device instead of using this allocation-only route.
    pub(crate) fn allocated_output_device(&self, buffer: &Buffer) -> Result<Self> {
        let view = match super::allocation_scope::active(self)? {
            Some(scope) => scope.view.clone(),
            None => self.clone(),
        };
        let origin = buffer
            .allocation_owner()
            .map_or(crate::memory_admission::AllocationOrigin::Legacy, |owner| {
                crate::memory_admission::AllocationOrigin::Domain(owner.domain_key())
            });
        view.declared_allocation_origin().validate_reuse(&origin)?;
        Ok(view)
    }

    /// Active output payer, not an input buffer's origin. Original-device and
    /// other-domain operands on the same queue route to the execution scope.
    pub fn allocation_origin(&self) -> Result<crate::memory_admission::AllocationOrigin> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return Ok(scope.view.declared_allocation_origin());
        }
        Ok(self.declared_allocation_origin())
    }

    /// Binding for explicit non-scoped constructors. Does not infer a first-use
    /// payer; an admitted original-device caller uses the scoped constructor.
    pub(crate) fn declared_allocation_origin(&self) -> crate::memory_admission::AllocationOrigin {
        crate::memory_admission::AllocationOrigin::Domain(
            self.allocation_namespace.native_domain.key(),
        )
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    /// How many allocations the residency set currently holds
    /// (`DESIGN.md` §6.3c).
    ///
    /// Exists so the guard can be observed rather than assumed. The defect it
    /// prevents is a machine panic, which is not something a test may provoke,
    /// so the tests assert on membership -- which is exactly this number.
    pub fn residency_set_len(&self) -> usize {
        self.residency_set.len()
    }

    /// Asks the OS to keep the residency set's memory wired
    /// ([`ResidencySet::request_residency`]).
    pub fn request_residency(&self) {
        self.residency_set.request_residency();
    }

    /// Tells the OS the residency set's memory need not stay wired
    /// ([`ResidencySet::end_residency`]): serve's keep-warm under memory
    /// pressure. The next request, or a command buffer on a queue the set
    /// is attached to, makes it resident again.
    pub fn end_residency(&self) {
        self.residency_set.end_residency();
    }

    /// Where the residency sets stand for the loaded line: the attachment
    /// (`kernels.residency.attach`) and the router's layout
    /// ([`ResidencySet::layout_line`]), e.g. `residency=queue 1 set 8.12 GiB`.
    pub fn residency_line(&self) -> String {
        format!(
            "residency={} {}",
            self.commands.residency_attachment().short(),
            self.residency_set.layout_line()
        )
    }

    /// The Metal 4 queue's per-slot residency counts
    /// ([`Commands::metal4_slot_residency`]).
    pub fn metal4_slot_residency(&self) -> Option<crate::kernels::metal::SlotResidency> {
        self.commands.metal4_slot_residency()
    }

    /// A handle to the residency set that outlives this device.
    ///
    /// Exists so the teardown guard can be observed from outside: after the
    /// last `MetalDevice` drops there is no device left to ask, so a test that
    /// wants to check the set was emptied has to be holding the set itself.
    pub fn residency_set_handle(&self) -> Arc<ResidencySet> {
        Arc::clone(&self.residency_set)
    }

    /// What the residency set has done — commits, adds, removes, retirements
    /// (`DESIGN.md` §6.3e).
    ///
    /// **`commits` is the quantity §6.3d's cost argument is about.** That
    /// section attributes eager unregistration's +0.062 ms/token to putting a
    /// `commit()` on the per-token path; this reports how many there are on
    /// each arm, so the argument can be checked against a rate rather than
    /// restated. `retired` nonzero with `removed` zero is the pre-#210
    /// behaviour, and it is the retention that reached 48 GB.
    pub fn residency_counters(&self) -> crate::kernels::metal::ResidencyCounters {
        self.residency_set.counters()
    }

    /// Zeroes the residency counters, so a caller can measure decode rather
    /// than decode-plus-load.
    pub fn reset_residency_counters(&self) {
        self.residency_set.reset_counters();
    }

    /// The two pools' free-list caps summed: the most their free lists hold.
    pub fn free_budgets(&self) -> usize {
        self.buffers.free_budget() + self.private_buffers.free_budget()
    }

    /// Sets the cap on bytes each pool retains in its free list, evicting
    /// immediately if the new cap is already exceeded.
    ///
    /// Eviction *destroys* buffers, so it unregisters them from the residency
    /// set on the way (`DESIGN.md` §6.3c).
    pub fn set_free_budget(&self, bytes: usize) {
        self.buffers.set_free_budget(bytes);
        self.private_buffers.set_free_budget(bytes);
    }

    /// Installs `DESIGN.md` §9.5k's derived residual as a cap on unplanned
    /// bytes, across both pools.
    ///
    /// `limit` is admission's `budget − predicted`; `planned_shared` and
    /// `planned_private` are the parts of the predicted set each pool serves.
    /// Both are needed because the two pools hold different classes and
    /// `live_bytes` already contains them (§9.5k):
    ///
    /// * **shared (`buffers`)** — the KV reserve, via `KvSlot::append` →
    ///   `Tensor::zeros` → `allocate_buffer`.
    /// * **private (`private_buffers`)** — the weights, via `to_dtype` →
    ///   `new_buffer_builder` → `new_buffer`, *and* every activation
    ///   intermediate (§6.3a's correction: `private_buffers` is not the weight
    ///   pool, it is where both live).
    ///
    /// The **arena is in neither**, because `install_arena` calls the raw
    /// device — it is the one class of the five genuinely outside the pools,
    /// which is why admission subtracts it from the budget and not from a
    /// pool's holdings.
    ///
    /// # The limit is per pool, and that is deliberate
    ///
    /// Each pool is given the whole residual rather than a share of it, so the
    /// bound is *"neither pool alone exceeds the residual"* rather than *"their
    /// sum does not"*. Splitting it would need a policy for the split that no
    /// measurement supports — §9.5b puts no figure on how unplanned bytes
    /// divide between the two — and the looser bound is the one that cannot
    /// refuse a run that would have been fine. **It is a backstop against
    /// unbounded growth, not an accountant**: §6.3b's stranding, the one
    /// instance of that shape ever measured, lands in a single pool.
    pub fn set_residual_cap(&self, limit: usize, planned_shared: usize, planned_private: usize) {
        self.buffers.set_residual_cap(limit, planned_shared);
        self.private_buffers
            .set_residual_cap(limit, planned_private);
    }

    /// Removes the residual cap from both pools, restoring the unbounded
    /// behaviour that shipped.
    pub fn clear_residual_cap(&self) {
        self.buffers.clear_residual_cap();
        self.private_buffers.clear_residual_cap();
    }

    /// Unplanned bytes in each pool, shared first, or `None` where no cap is
    /// installed.
    pub fn unplanned_bytes(&self) -> (Option<usize>, Option<usize>) {
        (
            self.buffers.unplanned_bytes(),
            self.private_buffers.unplanned_bytes(),
        )
    }

    /// How many `MetalDevice` handles are alive, counted through the teardown
    /// guard.
    ///
    /// The guard runs when this reaches zero. `MetalDevice` is `Clone` and every
    /// `Tensor` holds a clone, so this is what distinguishes "the last handle
    /// went" from "a handle went" -- the distinction that keeps the GPU wait off
    /// the per-tensor path.
    pub fn device_handles(&self) -> usize {
        self.teardown.handles()
    }

    /// Destroys every buffer currently sitting in a free list.
    ///
    /// This is the old `drop_unused_buffers` sweep, and the change is that it
    /// is no longer on the allocation path. It used to run on every
    /// `wait_until_completed`, walking every buffer in both pools to discover
    /// by `strong_count` which had become free -- so the pool oscillated
    /// between growing and being swept, and the sweep *destroyed* buffers that
    /// would immediately be re-created by the next allocation.
    ///
    /// Now a free buffer is already known to be free, so this exists only to
    /// hand memory back under pressure. Nothing calls it on the hot path.
    /// Returns the bytes given back.
    pub fn trim_unused_buffers(&self) -> usize {
        let mut freed = 0usize;
        for buffer in self.buffers.trim() {
            freed += buffer.length();
            if !self.backings.transient_shared.reclaim(&buffer) {
                self.residency_set.remove(&buffer);
            }
        }
        for buffer in self.private_buffers.trim() {
            freed += buffer.length();
            if !self.backings.transient_private.reclaim(&buffer) {
                self.residency_set.remove(&buffer);
            }
        }
        freed
    }

    /// A pass-level admission against the device: whether `need` more bytes
    /// fit under `budget` beside what the process holds — and, when they do
    /// not at first, after the pools' free lists are given back (the simplest
    /// cache yielding before a capability is refused,
    /// `profiles-and-memory-design.md` §2 principle 5). The free lists are
    /// elastic by definition and `currentAllocatedSize` counts them, so a
    /// refusal read on the first figure refuses a pass the process had room
    /// for. The caller decides on the figure returned; a reclaim is counted on
    /// the ledger. Not on the hot path: a decode step inside its slab asks
    /// nothing (§9.5e).
    ///
    /// Then the machine (`memory.md` §9.10): a pass that fits the quota and
    /// would take the process past what it reserved at load widens the
    /// reservation through the machine gate. When the machine cannot widen
    /// it, the quota's rungs run against the reservation — the free lists,
    /// then the pending lists after one wait — and the machine is asked
    /// again: a prefill pass is admitted while its predecessors' buffers
    /// still wait on their command buffers, bytes the next pass reuses once
    /// those complete, so a refusal read on the first figure would refuse a
    /// pass the reservation holds. Still past it, the pass is refused with
    /// the gate's refusal (`PassAdmission::machine`). Inside the
    /// reservation all of this is one comparison.
    pub fn admit_bytes(&self, need: usize, budget: usize) -> PassAdmission {
        let mut a = self.admit_quota(need, budget);
        if !a.fits {
            return a;
        }
        let ask = |committed: usize| {
            let target = crate::memory_arbitration::machine_target(committed, need)?;
            self.ledger.admit_machine(target)
        };
        let mut asked = ask(a.committed);
        if asked.is_err() {
            let mut reclaimed = self.trim_unused_buffers();
            a.committed = self.ledger.committed();
            asked = ask(a.committed);
            if asked.is_err() && !a.drained && admission_drain_on() {
                a.drained = true;
                self.ledger.note_drain();
                if self.wait_until_completed().is_ok() {
                    reclaimed += self.trim_unused_buffers();
                    a.committed = self.ledger.committed();
                    asked = ask(a.committed);
                }
            }
            a.reclaimed += reclaimed;
            self.ledger.note_reclaim(reclaimed);
        }
        if let Err(why) = asked {
            a.fits = false;
            a.machine = Some(why);
        }
        a
    }

    /// One stderr line for a pass-level refusal, on its cold path: what was
    /// refused, and the machine gate's arithmetic when the machine refused.
    /// The refusal itself travels as the `Copy` error.
    pub fn report_refusal(
        &self,
        what: &dyn std::fmt::Display,
        refusal: &crate::memory_admission::AllocationError,
    ) {
        use crate::memory_admission::{AllocationError, RefusalDomain};
        eprintln!("[{what}: {refusal}]");
        if let AllocationError::Refused {
            domain: RefusalDomain::Machine,
            ..
        } = refusal
        {
            if let Some(verdict) = self.ledger.last_machine_refusal() {
                eprintln!("{}", verdict.describe());
            }
        }
    }

    fn admit_quota(&self, need: usize, budget: usize) -> PassAdmission {
        use crate::memory_arbitration::{fits_budget, reclaim_target};
        let committed = self.ledger.committed();
        if fits_budget(committed, need, budget) {
            return PassAdmission {
                committed,
                reclaimed: 0,
                pledged: 0,
                yield_contended: 0,
                drained: false,
                fits: true,
                machine: None,
            };
        }
        let mut reclaimed = self.trim_unused_buffers();
        let mut committed = self.ledger.committed();
        let mut drained = false;
        if !fits_budget(committed, need, budget) && admission_drain_on() {
            drained = true;
            self.ledger.note_drain();
            // The second rung: what the host released and the GPU may still
            // read sits in the pending lists until their epochs complete.
            // Wait the queue out once — the pass would otherwise end the
            // reply — then give those back too. A host wait, on the refusal
            // path alone, never on a step (§21.13); the shared pool's rows
            // meet exhaustion the same way (`BatchPool::begin_rows`).
            if self.wait_until_completed().is_ok() {
                reclaimed += self.trim_unused_buffers();
                committed = self.ledger.committed();
            }
        }
        let mut yielded = crate::memory_arbitration::YieldReport::default();
        if !fits_budget(committed, need, budget) {
            // The third rung (§4.2, caches yield): the tenants that can
            // shrink pledge units — the expert tier's slots — until the
            // shortfall is covered or none can give more. The pledge is
            // applied by the tenants' owner, which asks again.
            let short = reclaim_target(committed, need, budget);
            yielded = self.ledger.yield_for_report(short);
            committed = self.ledger.committed();
        }
        self.ledger.note_reclaim(reclaimed);
        PassAdmission {
            committed,
            reclaimed,
            pledged: yielded.pledged,
            yield_contended: yielded.contended,
            drained,
            fits: fits_budget(committed, need, budget),
            machine: None,
        }
    }

    /// Scan and reuse counters for both pools. See `buffer_pool`.
    pub fn pool_counters(&self) -> (PoolCounters, PoolCounters) {
        (self.buffers.counters(), self.private_buffers.counters())
    }

    /// The by-value constants cache's hits and misses (`ConstantsByValue`):
    /// on a decode arm a miss per token is a params block or table whose
    /// bytes change every step and does not belong in the cache.
    pub fn constants_counters(&self) -> (u64, u64) {
        self.constants.counters()
    }

    /// What both free lists hold right now (`BufferPool::free_summary`).
    pub fn pool_free_summary(&self, top: usize) -> (FreeSummary, FreeSummary) {
        (
            self.buffers.free_summary(top),
            self.private_buffers.free_summary(top),
        )
    }

    pub fn reset_pool_counters(&self) {
        self.buffers.reset_counters();
        self.private_buffers.reset_counters();
    }

    /// Waits on both pools' state locks since construction, per side
    /// (`DESIGN.md` §6.4d). Difference two reads for a window.
    pub fn pool_contention(&self) -> crate::kernels::metal::contention::PoolContention {
        let (shared_host, shared_driver) = self.buffers.lock_contention();
        let (private_host, private_driver) = self.private_buffers.lock_contention();
        crate::kernels::metal::contention::PoolContention {
            shared_host,
            shared_driver,
            private_host,
            private_driver,
        }
    }

    /// Live and free occupancy for both pools, shared first.
    pub fn pool_occupancy(&self) -> (PoolOccupancySnapshot, PoolOccupancySnapshot) {
        (self.buffers.occupancy(), self.private_buffers.occupancy())
    }

    /// The transient heap chunks' `(shared, private)` stats (§6.3k).
    pub fn heap_stats(&self) -> (HeapStats, HeapStats) {
        (
            self.backings.transient_shared.stats(),
            self.backings.transient_private.stats(),
        )
    }

    /// Advisory ready, same-domain free capacity. Caller-owned scratch retains
    /// capacity across plans; duplicate physical sources count only once even
    /// through nested/aliased Fallback backings. It does not prove aligned fit.
    pub fn heap_reusable_bytes(
        &self,
        domain: Option<&crate::memory_admission::DomainKey>,
        scratch: &mut crate::kernels::metal::backing::ReuseCredit,
    ) -> Result<usize> {
        scratch.clear();
        self.backings
            .transient_shared
            .collect_reusable_allocations(domain, scratch);
        self.backings
            .transient_private
            .collect_reusable_allocations(domain, scratch);
        scratch
            .checked_total()
            .ok_or_else(|| crate::Error::Msg("reusable heap capacity exceeds usize".into()))
    }

    pub fn command_encoder<'a>(&'a self) -> Result<CommandsGuard<'a>> {
        let owner = self.allocation_namespace_anchor()?;
        let mut command_encoder = self.commands.command_encoder().map_err(MetalError::from)?;
        command_encoder
            .retain_namespace(owner)
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    /// Submit dispatches through `executor` (`DESIGN.md` §11.1, issue #115).
    ///
    /// Narrow on purpose: `commands` is `pub(crate)` and stays that way, so a
    /// harness can install an executor without reaching the command-buffer
    /// lifecycle around it. `ExecutorSlot::Classical` is the default and is the
    /// same code that ran before the seam existed (§11.1b), so *not* calling
    /// this leaves the path unchanged rather than equivalent.
    pub fn set_executor(&self, executor: Arc<crate::kernels::metal::ExecutorSlot>) {
        self.commands.set_executor(executor);
    }

    /// Tell the executor the model has reached graph node `index`
    /// (`DESIGN.md` §13.5m, issue #420 stage 3).
    ///
    /// A model calls this at each node boundary of its forward pass -- for a
    /// decoder stack, once per layer. Under the default cut policy it is a
    /// no-op that takes no lock and performs no store, so a model that calls it
    /// unconditionally costs nothing when the arm is not selected; under
    /// `CutPolicy::NodeIndex` it is what puts the command-buffer seam on a
    /// graph position instead of on a dispatch count.
    ///
    /// Narrow on purpose, exactly as `set_executor` above is: `commands` is
    /// `pub(crate)` and stays that way. The model announces *where it is*; it
    /// does not reach the command-buffer lifecycle around it.
    #[inline]
    pub fn mark_graph_node(&self, index: usize) {
        self.commands.mark_graph_node(index);
    }

    /// The command-buffer cut policy in force (`DESIGN.md` §13.5m).
    ///
    /// Read back from the mechanism that decides it rather than from the flag
    /// that selected it, per §7.1a-i: a `RESULT` line that renders an axis from
    /// its own parse of a flag can state an arm the engine is not running, and
    /// a wrong arm compares *equal* to real runs of another configuration where
    /// an absent one could not.
    pub fn cut_policy(&self) -> crate::kernels::metal::CutPolicy {
        self.commands.cut_policy()
    }

    /// Admit computing at `dtype` for the use `what` names: a type this GPU
    /// only emulates is refused unless allowed (§15.2a).
    pub fn admit_dtype(&self, dtype: DType, what: &str) -> Result<()> {
        match crate::kernels::metal::arms::Feature::emulable_dtype(dtype) {
            None => Ok(()),
            Some(feature) => Ok(self
                .emulated
                .admit(feature, || format!("{what} `{feature}`"))
                .map_err(MetalError::from)?),
        }
    }

    /// Allow these emulated types on this device as well as the ones the
    /// tuning group allows — a test's opt-in, scoped to its own device.
    pub fn allow_emulated(&self, features: crate::kernels::metal::arms::Features) {
        self.emulated.allow(features)
    }

    /// The device's emulated-arm gate: what it allows, the warnings printed.
    pub fn emulated_arms(&self) -> &crate::kernels::metal::arms::EmulatedArms {
        &self.emulated
    }

    /// The dispatch period the counter cut uses (`DESIGN.md` §13.5m).
    ///
    /// Reported alongside the policy because a counter cut at 50 and one at 55
    /// are different arms with different session counts, and #420 stage 3's
    /// control is a counter arm at a period chosen to match the node arm's
    /// count. A run that recorded only `Counter` would not be separable from
    /// the shipping default.
    pub fn compute_per_buffer(&self) -> usize {
        self.commands.compute_per_buffer()
    }

    /// The pipelined loop's cut period (`Commands::pipelined_compute_per_buffer`).
    pub fn pipelined_compute_per_buffer(&self) -> usize {
        self.commands.pipelined_compute_per_buffer()
    }

    /// Switch the counter cut's period (`Commands::set_compute_per_buffer`).
    pub fn set_compute_per_buffer(&self, period: usize) {
        self.commands.set_compute_per_buffer(period)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>> {
        let owner = self.allocation_namespace_anchor()?;
        let mut command_encoder = self
            .commands
            .blit_command_encoder()
            .map_err(MetalError::from)?;
        command_encoder
            .retain_namespace(owner)
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    /// The command stream, for a subsystem that orders its own queue against
    /// it (`kernels::metal::io_queue`).
    pub fn commands(&self) -> &Arc<Commands> {
        &self.commands
    }

    /// This device's own dispatch census: a window on it counts the
    /// dispatches encoded here and no other device's
    /// (`profile::DeviceCensus`).
    pub fn dispatch_census(&self) -> &Arc<crate::kernels::metal::profile::DeviceCensus> {
        self.commands.dispatch_census()
    }

    /// This device's own dispatch trace: a window on it records the
    /// dispatches encoded here and no other device's (`trace::DeviceTrace`).
    pub fn dispatch_trace(&self) -> &Arc<crate::kernels::metal::trace::DeviceTrace> {
        self.commands.dispatch_trace()
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        self.commands
            .wait_until_completed()
            .map_err(MetalError::from)?;
        // No sweep here any more. Buffers freed while this was waiting have
        // already returned themselves to their pool, so there is nothing to
        // discover -- and destroying them, which is what the sweep did, only
        // forced the next allocation to re-create them.
        //
        // Still a drain, though, and it is not a scan: everything is complete
        // now by definition, so this hands back whatever was waiting on the GPU
        // without depending on a completion handler having been delivered yet.
        // Metal dispatches those asynchronously, so a caller that synchronizes
        // and immediately allocates would otherwise miss buffers that are
        // provably free.
        self.drain_completed_buffers();
        Ok(())
    }

    /// Returns buffers whose GPU work has finished to their free lists.
    fn drain_completed_buffers(&self) {
        self.buffers.drain_completed();
        self.private_buffers.drain_completed();
    }

    /// Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks.
    pub fn flush_and_wait_current(&self) -> Result<()> {
        self.commands
            .flush_and_wait_current()
            .map_err(MetalError::from)?;
        self.drain_completed_buffers();
        Ok(())
    }

    /// Commit the current command buffer and return a marker for it
    /// (`Commands::flush_marker`).
    pub fn flush_marker(&self) -> Result<CommandMarker> {
        Ok(self.commands.flush_marker().map_err(MetalError::from)?)
    }

    /// Whether `marker`'s buffer has already retired, without waiting.
    pub fn marker_complete(&self, marker: &CommandMarker) -> bool {
        self.commands.marker_complete(marker)
    }

    /// Dispatches encoded into the open buffer and not yet committed
    /// (`Commands::open_dispatches`).
    pub fn open_dispatches(&self) -> usize {
        self.commands.open_dispatches()
    }

    /// The last committed, unpruned buffer (`Commands::last_committed_marker`).
    pub fn last_committed_marker(&self) -> Result<Option<CommandMarker>> {
        Ok(self
            .commands
            .last_committed_marker()
            .map_err(MetalError::from)?)
    }

    /// The EOS gate (lloom #735 step 4): see `kernels::metal::DispatchGate`.
    /// Refused with an executor installed — the gate covers directly encoded
    /// dispatches, not an ICB replay.
    pub fn eos_gate_begin(&self) -> Result<()> {
        if !self.commands.executor().is_classical() {
            crate::bail!(
                "eos gate: an executor is installed; the gate applies to directly encoded \
                 dispatches only"
            );
        }
        self.commands
            .eos_gate()
            .begin(&self.device)
            .map_err(MetalError::from)?;
        // Bind before any gated variant can be selected (`set_eos_gated`
        // last): an open encoder gets the flag now, later ones at creation.
        self.commands
            .bind_eos_gate_current()
            .map_err(MetalError::from)?;
        self.kernels.set_eos_gated(true);
        Ok(())
    }

    /// The gate's cores arm for the armed run: the quantized mat-vec cores
    /// load their gated variant too (`Kernels::set_eos_gate_cores`). Cleared
    /// by [`Self::eos_gate_end`].
    pub fn eos_gate_cores(&self, on: bool) {
        self.kernels.set_eos_gate_cores(on);
    }

    /// See `Device::eos_gate_step`. Offsets in bytes.
    pub fn eos_gate_step(
        &self,
        token: &crate::MetalStorage,
        token_offset: usize,
        eos_ids: &crate::MetalStorage,
        eos_offset: usize,
        n_eos: usize,
    ) -> Result<()> {
        let encoder = self.command_encoder()?;
        self.commands
            .eos_gate()
            .step(
                &self.device,
                &encoder,
                &self.kernels,
                token.buffer(),
                token_offset,
                eos_ids.buffer(),
                eos_offset,
                n_eos,
            )
            .map_err(MetalError::from)?;
        Ok(())
    }

    /// See `Device::argmax_publish`. Storages with byte offsets; `n` logits of
    /// `dtype` (F16 or F32). The gate's flag is written iff the gate is
    /// armed, with a buffers-scope barrier behind the dispatch so the gated
    /// variants (which read it outside the hazard tracker) see it.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_publish(
        &self,
        logits: (&crate::MetalStorage, usize),
        n: usize,
        token_slot: (&crate::MetalStorage, usize),
        history_slot: (&crate::MetalStorage, usize),
        eos_ids: (&crate::MetalStorage, usize, usize),
        step: (&crate::MetalStorage, usize),
        generation: u32,
    ) -> Result<()> {
        let kernel = match logits.0.dtype {
            DType::F16 => "argmax_publish_f16",
            DType::F32 => "argmax_publish_f32",
            other => crate::bail!("argmax_publish: logits must be F16 or F32, got {other:?}"),
        };
        let gate = self.commands.eos_gate();
        let armed = gate.armed_flag();
        let dummy: &crate::kernels::metal::Buffer = step.0.buffer();
        let (flag, n_eos) = match armed {
            Some(flag) => (flag, eos_ids.2),
            None => (dummy, 0),
        };
        fn bo(s: &crate::MetalStorage, off: usize) -> crate::kernels::BufferOffset<'_> {
            crate::kernels::BufferOffset {
                buffer: s.buffer(),
                offset_in_bytes: off,
            }
        }
        let encoder = self.command_encoder()?;
        // SAFETY: the storages are the caller's (binding authority, `mod.rs`
        // head). The gate's flag is the device's own: the host writes it only
        // in `EosGate::begin`, before the round's first dispatch; ordering that
        // write against an earlier round still in flight is the decode loop's.
        unsafe {
            crate::kernels::call_argmax_publish(
                &self.device,
                &encoder,
                &self.kernels,
                kernel,
                bo(logits.0, logits.1),
                n,
                bo(token_slot.0, token_slot.1),
                bo(history_slot.0, history_slot.1),
                bo(eos_ids.0, eos_ids.1),
                n_eos,
                flag,
                bo(step.0, step.1),
                generation,
            )
        }
        .map_err(MetalError::from)?;
        if armed.is_some() {
            use crate::kernels::utils::EncoderProvider;
            let guard = &encoder;
            let enc = guard.encoder();
            let enc: &crate::kernels::metal::ComputeCommandEncoder = enc.as_ref();
            enc.insert_memory_barrier();
            gate.note_step();
        }
        Ok(())
    }

    /// See `Device::eos_gate_end`.
    /// The speculative round's readback as a publish: `n` u32 words of `src`
    /// copied into the caller's CPU-mapped `out` (`words_publish`), then
    /// `generation` stamped into `step` by a `ring_stamp` dispatch bound to
    /// `out`. The host polls the stamp and reads the words in place — no
    /// completion-handler wait, no `to_cpu`. The stamp is a dispatch of its
    /// own because the command stream's ordering of a dispatch after the
    /// last writer of what it binds is what a host observes; the kernel's
    /// in-dispatch barrier orders its stores for the GPU only
    /// (`docs/dispatch.md` §11.5z-xvi).
    pub fn words_publish(
        &self,
        src: (&crate::MetalStorage, usize),
        n: usize,
        out: (&crate::MetalStorage, usize),
        step: (&crate::MetalStorage, usize),
        generation: u32,
    ) -> Result<()> {
        fn bo(s: &crate::MetalStorage, off: usize) -> crate::kernels::BufferOffset<'_> {
            crate::kernels::BufferOffset {
                buffer: s.buffer(),
                offset_in_bytes: off,
            }
        }
        {
            let encoder = self.command_encoder()?;
            // SAFETY: storage-owned buffers (binding authority, `mod.rs` head).
            unsafe {
                crate::kernels::call_words_publish(
                    &self.device,
                    &encoder,
                    &self.kernels,
                    bo(src.0, src.1),
                    n,
                    bo(out.0, out.1),
                    bo(step.0, step.1),
                    0,
                )
            }
            .map_err(MetalError::from)?;
        }
        self.ring_stamp(step, generation, [out, out])
    }

    /// See `Device::ring_stamp`. Storages with byte offsets.
    pub fn ring_stamp(
        &self,
        step: (&crate::MetalStorage, usize),
        generation: u32,
        certified: [(&crate::MetalStorage, usize); 2],
    ) -> Result<()> {
        fn bo(s: &crate::MetalStorage, off: usize) -> crate::kernels::BufferOffset<'_> {
            crate::kernels::BufferOffset {
                buffer: s.buffer(),
                offset_in_bytes: off,
            }
        }
        let encoder = self.command_encoder()?;
        // SAFETY: storage-owned buffers (binding authority, `mod.rs` head).
        unsafe {
            crate::kernels::call_ring_stamp(
                &self.device,
                &encoder,
                &self.kernels,
                bo(step.0, step.1),
                generation,
                certified.map(|(s, off)| bo(s, off)),
            )
        }
        .map_err(MetalError::from)?;
        Ok(())
    }

    /// Whether the EOS gate is armed on this device's commands.
    pub fn eos_gate_armed(&self) -> bool {
        self.commands.eos_gate().is_armed()
    }

    /// Disarm and report. The run's gated work completes first, under its
    /// own flag: `EosGate::end`'s report and the next `begin`'s clear both
    /// need the run drained, and a run that ended early (an error out of the
    /// decode loop, its guard disarming here) has not. Its queued forwards
    /// would otherwise run after the next run's host store cleared the flag,
    /// take their plain path and write past the EOS. On the normal path the
    /// loop has drained and the wait finds nothing in flight. A command
    /// buffer that failed is recorded for the next wait to report.
    pub fn eos_gate_end(&self) -> crate::EosGateReport {
        let _ = self.wait_until_completed();
        let cores = self.kernels.eos_gate_cores();
        self.kernels.set_eos_gated(false);
        self.kernels.set_eos_gate_cores(false);
        crate::EosGateReport {
            cores,
            ..self.commands.eos_gate().end()
        }
    }

    /// The adaptive commit (lloom #735): see `Commands::flush_marker_if_started`.
    pub fn flush_marker_if_started(&self) -> Result<Option<CommandMarker>> {
        Ok(self
            .commands
            .flush_marker_if_started()
            .map_err(MetalError::from)?)
    }

    /// Wait on `marker`'s buffer only — later commits keep running — then
    /// return completed buffers to their pools, as every wait here does.
    pub fn wait_marker(&self, marker: &CommandMarker) -> Result<()> {
        self.commands
            .wait_marker(marker)
            .map_err(MetalError::from)?;
        self.drain_completed_buffers();
        Ok(())
    }

    /// Recovery-only registered-prefix data boundary. M4 joins prior native
    /// queue work through its front receipt plus own data. M3 joins registered
    /// ordered callbacks, not untracked raw submissions.
    /// Error/refusal is not permission to reset mapped scratch or release
    /// unknown native custody. Later commits remain outside this receipt.
    pub fn wait_marker_prefix_data(&self, marker: &CommandMarker) -> Result<()> {
        self.commands
            .wait_marker_prefix_data(marker)
            .map_err(MetalError::from)?;
        self.drain_completed_buffers();
        Ok(())
    }

    /// Commit without waiting, prune finished command buffers, and return the
    /// number still in flight.
    ///
    /// The non-blocking counterpart of [`Self::flush_and_wait_current`], for
    /// the polled token path (`DESIGN.md` §11.5g). The pool drain is kept
    /// because the completion handlers that free buffers are delivered
    /// asynchronously, so a caller that never blocks would otherwise leave
    /// buffers parked against an epoch that has in fact retired.
    pub fn flush_and_prune(&self) -> Result<usize> {
        let in_flight = self.commands.flush_and_prune().map_err(MetalError::from)?;
        self.drain_completed_buffers();
        Ok(in_flight)
    }

    /// The completion-wait arm in force, as its `RESULT`-line name
    /// (`DESIGN.md` §11.5o, issue #559).
    ///
    /// **Read back from the resolved value**, never echoed from the string that
    /// selected it -- §7.1a-i's `AttnImpl=Generic` defect is what that
    /// distinction exists to prevent, and a wrong arm compares equal to every
    /// run of that arm where an absent one compares equal to nothing.
    pub fn completion_wait_name(&self) -> &'static str {
        self.commands.completion_wait().name()
    }

    /// `(signals encoded, polls performed)` by the async completion arms.
    ///
    /// The engagement pair. `(0, 0)` under an async arm means the arm did not
    /// run: #500 measured that a mechanism with no production callers produces
    /// a clean null indistinguishable from a real one, and a counter is the
    /// only thing that separates them.
    pub fn completion_counts(&self) -> (u64, u64) {
        self.commands.completion_counts()
    }

    /// The buffer-alignment arm in force (`DESIGN.md` §9.2, issue #586).
    ///
    /// **Read back from the resolved value**, like
    /// [`Self::completion_wait_name`] above and for the same reason: the arm is
    /// whatever `buf_size` actually rounds to, so a harness that rendered its
    /// own flag could report an alignment the process did not take. That is
    /// §7.1a-i's present-and-wrong case, which is the unsafe direction — a
    /// wrong arm compares **equal** to every real run of that arm.
    ///
    /// It is an associated function rather than a method because the arm is a
    /// property of the process (one `OnceLock`), not of a device: two
    /// `MetalDevice`s in one process cannot disagree about it, and pretending
    /// otherwise would invite a caller to ask the wrong one.
    pub fn buffer_alignment() -> usize {
        buffer_alignment()
    }

    /// Whether an `MTLSharedEvent` was actually allocated for the selected arm.
    ///
    /// Separates *"an async arm was selected"* from *"an async arm is running"*:
    /// a device that declined the event leaves this false and every wait takes
    /// the blocking fallback, which is correct but is not what was asked for.
    pub fn completion_event_present(&self) -> bool {
        self.commands.completion_event_present()
    }

    /// Notification blocks that fired, for the listener arm (`DESIGN.md`
    /// §11.5u, issue #583).
    ///
    /// **The listener arm's only engagement proof**, and the reason it is a
    /// third counter rather than a third element of
    /// [`Self::completion_counts`]: that pair reads `(nonzero, 0)` on this arm
    /// whether the block delivered or the wait timed out and blocked instead,
    /// because the arm polls zero times either way. A listener that has stopped
    /// firing computes the right answer and moves no digest, so nothing else in
    /// the tree can see it (§2.4, #500).
    pub fn completion_fires(&self) -> u64 {
        self.commands.completion_fires()
    }

    /// Whether an `MTLSharedEventListener` was allocated for the selected arm.
    ///
    /// [`Self::completion_event_present`]'s counterpart one level in, and for
    /// the same reason: *"the listener arm was selected"* and *"the listener
    /// arm can run"* are different facts.
    pub fn completion_listener_present(&self) -> bool {
        self.commands.completion_listener_present()
    }

    /// Waits that found the command buffer already complete (`DESIGN.md`
    /// §11.5u, issue #583).
    ///
    /// **Measured 0 on the decode path**, and kept for what it excluded: it
    /// falsified the first explanation for `completion_fires` sitting an order
    /// of magnitude below `completion_signals`. The two count different things
    /// -- commits against waits, ~13.5:1 under `CutPolicy::Counter(50)` -- so
    /// the ratio is arithmetic rather than a failure rate.
    pub fn completion_already_complete(&self) -> u64 {
        self.commands.completion_already_complete()
    }

    /// Write `byte` over every Shared-pool buffer the moment it becomes free,
    /// or stop -- the #706 GPU-side poison, reachable from a harness.
    /// A command buffer still reading a "free" buffer then reads the sentinel
    /// instead of the bytes its last owner left, which turns a rare stale
    /// read into a visible one. Inert (a note on stderr) without the
    /// `fuzzing` feature.
    pub fn set_free_poison(&self, byte: Option<u8>) {
        #[cfg(feature = "fuzzing")]
        {
            // The Shared pool only: a Private buffer has no host mapping
            // (`contents()` is null), and the poison is a host write.
            self.buffers.set_free_poison(byte);
        }
        #[cfg(not(feature = "fuzzing"))]
        {
            let _ = byte;
            eprintln!("candle: set_free_poison is inert without the `fuzzing` feature");
        }
    }

    /// The clock the buffer pools decide reuse against (`DESIGN.md` §6.3b).
    ///
    /// A page-pool caller quarantines released KV pages against
    /// `current_epoch()` and recycles them once `completed_count()` has passed
    /// it — the same two numbers the pools use, so a page cannot be handed to
    /// a second sequence while a command buffer still reads it (§10.3h).
    pub fn gpu_clock(&self) -> Arc<crate::kernels::metal::GpuClock> {
        self.commands.clock()
    }

    /// The kind of command buffer this device's queue creates.
    pub fn command_buffer_kind(&self) -> crate::kernels::metal::CommandBufferKind {
        self.commands.kind()
    }

    /// What the unretained kind's hold has taken (`in_flight.rs`); `None` on
    /// the retaining kind, which has no hold.
    pub fn in_flight_hold_counts(&self) -> Option<crate::kernels::metal::HoldCounts> {
        self.commands.in_flight_hold().map(|h| h.counts())
    }

    /// The memory ledger: committed bytes beside `currentAllocatedSize`, the
    /// tenants' claims, the quota (`profiles-and-memory-design.md` §4.2).
    pub fn ledger(&self) -> &Arc<crate::kernels::metal::ledger::Ledger> {
        &self.ledger
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Registers buffers in the device's residency set, keeping them
    /// permanently GPU-resident instead of paying per-command-buffer residency
    /// bookkeeping. Useful for buffers candle did not allocate, e.g.
    /// `newBufferWithBytesNoCopy` views over an mmap'd weights file. No-op on
    /// systems without residency-set support.
    pub fn register_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.insert_batch(bufs);
    }

    /// Unregisters buffers previously passed to `register_buffers`, releasing
    /// the set's retain so they can be deallocated. Only unregister buffers
    /// you registered yourself, after GPU work referencing them has completed.
    pub fn unregister_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.remove_batch(bufs);
    }

    /// Returns a builder for buffer allocation. See `BufferBuilder`.
    pub fn new_buffer_builder(&self) -> BufferBuilder<'_> {
        BufferBuilder::new(self)
    }

    /// The installed arena, if any. Cheap: one lock and an `Option` clone.
    fn arena_slot(&self) -> Option<Arena> {
        self.arena.read().clone()
    }

    /// The installed session region, if any. Cheap, on `arena_slot`'s argument.
    fn session_region_slot(&self) -> Option<SessionRegion> {
        self.session_region.read().clone()
    }

    /// Install a session region serving the ordinals the arena declines as
    /// session state (#47, `DESIGN.md` §9.2x).
    ///
    /// `reservations` is `(ordinal, bytes)`: the allocation ordinal within a
    /// decode step, and the bytes that ordinal's tenant needs **at full
    /// capacity**. That second figure is ggml's `reserve` (§9.2t): a tenant
    /// growing by a fixed stride per token to a known capacity needs
    /// `stride × capacity` bytes and never more, so reserving it once means no
    /// allocation on any later token.
    ///
    /// # This is additive, exactly as the arena is (§9.2a)
    ///
    /// Nothing about the pool changes and no caller releases anything. An
    /// ordinal with no reservation, or a request past its reservation, falls
    /// through to the pool as before — which costs coverage, never correctness,
    /// and is the direction a wrong reservation must fail in.
    ///
    /// One `MTLBuffer`, registered in the residency set once, on §9.2's
    /// *"residency is a CPU-side fact established once"*: the whole point is
    /// that the per-token path stops inserting.
    pub fn install_session_region(&self, reservations: &[(usize, usize)]) -> Result<usize> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .install_session_region_in(reservations, Some(&scope.context));
        }
        self.install_session_region_in(reservations, None)
    }

    fn install_session_region_in(
        &self,
        reservations: &[(usize, usize)],
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<usize> {
        let plan = RegionPlan::checked_reserve(reservations).map_err(MetalError::Message)?;
        let bytes = plan.bytes().max(1);
        let base = crate::residency_class::residency_class_scope(
            crate::residency_class::ResidencyClass::Kv,
            || Self::allocate_backing(&self.backings.resident_private, bytes, context),
        )?;
        let region = SessionRegion::new(base.buffer, plan)
            .map_err(|e| MetalError::Message(format!("session region rejected: {e}")))?;
        let previous = {
            let mut slot = self.session_region.write();
            *self.allocation_namespace.session_generation.lock() = Arc::new(());
            slot.replace(region)
        };
        if let Some(previous) = previous {
            self.residency_set.remove(previous.base());
            self.hold_in_flight(previous.base());
        }
        Ok(bytes)
    }

    /// Remove the session region. Subsequent allocations take the pool path.
    ///
    /// Ordered as [`Self::clear_arena`] is: the base leaves the residency set
    /// **while the region still owns it**, so `removeAllocation` names an object
    /// that provably still exists (§6.3c, #489).
    pub fn clear_session_region(&self) {
        let previous = {
            let mut slot = self.session_region.write();
            *self.allocation_namespace.session_generation.lock() = Arc::new(());
            slot.take()
        };
        if let Some(region) = previous {
            self.residency_set.remove(region.base());
            self.hold_in_flight(region.base());
        }
    }

    /// The session region's counters, if one is installed.
    pub fn session_region_counters(&self) -> Option<RegionCounters> {
        self.session_region_slot().map(|r| r.counters())
    }

    /// Bytes the installed session region holds.
    pub fn session_region_bytes(&self) -> Option<usize> {
        self.session_region_slot().map(|r| r.bytes())
    }

    /// How many tenants the installed session region reserves for.
    ///
    /// `None` when none is installed, and `Some(0)` when one is installed with
    /// nothing to serve — two states a harness must be able to tell apart, since
    /// an empty region and an absent one produce identical counters otherwise
    /// (#500's lesson: a clean null and an unwired mechanism look the same).
    pub fn session_region_tenants(&self) -> Option<usize> {
        self.session_region_slot().map(|r| r.tenants())
    }

    /// Install an activation arena built from `plan`, replacing any previous
    /// one.
    ///
    /// Allocates one `MTLBuffer` of the plan's size and registers it in the
    /// residency set -- §9.2's "residency is a CPU-side fact established once":
    /// one buffer in the set rather than 674.
    ///
    /// The arena is *additive* (§9.2a). Nothing about the pool changes, no
    /// caller is asked to release anything, and removing the arena with
    /// [`Self::clear_arena`] returns the device to the classical path exactly.
    /// Whether copies and fills must avoid the blit encoder (see `base_mode`).
    #[inline]
    pub fn base_mode(&self) -> bool {
        self.base_mode.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn set_base_mode(&self, on: bool) {
        self.base_mode
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn install_arena(&self, plan: StepPlan, layout: ArenaLayout) -> Result<()> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .install_arena_in(plan, layout, Some(&scope.context));
        }
        self.install_arena_in(plan, layout, None)
    }

    fn install_arena_in(
        &self,
        plan: StepPlan,
        layout: ArenaLayout,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<()> {
        let bytes = plan.checked_base_bytes().map_err(MetalError::Message)?;
        self.set_base_mode(true);
        // The arena is one pointer for every activation, so the hazard tracker
        // must key on ranges to see only true edges (§9.2e/§9.2f, route (a)).
        // Selected here, by the mechanism that needs it, unless a harness
        // pinned the key on purpose. Takes effect at the next encoder session,
        // which is before any dispatch binds an arena slot.
        crate::kernels::metal::set_hazard_key_unless_explicit(
            crate::kernels::metal::HazardKey::Range,
        );
        // `bump_capacity`, not `arena_bytes`: the two differ when the last
        // slot's size is not a multiple of the 128 B alignment, and a GPU bump
        // allocator rounds every request including the last (issue #70). Sizing
        // to the larger figure costs at most 127 bytes of tail and lets either
        // offset source serve every ordinal; sizing to the smaller one makes the
        // GPU path decline an ordinal that fits the plan perfectly, which is a
        // silent loss of coverage rather than an error.
        let base = crate::residency_class::residency_class_scope(
            crate::residency_class::ResidencyClass::Scratch,
            || Self::allocate_backing(&self.backings.resident_private, bytes, context),
        )?;
        let arena = Arena::new(base.buffer, plan, layout)
            .map_err(|e| MetalError::Message(format!("arena plan rejected: {e}")))?;
        let previous = self.arena.write().replace(arena);
        let previous_reservation =
            std::mem::replace(&mut *self.arena_reservation.lock(), base.reservation);
        if let Some(previous) = previous {
            self.residency_set.remove(previous.base());
            self.hold_in_flight(previous.base());
        }
        drop(previous_reservation);
        Ok(())
    }

    /// Compute this arena's offsets on the GPU and adopt them if they agree
    /// with the plan (`DESIGN.md` §9.2d, issue #70).
    ///
    /// Dispatches the bump allocator over the installed arena's plan, reads the
    /// offsets back, and hands them to
    /// [`Arena::adopt_gpu_offsets`](crate::kernels::metal::Arena::adopt_gpu_offsets),
    /// which verifies them element-wise before switching. A disagreement is an
    /// error and the arena keeps binding through the CPU plan.
    ///
    /// # Why this is a one-off rather than per step
    ///
    /// The plan is fixed once recorded and the request sizes never change, so
    /// the allocator computes the same table every step. Running it per token
    /// would burn a dispatch and a readback to re-derive a constant.
    ///
    /// **Nothing here is per step, and this comment used to claim otherwise**
    /// (#526). It read *"what **is** per step is the reset"*; measured on the
    /// call graph, `call_arena_reset` has no caller outside tests, and it
    /// cannot acquire one as this path is built: the `ArenaCursor` below is a
    /// local, dropped when this function returns, and `adopt_gpu_offsets`
    /// copies the verified table into the plan that binding then reads. **There
    /// is no cursor left to reset.**
    ///
    /// The reset becomes per step under §11.5's GPU-driven loop, where an
    /// encoding kernel consumes offsets without the readback below and the
    /// cursor is genuinely live across steps. Until then it is a tested
    /// capability with no consumer, which `arena_alloc.rs` now says in the one
    /// place a reader of that kernel will look.
    ///
    /// # The readback, stated plainly
    ///
    /// The offsets cross back to the host because `setBuffer_offset_atIndex` is
    /// a CPU call -- a classical dispatch cannot consume a GPU-computed offset
    /// any other way. That is the boundary §11.3c describes, not a shortcut:
    /// consuming an offset without a round-trip requires an ICB command written
    /// by an encoding kernel, and the ICB executor is out of scope here. Doing
    /// it once at install rather than per token is what keeps the cost off the
    /// per-token path entirely.
    pub fn install_gpu_arena_offsets(&self) -> Result<usize> {
        use crate::kernels::call_arena_bump;

        let arena = self
            .arena_slot()
            .ok_or_else(|| MetalError::Message("no arena installed".to_string()))?;
        let plan = arena.plan_snapshot();
        let sizes = plan.request_sizes();
        let capacity = plan.bump_capacity().max(plan.arena_bytes()).max(1);

        let cursor = ArenaCursor::new(&self.device, &sizes, capacity)
            .map_err(|e| MetalError::Message(format!("arena cursor: {e}")))?;
        {
            let guard = self.commands.command_encoder().map_err(MetalError::from)?;
            // SAFETY: the cursor's three buffers are created above, written by
            // no one else, and outlive the wait below.
            unsafe {
                call_arena_bump(
                    &self.device,
                    &guard,
                    &self.kernels,
                    cursor.cursor_buffer(),
                    cursor.sizes_buffer(),
                    cursor.offsets_buffer(),
                    cursor.len(),
                    cursor.capacity(),
                )
            }
            .map_err(MetalError::from)?;
        }
        // The offsets live in shared storage, so they are only meaningful once
        // the command buffer carrying the allocator has completed. Reading them
        // sooner is a race the host cannot detect.
        self.wait_until_completed()?;

        Ok(arena
            .adopt_gpu_offsets(&cursor.offsets())
            .map_err(|e| MetalError::Message(format!("GPU offsets rejected: {e}")))?)
    }

    /// Remove the arena. Subsequent allocations all take the pool path.
    ///
    /// # The residency removal, and why it is ordered this way (issue #489)
    ///
    /// [`Self::install_arena`] registers the base in the residency set, so
    /// removing the arena has to unregister it -- otherwise the device retains
    /// an allocation the arena has dropped, which is `DESIGN.md` §6.7 L4 at the
    /// arena and the mechanism §6.3c records **panicking the machine**.
    ///
    /// The base is taken out of the set **while the arena still owns it**, so
    /// `removeAllocation` names an object that provably still exists -- the same
    /// ordering `PoolInner::drain_completed` uses when it tells the eviction
    /// observer before destroying, and the ordering §6.3c's whole finding is
    /// about. Dropping first and removing after would be the exact shape that
    /// aborts.
    ///
    /// # Why this is here and not in `impl Drop for Arena`
    ///
    /// `Arena` is `Clone` and holds no residency handle, so a `Drop` there
    /// would have to acquire one and would fire on whichever clone happened to
    /// go last -- §6.3d's finding in a second place: the participant that owns
    /// the set is the device, so the device is where the removal belongs.
    ///
    /// Nothing was calling this until #487's probe, which is why the omission
    /// survived from #69 (§15.2 #11b, on a lifetime path rather than a
    /// capability).
    pub fn clear_arena(&self) {
        let mut slot = self.arena.write();
        if let Some(arena) = slot.as_ref() {
            self.residency_set.remove(arena.base());
            self.hold_in_flight(arena.base());
        }
        *slot = None;
        *self.arena_reservation.lock() = None;
        drop(slot);
        if !self.pass_plans_enabled()
            && self.arena_recorder.lock().is_none()
            && self.arena_auto.lock().is_none()
        {
            self.set_base_mode(false);
        }
    }

    pub fn arena(&self) -> Option<Arena> {
        self.arena_slot()
    }

    /// Mark the start of a decode step, resetting the arena's allocation
    /// ordinal.
    ///
    /// The ordinal is what ties an allocation to a slot, so it has to restart at
    /// the same point every token. No-op when no arena is installed and no
    /// recording is armed.
    pub fn begin_decode_step(&self) {
        {
            let mut g = self.arena_auto.lock();
            if let Some(auto) = g.as_mut() {
                if auto.step == auto.skip {
                    self.begin_arena_recording();
                } else if auto.step > auto.skip && auto.step < auto.skip + auto.record_steps {
                    self.next_arena_recording_step();
                }
            }
        }
        if let Some(a) = self.arena_slot() {
            a.begin_step();
        }
    }

    /// Record the arena from the decode steps the loop is about to run, and
    /// install it when the recording is complete.
    ///
    /// `skip` steps pass first, `record_steps` are recorded, and the arena is
    /// installed at the end of the last recorded step; every step after that
    /// is served from it. Takes effect through [`Self::begin_decode_step`] and
    /// [`Self::finish_decode_step`], so a loop that brackets its steps with
    /// those needs nothing else. Replaces an earlier arming.
    pub fn arm_arena_recording(&self, layout: ArenaLayout, skip: usize, record_steps: usize) {
        self.set_base_mode(true);
        {
            let mut g = self.arena_auto.lock();
            *g = Some(ArenaAutoRecord {
                layout,
                skip,
                record_steps: record_steps.max(2),
                step: 0,
            });
        }
        {
            let mut r = self.arena_rerecord.lock();
            *r = Some((layout, record_steps.max(2)));
        }
        {
            let mut r = self.arena_auto_report.lock();
            *r = None;
        }
    }

    /// Arenas dropped and recorded again after a step drifted from their plan
    /// (`Arena::take_drifted`).
    pub fn arena_rerecordings(&self) -> u64 {
        self.arena_rerecordings
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What an armed recording installed, once it has.
    pub fn arena_auto_report(&self) -> Option<ArenaAutoReport> {
        *self.arena_auto_report.lock()
    }

    /// [`Self::end_decode_step`], plus the install an armed recording is due
    /// (`arm_arena_recording`). What a decode loop calls after a step's forward
    /// pass; the executor's step is closed separately (`end_executor_step`),
    /// after everything the step encodes.
    pub fn finish_decode_step(&self) -> Result<()> {
        self.end_decode_step();
        // A step that drifted from the plan is served no further (the arena
        // deactivated itself at the drift) and the plan is dropped here: the
        // sequence it describes no longer runs, and the recording is re-armed
        // over the steps that follow -- none skipped, because the drifted
        // sequence is the steady state now. Each re-recording is counted, so
        // a run whose plan moved twice says so beside its coverage.
        if let Some(arena) = self.arena_slot() {
            if arena.take_drifted() {
                let rearm = *self.arena_rerecord.lock();
                self.clear_arena();
                if let Some((layout, record_steps)) = rearm {
                    // The drifted step is the record's step 0 (the count
                    // below advances it), so `skip: 1` records from the next
                    // step on -- nothing but the step that drifted is skipped.
                    {
                        let mut g = self.arena_auto.lock();
                        *g = Some(ArenaAutoRecord {
                            layout,
                            skip: 1,
                            record_steps,
                            step: 0,
                        });
                    }
                    self.arena_rerecordings
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        // `parking_lot` guards are infallible; an IIFE returning `Option` keeps
        // the `?` early-return the `and_then` closure used.
        let due = (|| {
            let mut g = self.arena_auto.lock();
            let auto = g.as_mut()?;
            let step = auto.step;
            auto.step += 1;
            if step + 1 == auto.skip + auto.record_steps {
                let layout = auto.layout;
                *g = None;
                Some((layout, step))
            } else {
                None
            }
        })();
        if let Some((layout, step)) = due {
            let plan = self.finish_arena_recording(layout).ok_or_else(|| {
                MetalError::Message(
                    "arena recording over the decode steps produced no allocations".to_string(),
                )
            })?;
            let (covered, total) = plan.covered();
            self.install_arena(plan, layout)?;
            {
                let mut r = self.arena_auto_report.lock();
                *r = Some(ArenaAutoReport {
                    layout,
                    covered,
                    total,
                    installed_at_step: step,
                });
            }
        }
        Ok(())
    }

    /// Mark the end of a decode step. Allocations outside a step take the pool
    /// path, which is what keeps prefill and setup on the classical route.
    pub fn end_decode_step(&self) {
        if let Some(a) = self.arena_slot() {
            a.end_step();
        }
    }

    /// Turn the pass-keyed plan cache on (`DESIGN.md` §9.2aa). Off, the
    /// pass brackets are the decode-step brackets they always were.
    pub fn enable_pass_plans(&self, layout: ArenaLayout) {
        self.set_base_mode(true);
        self.pass_state
            .fetch_or(0b01, std::sync::atomic::Ordering::Relaxed);
        let mut g = self.pass_plans.lock();
        if g.is_none() {
            *g = Some(PassPlans::new(layout));
        }
    }

    /// Resolve the checked active payer for receiver-bound planner queries and
    /// configuration. Same-device input origins do not select the output payer.
    /// Outside a scope a legacy receiver is unchanged; accounted receivers refuse
    /// missing/cancelled scopes. Retain this view across related query/config calls.
    pub fn execution_device(&self) -> Result<Self> {
        Ok(super::allocation_scope::active(self)?
            .map_or_else(|| self.clone(), |scope| scope.view.clone()))
    }

    /// Receiver-bound query. Scoped callers first resolve `execution_device`.
    pub fn pass_plans_enabled(&self) -> bool {
        self.pass_plans.lock().is_some()
    }

    /// Set the pass-plan cache aside until the returned guard drops, for
    /// passes that belong to no sequence (a load warmup): they run on the
    /// pool, record nothing and are served nothing, and the cache comes back
    /// exactly as it was.
    ///
    /// Unpaused, such passes would leave plans behind, recorded at positions
    /// no real prompt takes, which no `reset` undoes. A key left pending
    /// would have the paused passes recorded into it; its recording is
    /// dropped and it records again on its next sight. Refused while a pass
    /// is open.
    pub fn pause_pass_plans(&self) -> Result<PassPlansPaused<'_>> {
        let device = match super::allocation_scope::active(self)? {
            Some(scope) => std::borrow::Cow::Owned(scope.view.clone()),
            None => std::borrow::Cow::Borrowed(self),
        };
        let mut g = device.pass_plans.lock();
        if g.as_ref().is_some_and(|pp| pp.state.current.is_some()) {
            return Err(MetalError::Message(
                "pause_pass_plans: a pass is open (finish_pass was not called)".to_string(),
            )
            .into());
        }
        let mut saved = g.take();
        drop(g);
        if let Some(pp) = saved.as_mut() {
            if pp.state.pending.take().is_some() {
                let _ = device.finish_arena_recording(pp.layout);
            }
            device
                .pass_state
                .fetch_and(!0b01, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(PassPlansPaused { device, saved })
    }

    /// Whether a plan for `key` is cached: the pass will be served from the
    /// base, allocating none of its activations (its admission owes no
    /// transient).
    /// Drop every cached pass plan and its arena, keeping the base: a
    /// buffer the plans bound was reallocated (an elastic tenant's slot
    /// pools, `profiles-and-memory-design.md` §4.2), so a replay would bind
    /// the old one. The next pass of each key records again. Returns how
    /// many plans were dropped; a no-op while a pass is open.
    pub fn drop_pass_plans(&self) -> usize {
        let mut g = self.pass_plans.lock();
        let Some(pp) = g.as_mut() else {
            return 0;
        };
        if pp.state.current.is_some() {
            return 0;
        }
        let n = pp.plans.len();
        pp.plans.clear();
        pp.arenas.clear();
        pp.arena_generation.clear();
        pp.escapes.clear();
        pp.ragged_order.clear();
        pp.state.pending = None;
        pp.state.base_generation += 1;
        pp.counters.evicted += n as u64;
        n
    }

    pub fn pass_plan_cached(&self, key: &PassKey) -> bool {
        self.pass_plans
            .lock()
            .as_ref()
            .is_some_and(|pp| pp.plans.contains_key(key))
    }

    /// How many `Ragged` plans the cache keeps (default 32); beyond it the
    /// oldest is dropped and counted (`PassPlanCounters::capped`). The other
    /// kinds are bounded by the configuration (one `Decode{B}` per B, one
    /// `Prefill{w}` per width the schedule names) and are never dropped.
    /// The fraction of `recommendedMaxWorkingSetSize` a base grow is
    /// admitted against (default 0.85, the KV grow's); 0 turns the check off.
    pub fn set_pass_plan_admit_fraction(&self, fraction: f64) {
        if let Some(pp) = self.pass_plans.lock().as_mut() {
            pp.admit_fraction = fraction.max(0.0);
        }
    }

    pub fn set_pass_plan_ragged_cap(&self, cap: usize) {
        if let Some(pp) = self.pass_plans.lock().as_mut() {
            pp.ragged_cap = cap.max(1);
            // Lowering the cap drops the oldest at once rather than waiting
            // for the next plan: the bound is on what is HELD.
            while pp.ragged_order.len() > pp.ragged_cap {
                if let Some(old) = pp.ragged_order.pop_front() {
                    if pp.plans.remove(&old).is_some() {
                        pp.arenas.remove(&old);
                        pp.escapes.remove(&old);
                        pp.counters.capped += 1;
                    }
                }
            }
        }
    }

    /// Whether an activation this device hands out may live in a slot a
    /// later pass reuses — the pass cache is on, an arena is installed, or
    /// one is being recorded. A value held ACROSS passes (resumable logits)
    /// must then be copied into memory the holder owns before any other
    /// pass is encoded (`DESIGN.md` §9.2aa, escapes are declared); off, the
    /// pool keeps a held buffer alive for as long as it is referenced.
    pub fn activations_may_be_reused(&self) -> bool {
        self.pass_plans_enabled()
            || self.arena_slot().is_some()
            || self.arena_recorder.lock().is_some()
    }

    pub fn pass_plan_counters(&self) -> Option<PassPlanCounters> {
        self.pass_plans.lock().as_ref().map(|p| p.counters)
    }

    /// Requests this device's pools received while a served pass was open,
    /// `(shared, private)`: zero when every served pass allocated from its
    /// plan alone (`DESIGN.md` §9.2aa, I1).
    ///
    /// The pools' own `allocations` cannot say this. They count misses over
    /// every allocation, and a request made between passes (a staged input)
    /// misses or hits according to whether the completion handler has
    /// drained its same-size predecessor yet (§6.3b), which is the GPU's
    /// timing, not the plan's coverage. Scoped to the device: a request
    /// counts whichever thread made it.
    pub fn served_pass_pool_requests(&self) -> (u64, u64) {
        let [shared, private] = &*self.served_pass_pool_requests;
        (
            shared.load(std::sync::atomic::Ordering::Relaxed),
            private.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Counts a request that reached pool `which` (0 shared, 1 private) if a
    /// served pass is open. One relaxed load when none is.
    fn note_pool_request(&self, which: usize) {
        if self.pass_state.load(std::sync::atomic::Ordering::Relaxed) & 0b100 != 0 {
            self.served_pass_pool_requests[which]
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if served_request_trace_on() {
                #[cfg(any(debug_assertions, feature = "profile"))]
                note_served_request_site(which);
            }
        }
    }

    /// The undeclared escapes of `key`'s plan as `(ordinal, bytes, allocated
    /// at dispatch, first read at dispatch of the following pass)` — empty
    /// when it has none or no plan.
    pub fn pass_plan_escapes(&self, key: &PassKey) -> Vec<(usize, usize, usize, usize)> {
        let g = self.pass_plans.lock();
        g.as_ref()
            .and_then(|p| p.escapes.get(key).cloned())
            .unwrap_or_default()
    }

    /// Keys with a cached plan, and the base's bytes.
    pub fn pass_plan_report(&self) -> Option<(Vec<PassPlanEntry>, usize)> {
        let g = self.pass_plans.lock();
        let p = g.as_ref()?;
        let mut keys: Vec<PassPlanEntry> = p
            .plans
            .iter()
            .map(|(k, plan)| {
                let (covered, total) = plan.covered();
                (*k, covered, total, p.escapes.get(k).map_or(0, |e| e.len()))
            })
            .collect();
        keys.sort_by_key(|(k, ..)| format!("{k:?}"));
        Some((keys, p.base.as_ref().map_or(0, |b| b.length())))
    }

    /// The pass in progress took a data-dependent branch its key does not
    /// name: a model that settles a step and, on a miss, replays it inside
    /// the same pass. Its recording holds the branch, which a pass of the
    /// same key that does not take it never runs, so the values the branch
    /// supersedes (the first computation's logits) would be planned as dead
    /// and served to a pass whose outputs they are. Such a pass still closes
    /// the key it observes, and is never itself a plan; its key records
    /// again on its next sight (`DESIGN.md` §9.2aa). A no-op when no pass is
    /// open or the cache is off.
    pub fn pass_diverged(&self) {
        if let Some(pp) = self.pass_plans.lock().as_mut() {
            if pp.state.current.is_some() && !pp.diverged {
                pp.diverged = true;
                pp.counters.diverged += 1;
            }
        }
    }

    /// Whether the open pass is one a model bracketed itself
    /// (`PassOpener::Model`): the model reads some of its values after it
    /// closes the pass (a settle after the pass body), so those must be
    /// declared escapes; under a driver's pass the same reads are inside it.
    /// One relaxed load. This is body-open, not transaction-open: Closing
    /// clears these bits while `pass_open` remains true until cleanup finishes.
    /// Receiver-bound; scoped callers first resolve `execution_device`.
    pub fn model_pass_open(&self) -> bool {
        self.pass_state.load(std::sync::atomic::Ordering::Relaxed) & 0b1010 == 0b1010
    }

    /// Whether a planned pass is open (`begin_pass` without its
    /// `finish_pass`): a model that brackets its own passes skips the
    /// bracket when a driver already holds one. Includes the Closing transaction
    /// after body completion; receiver-bound as are the other planner queries.
    pub fn pass_open(&self) -> bool {
        self.pass_plans
            .lock()
            .as_ref()
            .is_some_and(|pp| pp.state.current.is_some())
    }

    /// Name the allocation sequence the next pass runs, for a model whose
    /// passes of one kind take one of several sequences it can tell apart
    /// before the pass ([`PassKey::variant`]). Called from the model's
    /// `prepare_pass`, which a driver calls immediately before `begin_pass`;
    /// `begin_pass` folds it into the driver's key and clears it, and
    /// `finish_pass` clears one set inside a pass, so a variant never
    /// outlives the pass it was named for. A relaxed store; nothing when the
    /// cache is off.
    pub fn set_next_pass_variant(&self, variant: u32) {
        self.next_pass_variant
            .store(variant, std::sync::atomic::Ordering::Relaxed);
    }

    /// Open a forward pass of shape `key`. Under a cached plan the pass is
    /// served from the base; on a key's first sight, and on the pass after
    /// it, the pass runs on the pool under the recorder. Falls back to
    /// [`Self::begin_decode_step`] when the cache is off.
    pub fn begin_pass(&self, key: PassKey) -> Result<()> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope.view.begin_pass_in(key);
        }
        self.begin_pass_in(key)
    }

    fn begin_pass_in(&self, key: PassKey) -> Result<()> {
        let variant = self
            .next_pass_variant
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        // A driver's key names the kind; the variant is the model's, named
        // in its `prepare_pass`. A model that opens its own pass keys it
        // itself.
        let key = match key.opener {
            PassOpener::Driver => key.with_variant(variant),
            PassOpener::Model => key,
        };
        let mut g = self.pass_plans.lock();
        let Some(pp) = g.as_mut() else {
            drop(g);
            self.begin_decode_step();
            return Ok(());
        };
        if pp.state.current.is_some() {
            return Err(MetalError::Message(
                "begin_pass: a pass is already open (finish_pass was not called)".to_string(),
            )
            .into());
        }
        let opener_bit = match key.opener {
            PassOpener::Model => 0b1000,
            PassOpener::Driver => 0,
        };
        self.pass_state
            .fetch_or(0b10 | opener_bit, std::sync::atomic::Ordering::Relaxed);
        pp.diverged = false;
        // Whatever this pass does, the arena the last pass installed must not
        // serve it: a pass is served only under its own key's plan.
        self.clear_arena_keep_base();
        // A refusal past the bits above leaves no half-open pass: the body
        // bits are cleared and no `current` is registered, so the caller's
        // error path owes no `finish_pass` and the next pass opens on the
        // state this one found.
        let opened = match (pp.state.pending, pp.plans.contains_key(&key)) {
            (Some(closes), cached) => {
                // The recorder still holds the pending key's pass as its
                // current step; this pass becomes the step that says what
                // escaped it.
                self.next_arena_recording_step();
                pp.counters.observed += 1;
                Ok(if cached {
                    PassState::Observing { closes }
                } else {
                    PassState::Recording {
                        key,
                        closes: Some(closes),
                    }
                })
            }
            (None, true) => {
                let generation = pp.state.base_generation;
                let stale = pp.arena_generation.get(&key) != Some(&generation);
                let built = if stale || !pp.arenas.contains_key(&key) {
                    let plan = pp.plans.get(&key).expect("cached").clone();
                    let base = pp.base.clone().expect("a cached plan has a base");
                    Arena::new(base, plan, pp.layout)
                        .map_err(|e| MetalError::Message(format!("pass plan rejected: {e}")))
                        .map(|arena| {
                            pp.arenas.insert(key, arena);
                            pp.arena_generation.insert(key, generation);
                        })
                } else {
                    Ok(())
                };
                built.map(|()| {
                    let arena = pp.arenas.get(&key).expect("built above").clone();
                    crate::kernels::metal::set_hazard_key_unless_explicit(
                        crate::kernels::metal::HazardKey::Range,
                    );
                    {
                        let mut slot = self.arena.write();
                        *slot = Some(arena.clone());
                    }
                    arena.begin_step();
                    pp.counters.served += 1;
                    self.pass_state
                        .fetch_or(0b100, std::sync::atomic::Ordering::Relaxed);
                    PassState::Served(key)
                })
            }
            (None, false) => {
                self.begin_arena_recording();
                pp.counters.recorded += 1;
                Ok(PassState::Recording { key, closes: None })
            }
        };
        match opened {
            Ok(state) => {
                pp.state.current = Some(state);
                Ok(())
            }
            Err(error) => {
                self.pass_state
                    .fetch_and(!0b1110, std::sync::atomic::Ordering::Relaxed);
                Err(error.into())
            }
        }
    }

    /// Close the pass `begin_pass` opened: build the plan a pending key was
    /// waiting for, grow the base if it needs more, and register this pass's
    /// key as pending if it was recorded. Falls back to
    /// [`Self::finish_decode_step`] when the cache is off.
    pub fn finish_pass(&self) -> Result<()> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope.view.finish_pass_in(Some(&scope.context));
        }
        self.finish_pass_in(None)
    }

    fn finish_pass_in(
        &self,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<()> {
        self.next_pass_variant
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let mut g = self.pass_plans.lock();
        let Some(pp) = g.as_mut() else {
            drop(g);
            return self.finish_decode_step();
        };
        if pp.state.closing.is_some() {
            return Err(crate::Error::wrap(PlannerInvalidated(
                "finish already closing",
            )));
        }
        let state = pp
            .state
            .current
            .take()
            .ok_or_else(|| MetalError::Message("finish_pass: no pass is open".into()))?;
        self.pass_state
            .fetch_and(!0b1110, std::sync::atomic::Ordering::Relaxed);
        let closes = match state {
            PassState::Recording { closes, .. } => closes,
            PassState::Observing { closes } => Some(closes),
            PassState::Served(_) => None,
        };
        if let Some(closes) = closes {
            let (plan, undeclared, recorder) = {
                let g = self.arena_recorder.lock();
                let rec = g.as_ref().ok_or_else(|| {
                    MetalError::Message("finish_pass: no recorder while a key was pending".into())
                })?;
                let r = rec.lock();
                (
                    r.plan_previous(pp.layout),
                    r.previous_undeclared_ordinals(),
                    rec.clone(),
                )
            };
            // Preserve historical diagnostics before a candidate failure; these
            // are not publication of a cached plan.
            pp.counters.escapes += undeclared.len() as u64;
            pp.escapes.insert(closes, undeclared);
            let needed = plan
                .as_ref()
                .map(StepPlan::checked_base_bytes)
                .transpose()
                .map_err(MetalError::Message)?;
            let token = Arc::new(());
            let session = self.session_region.read();
            let session_generation = self.allocation_namespace.session_generation.lock().clone();
            drop(session);
            let transaction = ClosingPlan {
                identity: crate::planner_transaction::ClosingIdentity {
                    cache: pp.state.identity.clone(),
                    token: token.clone(),
                    namespace: self.allocation_namespace.clone(),
                    session: session_generation,
                    base_generation: pp.state.base_generation,
                    recorder,
                },
                have: pp.base.as_ref().map_or(0, Buffer::length),
                fraction: pp.admit_fraction,
                closes,
                state,
                plan,
                needed,
            };
            pp.state.current = Some(state);
            pp.state.closing = Some(token);
            drop(g);
            return self.complete_closing(transaction, context);
        }

        pp.state.current = None;
        self.pass_state
            .fetch_and(!0b1110, std::sync::atomic::Ordering::Relaxed);
        let mut discarded = None;
        match state {
            PassState::Served(key) => {
                self.end_decode_step();
                if let Some(arena) = self.arena_slot() {
                    if arena.take_drifted() {
                        pp.plans.remove(&key);
                        discarded = pp.arenas.remove(&key);
                        pp.counters.evicted += 1;
                    }
                }
                self.clear_arena_keep_base();
            }
            PassState::Recording { key, closes: None } => {
                if pp.plans.contains_key(&key) || pp.diverged {
                    pp.state.pending = None;
                    self.finish_arena_recording(pp.layout);
                } else {
                    pp.state.pending = Some(key);
                }
            }
            _ => unreachable!("closing paths dispatched above"),
        }
        drop(g);
        drop(discarded);
        Ok(())
    }

    fn check_closing_scope(&self) -> Result<()> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            if !Arc::ptr_eq(&scope.view.allocation_namespace, &self.allocation_namespace) {
                return Err(crate::Error::wrap(PlannerInvalidated(
                    "active allocation namespace changed",
                )));
            }
        }
        Ok(())
    }

    fn complete_closing(
        &self,
        mut transaction: ClosingPlan,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<()> {
        let mut closing = ClosingGuard {
            device: self,
            cache: transaction.identity.cache.clone(),
            token: transaction.identity.token.clone(),
            armed: true,
        };
        let grow = transaction
            .needed
            .filter(|&needed| needed > transaction.have);
        let staged = if let Some(needed) = grow {
            // Compatibility diagnostic remains net growth. The authoritative
            // ticket below reserves the FULL candidate while old bases survive.
            if transaction.fraction > 0.0 {
                let working_set = self.device.recommended_max_working_set_size();
                let budget = (working_set as f64 * transaction.fraction) as usize;
                let grow = needed.saturating_sub(transaction.have);
                let a = self.admit_bytes(grow, budget);
                if !a.fits {
                    let refusal =
                        a.machine
                            .unwrap_or(crate::memory_admission::AllocationError::Refused {
                                requested: grow,
                                available: budget.saturating_sub(a.committed),
                                domain: crate::memory_admission::RefusalDomain::Budget,
                            });
                    self.report_refusal(
                        &format_args!(
                            "pass plan base grow refused: {} B committed, {} B budget, \
                             {} B net grow; no plan cached, nothing allocated",
                            a.committed, budget, grow
                        ),
                        &refusal,
                    );
                    return Err(refusal.into());
                }
            }
            Some(self.backings.resident_private.stage(needed, context)?)
        } else {
            None
        };

        // A never-selected loser drops raw before its uncommitted ticket. All
        // guards here end BEFORE the local staged candidate can be destroyed.
        self.check_closing_scope()?;
        {
            let g = self.pass_plans.lock();
            let session = self.session_region.read();
            let version = self.allocation_namespace.session_generation.lock();
            let recorder = self.arena_recorder.lock();
            transaction.validate(
                g.as_ref(),
                &self.allocation_namespace,
                &version,
                recorder.as_ref(),
            )?;
            drop(recorder);
            drop(version);
            drop(session);
        }
        let mut candidate = match staged {
            Some(staged) => Some(staged.commit()?),
            None => None,
        };

        // Commit ran outside planner locks. If a void/Drop mutator replaced
        // state meanwhile, this is a committed loser: release physical charge,
        // but NEVER resurrect consumed pass demand. Caller needs a new reserve.
        let checked = self.check_closing_scope();
        let mut g = self.pass_plans.lock();
        let session = self.session_region.read();
        let version = self.allocation_namespace.session_generation.lock();
        let recorder = self.arena_recorder.lock();
        let checked = checked.and_then(|_| {
            transaction.validate(
                g.as_ref(),
                &self.allocation_namespace,
                &version,
                recorder.as_ref(),
            )
        });
        if let Err(error) = checked {
            drop(recorder);
            drop(version);
            drop(session);
            drop(g);
            if let Some(candidate) = candidate {
                self.backings.resident_private.discard_selected(candidate);
            }
            return Err(error);
        }
        let pp = g.as_mut().expect("validated cache");
        let mut old_base = None;
        let mut old_reservation = None;
        let mut old_arenas = HashMap::new();
        if let Some(candidate) = candidate.take() {
            old_base = pp.base.replace(candidate.buffer);
            old_reservation = std::mem::replace(&mut pp.base_reservation, candidate.reservation);
            old_arenas = std::mem::take(&mut pp.arenas);
            pp.state.base_generation += 1;
            pp.counters.base_grows += 1;
        }
        let mut evicted_arenas = Vec::new();
        if let Some(plan) = transaction.plan.take() {
            pp.plans.insert(transaction.closes, plan);
            pp.counters.plans += 1;
            if matches!(transaction.closes.kind, PassKind::Ragged { .. }) {
                pp.ragged_order.push_back(transaction.closes);
                while pp.ragged_order.len() > pp.ragged_cap {
                    if let Some(old) = pp.ragged_order.pop_front() {
                        if pp.plans.remove(&old).is_some() {
                            if let Some(arena) = pp.arenas.remove(&old) {
                                evicted_arenas.push(arena);
                            }
                            pp.escapes.remove(&old);
                            pp.counters.capped += 1;
                        }
                    }
                }
            }
        }
        let finish_recording = match transaction.state {
            PassState::Recording { key, .. } if !pp.plans.contains_key(&key) && !pp.diverged => {
                pp.state.pending = Some(key);
                false
            }
            _ => {
                pp.state.pending = None;
                true
            }
        };
        let layout = pp.layout;
        // Body bits are closed, but current/closing keep the transaction open.
        // Legacy Drop/raw recorder mutators can still replace state; cleanup
        // below is identity-conditional, never a blind clear of its successor.
        drop(recorder);
        drop(version);
        drop(session);
        drop(g);
        if let Some(previous) = old_base {
            self.residency_set.remove(&previous);
            self.hold_in_flight(&previous);
        }
        drop(old_arenas);
        drop(evicted_arenas);
        drop(old_reservation);
        if finish_recording {
            self.finish_arena_recording_if(layout, &transaction.identity.recorder)?;
        }
        self.trim_unused_buffers();
        let mut g = self.pass_plans.lock();
        let valid = g.as_ref().is_some_and(|pp| {
            Arc::ptr_eq(&pp.state.identity, &transaction.identity.cache)
                && pp
                    .state
                    .closing
                    .as_ref()
                    .is_some_and(|token| Arc::ptr_eq(token, &transaction.identity.token))
        });
        if !valid {
            return Err(crate::Error::wrap(PlannerInvalidated(
                "cache replaced during closing cleanup",
            )));
        }
        let pp = g.as_mut().expect("validated cache");
        pp.state.current = None;
        pp.state.closing = None;
        self.pass_state
            .fetch_and(!0b1110, std::sync::atomic::Ordering::Relaxed);
        closing.armed = false;
        Ok(())
    }

    /// Keep `base` alive until every command buffer encoded so far has run,
    /// when this queue's command buffers do not retain what they read: an
    /// arena or region base being replaced was bound by the step just encoded.
    /// A no-op on the retaining kind.
    fn hold_in_flight(&self, base: &Buffer) {
        crate::kernels::metal::in_flight::hold_buffer(self.commands.in_flight_hold(), base);
    }

    /// Uninstall the arena slot without releasing any base: the pass cache
    /// owns its base and the arena is one view of it per key.
    fn clear_arena_keep_base(&self) {
        let mut slot = self.arena.write();
        *slot = None;
    }

    /// Tell the installed executor the decode step is complete
    /// (`Executor::end_step`, `DESIGN.md` §11.3).
    ///
    /// The decode loop calls this once per token, after everything the step
    /// encodes. Separate from [`Self::end_decode_step`] because the harnesses
    /// that drive an executor and an arena together close the two on their own
    /// schedules, and closing an executor's step twice records an empty window.
    /// A no-op under `ExecutorSlot::Classical`.
    pub fn end_executor_step(&self) -> Result<()> {
        self.commands
            .executor()
            .end_step(&self.device, &self.commands)
            .map_err(MetalError::from)?;
        Ok(())
    }

    pub fn arena_counters(&self) -> Option<ArenaCounters> {
        self.arena_slot().map(|a| a.counters())
    }

    /// Where the installed arena's offsets came from (issue #70).
    ///
    /// Reported so a harness can show the GPU path *engaged* rather than
    /// asserting that a flag was passed. §2.4: an instrument that cannot be
    /// shown to have engaged has not measured anything, and #69's first
    /// determinism gate was vacuous for exactly that reason.
    pub fn arena_offsets(&self) -> Option<ArenaOffsets> {
        self.arena_slot().map(|a| a.offsets())
    }

    /// Begin observing allocations so a plan can be derived from one step.
    ///
    /// Record over a *steady-state* decode step, not the first one: the first
    /// token after prefill allocates a different set, and #68 plans from
    /// `decode[1]` for the same reason.
    pub fn begin_arena_recording(&self) {
        self.set_base_mode(true);
        let rec = Arc::new(Mutex::new(ArenaRecorder::new()));
        // Namespace recorder then global observer: begin and conditional finish
        // use the same lock order, so cleanup cannot clear a newer recording.
        let previous = {
            let mut g = self.arena_recorder.lock();
            crate::kernels::metal::arena::set_bind_observer(Some(rec.clone()));
            g.replace(rec)
        };
        drop(previous);
    }

    /// Close the decode step being recorded and begin recording another.
    ///
    /// **Two steps are the minimum a plan can be built from**, because comparing
    /// them is what separates an activation from session state: an allocation
    /// whose size moved between them is sized by `kv_len` and must not enter the
    /// arena (`DESIGN.md` §9.1, #68 finding 4). #68's planner likewise requires
    /// two decode steps and refuses rather than silently returning nothing.
    pub fn next_arena_recording_step(&self) {
        {
            let g = self.arena_recorder.lock();
            if let Some(rec) = g.as_ref() {
                {
                    let mut r = rec.lock();
                    r.next_step();
                }
            }
        }
    }

    /// How many recorded ordinals were excluded as session state, of the total.
    pub fn arena_recording_excluded(&self) -> Option<(usize, usize)> {
        let g = self.arena_recorder.lock();
        let rec = g.as_ref()?;
        let r = rec.lock();
        Some(r.excluded())
    }

    /// Exclusions split by which test caught them: `(size_grew, outlived_step)`.
    ///
    /// Reported separately because the two detectors see different populations
    /// and neither subsumes the other -- size growth finds the KV cache, and
    /// cross-step liveness finds the conv state, which is fixed at
    /// `[B, 2048, 3]` and invisible to a size comparison (`DESIGN.md` §5.7,
    /// §9.2c). A run where the second number is zero would mean it is not
    /// earning its place; a run where it is nonzero names values the size test
    /// admitted.
    ///
    /// The two counts overlap where both tests fire, so they do not sum to the
    /// total exclusion count.
    pub fn arena_recording_excluded_by_test(&self) -> Option<(usize, usize)> {
        let g = self.arena_recorder.lock();
        let rec = g.as_ref()?;
        let r = rec.lock();
        Some(r.excluded_by_test())
    }

    /// Reservations for a session region, derived from the recording (#47).
    ///
    /// `(ordinal, bytes at capacity)` for each ordinal whose size grew between
    /// the two recorded steps, extrapolated from the **measured** stride to
    /// `capacity_tokens`. See
    /// [`ArenaRecorder::session_reservations`](crate::kernels::metal::ArenaRecorder::session_reservations)
    /// for why the stride is measured rather than computed from the geometry,
    /// and why the fixed-size session state is deliberately not included.
    pub fn arena_recording_session_reservations(
        &self,
        tokens_recorded: usize,
        capacity_tokens: usize,
    ) -> Option<Vec<(usize, usize)>> {
        let g = self.arena_recorder.lock();
        let rec = g.as_ref()?;
        let r = rec.lock();
        Some(r.session_reservations(tokens_recorded, capacity_tokens))
    }

    /// Stop observing and build a plan from what was seen.
    ///
    /// Returns `None` if nothing was recorded, which is a real outcome worth
    /// distinguishing from an empty plan -- it means recording was never begun
    /// or the step allocated nothing.
    pub fn finish_arena_recording(&self, layout: ArenaLayout) -> Option<StepPlan> {
        let rec = {
            let mut g = self.arena_recorder.lock();
            let rec = g.take()?;
            crate::kernels::metal::arena::clear_bind_observer_if(&rec);
            rec
        };
        let r = rec.lock();
        if r.is_empty() {
            None
        } else {
            Some(r.plan(layout))
        }
    }

    fn finish_arena_recording_if(
        &self,
        layout: ArenaLayout,
        expected: &Arc<Mutex<ArenaRecorder>>,
    ) -> Result<Option<StepPlan>> {
        let rec = {
            let mut g = self.arena_recorder.lock();
            if !g.as_ref().is_some_and(|r| Arc::ptr_eq(r, expected)) {
                return Err(crate::Error::wrap(PlannerInvalidated(
                    "recorder replaced during cleanup",
                )));
            }
            crate::kernels::metal::arena::clear_bind_observer_if(expected);
            g.take().expect("identity checked")
        };
        let r = rec.lock();
        Ok(if r.is_empty() {
            None
        } else {
            Some(r.plan(layout))
        })
    }

    /// Record one decode step and install the resulting arena.
    ///
    /// The whole sequence a caller needs: observe a steady-state step, derive
    /// the plan, allocate the arena, and switch to it. `step` runs the model for
    /// exactly one token.
    pub fn record_and_install_arena<F>(&self, layout: ArenaLayout, step: F) -> Result<bool>
    where
        F: FnOnce() -> Result<()>,
    {
        self.begin_arena_recording();
        let outcome = step();
        // Stop recording whether or not the step succeeded, so a failure cannot
        // leave the recorder attached and quietly accumulating.
        let plan = self.finish_arena_recording(layout);
        outcome?;
        match plan {
            Some(plan) => {
                self.install_arena(plan, layout)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Creates a new buffer (not necessarily zeroed).
    ///
    /// Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
    /// Falls back to StorageModeShared on iOS where Private is not always available.
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .new_buffer_in(element_count, dtype, _name, Some(&scope.context));
        }
        self.new_buffer_in(element_count, dtype, _name, None)
    }

    fn new_buffer_in(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        // An op's result buffer: writing it at a type this GPU only emulates
        // is that type's arithmetic, admitted only by the opt-in (§15.2a).
        if let Some(feature) = crate::kernels::metal::arms::Feature::emulable_dtype(dtype) {
            self.emulated
                .admit(feature, || format!("a `{feature}` op"))
                .map_err(MetalError::from)?;
        }
        let size = crate::allocation_geometry::typed_bytes(element_count, dtype)?;
        // Validate miss geometry before arena ordinals or pool drain/reuse can
        // change state. Reuse still takes the original requested byte count.
        let alloc = buf_size(size)?;
        // Layer 3 (`DESIGN.md` §9.2a). The arena is offered the allocation
        // first; it serves it only inside a decode step, only while the plan
        // still has an ordinal, and only when the size matches what that ordinal
        // was planned for. Everything else -- prefill, model setup, and every
        // session-state allocation whose size grows with `kv_len` -- falls
        // through to the pool below, unchanged.
        if let Some(arena) = self.arena_slot() {
            let offer = arena.acquire_attributed(size);
            if let Some(b) = offer.served {
                return Ok(b);
            }
            // Layer 3b (#47, `DESIGN.md` §9.2x). The arena declined; if it
            // declined because this ordinal is **session state**, the session
            // region may own a reservation for it.
            //
            // # Why only the `Session` decline, and not every decline
            //
            // The region's tenants are keyed by allocation ordinal, and a
            // reservation means "these bytes belong to this ordinal for the
            // whole session". That is meaningful only for the population the
            // plan classified as session state. `PastPlan` and `SizeMismatch`
            // say the step has drifted from what was recorded, and serving a
            // drifted allocation from a fixed reservation is exactly the
            // aliasing §9.3 gives no safety net against — so those keep falling
            // to the pool, unchanged.
            //
            // `Inactive` is outside a decode step (prefill, setup, sampling) and
            // carries no ordinal at all, which is why the `ordinal` is taken
            // from the offer rather than recomputed: the `fetch_add` already
            // happened inside `acquire_attributed`, and a second one here would
            // walk the plan out of alignment.
            if offer.decline == Some(ArenaDecline::Session) {
                if let (Some(ordinal), Some(region)) = (offer.ordinal, self.session_region_slot()) {
                    if let Some(b) = region.acquire(ordinal, size) {
                        return Ok(self.note_for_recording(b, size));
                    }
                }
            }
        }
        self.note_pool_request(1);
        let origin = self.declared_allocation_origin();
        let buffer = match self
            .private_buffers
            .acquire_for_allocation_in(size, &origin)
        {
            Ok(b) => b,
            Err(miss) => {
                // `DESIGN.md` §9.5k's one branch, on the derived residual. It
                // sits on the pool MISS, after `acquire` has declined, because
                // a hit allocates nothing and there is nothing to bound; and
                // BEFORE `new_buffer`, because reporting an overrun after the
                // allocation is committed is useless. Inert with no cap
                // installed, which is every process that has not run admission.
                self.private_buffers
                    .check_residual(alloc)
                    .map_err(AllocationError::from)?;
                let split = self.residency_set.policy().is_split();
                let new_buffer =
                    Self::allocate_backing(self.backings.for_miss(true, split), alloc, context)?;
                // `size` is the pre-rounding request and `alloc` is what Metal
                // was given; `adopt_requesting` counts both so §9.2's residual
                // overhang is derivable. The rounding happened at `alloc`
                // above, which is the last point where the two are both in
                // scope (#47).
                self.private_buffers
                    .adopt_after_miss_in(new_buffer, alloc, miss, &origin)?
            }
        };
        Ok(self.note_for_recording(buffer, size))
    }

    /// While a plan is being recorded, note this allocation.
    ///
    /// Records the allocation's *size* and the *address* of the buffer serving
    /// it. The address is only how later binds are attributed back to this
    /// value; the value itself is the **allocation event**, never the buffer
    /// (`DESIGN.md` §9.2c). When a later allocation is handed the same address
    /// it takes the address over, and the earlier value keeps the interval it
    /// accumulated while it held it -- so one pooled buffer serving 60 values
    /// yields 60 intervals rather than one merged one. That merge is the mistake
    /// that made #68's first planner report 3.55 MB against the true 68 KB.
    ///
    /// The interval **ends at the last bind**, not here and not when the handle
    /// drops -- see `ArenaRecorder::record_bind` for why the CPU's clock is the
    /// wrong one.
    fn note_for_recording(&self, buffer: BufferHandle, size: usize) -> BufferHandle {
        // **One relaxed load on the allocation path when nothing records**,
        // which is every run that did not ask for an arena or a plan cache.
        // `observing_binds` is set exactly while a recorder is installed
        // (`set_bind_observer`), and it is the same gate `note_bind` and
        // `note_dispatch` take for the same reason: this function is called
        // for every buffer the device hands out, and taking the recorder's
        // lock to discover there is no recorder is a lock per allocation
        // bought for nothing.
        if !crate::kernels::metal::arena::observing_binds() {
            return buffer;
        }
        // Under the pass cache only a pass's own allocations are ordinals:
        // one made between passes takes none, and the address it lands on
        // stops naming the value that held it (`forget_addr`).
        let between_passes = self.pass_state.load(std::sync::atomic::Ordering::Relaxed) == 0b01;
        let guard = self.arena_recorder.lock();
        let Some(rec) = guard.as_ref() else {
            return buffer;
        };
        let mut r = rec.lock();
        if between_passes {
            r.forget_addr(buffer.raw_addr());
        } else {
            r.record_alloc(buffer.raw_addr(), size);
        }
        drop(r);
        buffer
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope.view.new_private_buffer_in(
                element_count,
                dtype,
                _name,
                Some(&scope.context),
            );
        }
        self.new_private_buffer_in(element_count, dtype, _name, None)
    }

    fn new_private_buffer_in(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let size = crate::allocation_geometry::typed_bytes(element_count, dtype)?;
        let a = Self::allocate_backing(&self.backings.resident_private, size, context)?;
        // Deliberately not pooled: this exists for persistent buffers, so it
        // returns a handle with no pool behind it. Dropping it frees the buffer
        // outright rather than offering it for reuse.
        Ok(BufferHandle::new(
            PooledBuffer::unpooled(a.buffer, size)
                .with_reservation(a.reservation)
                .held_in_flight(self.commands.in_flight_hold()),
        ))
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    /// The device buffer holding exactly `bytes`, shared with every other
    /// caller that asked for the same bytes ([`ConstantsByValue`]). For a
    /// kernel's argument block, chunk table or live count: a repeated value
    /// costs a hash and a lock, a new one a pooled upload. Larger than
    /// `CONSTANT_MAX_BYTES` falls back to an upload.
    pub fn constant_buffer(&self, bytes: &[u8]) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope.view.constant_buffer_in(bytes, Some(&scope.context));
        }
        self.constant_buffer_in(bytes, None)
    }

    fn constant_buffer_in(
        &self,
        bytes: &[u8],
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        if bytes.len() > CONSTANT_MAX_BYTES {
            return self.new_buffer_with_data_in(bytes, context);
        }
        {
            let mut g = self.constants.inner.lock();
            if let Some(b) = g.map.get(bytes) {
                let b = BufferHandle::clone(b);
                g.hits += 1;
                return Ok(b);
            }
        }
        let buf = self.new_buffer_with_data_in(bytes, context)?;
        let mut g = self.constants.inner.lock();
        g.misses += 1;
        if g.map.len() >= CONSTANT_ENTRIES {
            if let Some(old) = g.order.pop_front() {
                g.map.remove(&old);
            }
        }
        g.map.insert(bytes.to_vec(), BufferHandle::clone(&buf));
        g.order.push_back(bytes.to_vec());
        Ok(buf)
    }

    /// [`Self::constant_buffer`] for one plain-data value (a params block).
    pub fn constant_of<T: Copy>(&self, v: &T) -> Result<BufferHandle> {
        self.constant_buffer(pod_bytes(v))
    }

    /// A/B instrument (`LLOOM_UNPOOLED_UPLOADS`, read once): every host upload
    /// takes the unpooled `newBufferWithBytes` path, so a host write can never
    /// land in a recycled pool buffer. Never the shipping path.
    fn unpooled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::tuning::tuning().pool.unpooled_uploads)
    }

    /// A Shared buffer of `size` bytes the host writes and the GPU reads once,
    /// outside every pool: a fresh `newBuffer` registered in the residency set
    /// and unregistered when the handle drops (the `unpooled_resident` drop).
    /// The staging tier of a streamed weight load (`QMetalStorage::write_with`)
    /// takes this rather than `allocate_buffer` because a pooled host upload is
    /// the one tenant class whose reuse gate is known to leak at long context
    /// (lloom #748 / PR #754, which records it as `docs/invariants.md` §15.1m) — a recycled Shared buffer
    /// written by the host while a still-running command buffer reads it.
    pub fn new_unpooled_shared(&self, size: usize) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .new_unpooled_shared_in(size, Some(&scope.context));
        }
        self.new_unpooled_shared_in(size, None)
    }

    fn new_unpooled_shared_in(
        &self,
        size: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let a = Self::allocate_backing(&self.backings.resident_shared, size, context)?;
        Ok(BufferHandle::new(
            PooledBuffer::unpooled_resident(a.buffer, size, Arc::clone(&self.residency_set))
                .with_reservation(a.reservation)
                .held_in_flight(self.commands.in_flight_hold()),
        ))
    }

    /// A Private buffer of `size` bytes that lives as long as its handle —
    /// outside every pool, registered in the residency set and unregistered
    /// when the handle drops. For a persistent structure of known geometry
    /// (a streamed tier's slot pool) that must not be a pool bucket.
    /// As [`Self::new_persistent_private`], Shared: a persistent buffer the
    /// CPU writes into place (the streamed tier's slot pools under the
    /// direct read, `DESIGN.md` §20.4a).
    pub fn new_persistent_shared(&self, size: usize) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .new_persistent_shared_in(size, Some(&scope.context));
        }
        self.new_persistent_shared_in(size, None)
    }

    fn new_persistent_shared_in(
        &self,
        size: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        // A Shared buffer's pages are the process's, and under memory
        // pressure the kernel compresses them whatever the residency set
        // says (measured: 28 GB of Shared slot pools un-wired and 7.5 GB
        // of them compressed mid-prefill, #749 P27); a Private buffer's are
        // not. The `Mapped` backing pins them.
        let a = Self::allocate_backing(&self.backings.mapped_shared, size, context)?;
        Ok(BufferHandle::new(
            PooledBuffer::unpooled_resident(a.buffer, size, Arc::clone(&self.residency_set))
                .with_reservation(a.reservation)
                .held_in_flight(self.commands.in_flight_hold()),
        ))
    }

    /// Reuse one scratch per allocation namespace. Growth is admitted outside
    /// the cache lock; concurrent candidates remain charged until discarded.
    pub(crate) fn dequant_scratch(&self, bytes: usize) -> Result<BufferHandle> {
        match super::allocation_scope::active(self)? {
            Some(scope) => scope.view.dequant_scratch_in(bytes, Some(&scope.context)),
            None => self.dequant_scratch_in(bytes, None),
        }
    }

    fn dequant_scratch_in(
        &self,
        bytes: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        {
            let slot = self.allocation_namespace.dequant_scratch.lock();
            if let Some(buffer) = slot.as_ref().filter(|b| b.size() >= bytes) {
                return Ok(buffer.clone());
            }
        }
        let candidate = self.new_persistent_private_in(bytes, context)?;
        let (result, discarded) = {
            let mut slot = self.allocation_namespace.dequant_scratch.lock();
            if let Some(buffer) = slot.as_ref().filter(|b| b.size() >= bytes) {
                (buffer.clone(), Some(candidate))
            } else {
                let result = candidate.clone();
                (result, slot.replace(candidate))
            }
        };
        drop(discarded);
        Ok(result)
    }

    pub fn new_persistent_private(&self, size: usize) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .new_persistent_private_in(size, Some(&scope.context));
        }
        self.new_persistent_private_in(size, None)
    }

    fn new_persistent_private_in(
        &self,
        size: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let a = Self::allocate_backing(&self.backings.resident_private, size, context)?;
        Ok(BufferHandle::new(
            PooledBuffer::unpooled_resident(a.buffer, size, Arc::clone(&self.residency_set))
                .with_reservation(a.reservation)
                .held_in_flight(self.commands.in_flight_hold()),
        ))
    }

    /// Pin a Shared buffer's pages (`mlock`) so the kernel cannot compress
    /// them under memory pressure. A no-op for a buffer the CPU cannot see
    /// (Private storage: its pages are not the process's to compress).
    pub fn wire_buffer(&self, buffer: &PooledBuffer) {
        // `contents()` is null for a Private buffer (`Buffer::new`).
        crate::kernels::metal::backing::wire_pages(unsafe { buffer.contents() }, buffer.size());
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .new_buffer_with_data_in(data, Some(&scope.context));
        }
        self.new_buffer_with_data_in(data, None)
    }

    /// Cold real output/data plan position using this namespace's actual payer
    /// and transient Shared backing (including exact heap range custody).
    /// # Safety
    /// Same exhaustive closed rich/native/IO/foreign/raw contract as the private
    /// rich upload constructor. The plan admits complete capsule/control costs,
    /// stores this genuine descriptor by fixed position, and obtains admitted
    /// cold spill/growth before effects when checkout is Busy/Full. Outputs are
    /// not initialized values until the real operation establishes its writes.
    /// Legacy Tensor/PoolState selectors are NOT switched by this addition.
    pub(crate) unsafe fn new_prepared_buffer_in(
        &self,
        bytes: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<crate::kernels::metal::PreparedPoolSlot> {
        unsafe {
            self.new_prepared_buffer_profile_in(
                bytes,
                super::planned_output_catalog::OutputProfile::Shared,
                context,
            )
        }
    }

    /// Actual same factory, preserving requested geometry and physical custody;
    /// the original Shared method above is retained as a forwarding entry.
    /// # Safety
    /// Same complete closed-use contract as new_prepared_buffer_in. Profile is
    /// selected before any birth; no bare/legacy handle is enrolled afterward.
    unsafe fn new_prepared_buffer_profile_in(
        &self,
        bytes: usize,
        profile: super::planned_output_catalog::OutputProfile,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<crate::kernels::metal::PreparedPoolSlot> {
        let bytes = buf_size(bytes)?;
        let default_pass = match context {
            Some(_) => None,
            None => Some(
                self.allocation_namespace.native_domain.reserve(
                    crate::kernels::metal::ColdUnpooled::requested_pool_owned_bytes()
                        .map_err(|e| crate::Error::Msg(format!("pool metadata quote: {e:?}")))?,
                )?,
            ),
        };
        let default_context = default_pass.as_ref().map(|pass| pass.context());
        let metadata = context
            .or(default_context.as_ref())
            .expect("cold pool context");
        metadata.check_open()?;
        self.commands.check_failed().map_err(crate::Error::wrap)?;
        let capsule = unsafe { crate::kernels::metal::ColdUnpooled::cold_in(metadata) }
            .map_err(|e| crate::Error::Msg(format!("pool metadata: {e:?}")))?;
        unsafe {
            capsule.allocate_pool_with(
                metadata,
                || {
                    match profile {
                        super::planned_output_catalog::OutputProfile::Shared => self
                            .backings
                            .transient_shared
                            .alloc_custodied(bytes, context),
                        super::planned_output_catalog::OutputProfile::Private => self
                            .backings
                            .transient_private
                            .alloc_custodied(bytes, context),
                    }
                    .map_err(crate::Error::from)
                },
                self.commands.clock(),
                bytes,
                &self.commands,
            )
        }
    }

    /// Cold root installation only. Native operation positions are admitted
    /// separately and returned as genuine direct capabilities, never selected
    /// by raw identity or merely capacity. No legacy/Tensor selector changes.
    /// # Safety
    /// Same complete closed-use and full factory quote contract as the actual
    /// prepared buffer/staging constructors; no Commands/EDIT guard is held.
    pub(crate) unsafe fn prepare_output_catalog_in(
        &self,
        context: &crate::memory_admission::AllocationContext,
    ) -> Result<()> {
        use super::planned_output_catalog::PlannedOutputCatalog;
        if context.domain_key() != self.allocation_namespace.native_domain.key() {
            return Err(crate::memory_admission::AllocationError::DifferentDomain.into());
        }
        context.check_open()?;
        self.commands.check_failed().map_err(crate::Error::wrap)?;
        if self.allocation_namespace.output_catalog.get().is_some() {
            return Err(crate::Error::Msg("output catalog already installed".into()));
        }
        let catalog = unsafe { PlannedOutputCatalog::cold_in(context) }
            .map_err(|e| crate::Error::Msg(format!("output catalog root: {e:?}")))?;
        context.check_open()?;
        self.commands.check_failed().map_err(crate::Error::wrap)?;
        self.allocation_namespace
            .output_catalog
            .set(catalog)
            .map_err(|_| crate::Error::Msg("output catalog installation raced".into()))
    }

    /// Actual SAME-promise output + preloaded shader + direct params factory.
    /// The returned position identifies THIS immutable operation node, not a
    /// size/profile search result. Busy needs cold admitted alternate/growth.
    /// # Safety
    /// All Node/capsule/shadow/native/control/pipeline/cache terms are quoted
    /// before their births. All rows are prepared before any row's phase moves.
    /// The real safe caller authenticates original initialized input, implicit
    /// EOS/IO/foreign/raw/ICB users and native terminal/emission custody.
    pub(crate) unsafe fn prepare_affine_position_in(
        &self,
        context: &crate::memory_admission::AllocationContext,
        bytes: usize,
        profile: super::planned_output_catalog::OutputProfile,
        dtype: crate::kernels::kernels::affine::AffineDType,
        operation: crate::kernels::kernels::affine::AffineOperation,
        style: crate::kernels::kernels::params::ParamStyle,
        strided_rank: Option<usize>,
    ) -> Result<super::planned_output_catalog::DirectOutputPosition> {
        use super::operation_payload::{OperationPayload, PreparedAffineOperation};
        use super::planned_output_catalog::CatalogError;
        if context.domain_key() != self.allocation_namespace.native_domain.key() {
            return Err(crate::memory_admission::AllocationError::DifferentDomain.into());
        }
        let catalog = self
            .allocation_namespace
            .output_catalog
            .get()
            .ok_or_else(|| crate::Error::Msg("output catalog requires cold installation".into()))?;
        unsafe {
            catalog.append_cold_with(context, profile, || {
                let slot = self
                    .new_prepared_buffer_profile_in(bytes, profile, Some(context))
                    .map_err(CatalogError::Creator)?;
                let operation = PreparedAffineOperation::cold_in(
                    self,
                    context,
                    dtype,
                    operation,
                    style,
                    strided_rank,
                )
                .map_err(CatalogError::Operation)?;
                Ok((slot, OperationPayload::Affine(operation)))
            })
        }
        .map_err(|e| crate::Error::Msg(format!("affine plan birth: {e:?}")))
    }

    /// Genuine initialized F32 input factory for the first actual typed affine
    /// caller. No legacy Handle/raw view is rewrapped as counted provenance.
    /// # Safety
    /// Same complete native/IO/foreign/raw/CPU/emission contract as the actual
    /// rich upload. Complete default/supplied factory pricing precedes birth;
    /// logical reads are bounded by elements, never unused native padding.
    pub(crate) unsafe fn prepare_affine_f32_input_in(
        &self,
        data: &[f32],
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<PreparedAffineInput> {
        let handle = unsafe { self.new_rich_upload_with_data_in(data, context) }?;
        Ok(PreparedAffineInput {
            handle,
            elements: data.len(),
        })
    }

    /// Cold plan-owned Shared staging allocation. Default and supplied routes
    /// preserve the same actual native payer; supplied metadata partitions the
    /// existing promise and never reserves physical bytes twice.
    /// # Safety
    /// Same exhaustive tracked CPU/native/IO/foreign/raw/ICB future-emission
    /// contract as the private rich upload constructor. All complete capsule,
    /// control, plan-shadow and native allocation costs precede construction.
    /// Caller maps this returned slot to a fixed plan position, prepares writes
    /// BEFORE exact generation EDIT, and obtains admitted spill/growth outside
    /// guards on Busy/Full. This method is not selected by legacy ConstantsPool
    /// or Tensor upload paths today. Instrumentation/rewrite caller companions
    /// and native/ICB terminal closure remain activation obligations.
    pub(crate) unsafe fn new_prepared_staging_in(
        &self,
        bytes: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<crate::kernels::metal::PreparedStagingSlot> {
        if bytes == 0 {
            return Err(crate::Error::Msg(
                "staging requires nonempty storage".into(),
            ));
        }
        let default_pass = match context {
            Some(_) => None,
            None => Some(
                self.allocation_namespace.native_domain.reserve(
                    crate::kernels::metal::ColdUnpooled::requested_pool_owned_bytes()
                        .map_err(|e| crate::Error::Msg(format!("staging metadata quote: {e:?}")))?,
                )?,
            ),
        };
        let default_context = default_pass.as_ref().map(|pass| pass.context());
        let metadata = context
            .or(default_context.as_ref())
            .expect("cold staging context");
        metadata.check_open()?;
        self.commands.check_failed().map_err(crate::Error::wrap)?;
        let capsule = unsafe { crate::kernels::metal::ColdUnpooled::cold_in(metadata) }
            .map_err(|e| crate::Error::Msg(format!("staging metadata: {e:?}")))?;
        unsafe {
            capsule.allocate_staging_with(
                metadata,
                || {
                    self.backings
                        .resident_shared
                        .alloc_custodied(bytes, context)
                        .map_err(crate::Error::from)
                },
                self.commands.clock(),
                bytes,
                &self.commands,
            )
        }
    }

    /// Cold actual backing→physical→rich Handle route. The existing Tensor large
    /// upload branch may select this ONLY after actual native/IO/raw closure and
    /// complete pricing are enforced. It is not an enabled fallback today.
    /// # Safety
    /// EVERY safe/native/IO/foreign/raw/CPU emission from the returned handle
    /// retains counted External custody through own terminality. Submitted
    /// creator-generation cleanup observes its exact epoch before releasing the
    /// pin; no-submit cancellation and unknown differ. Same process account,
    /// actual native payer and clock are preserved; no sampled OPEN cutoff.
    /// Supplied producer prediction includes both requested capsule Layouts
    /// ONCE plus complete still-unquoted Arc/control/service/constructor overhead.
    unsafe fn new_rich_upload_with_data_in<T>(
        &self,
        data: &[T],
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let size = core::mem::size_of_val(data);
        // Empty logical tensors are legal. Metal still requires nonzero physical
        // storage: preserve the existing minimum-allocation geometry, retaining
        // real payer/range/accounting custody. No logical byte is readable, so
        // this branch does not manufacture an initialized-padding certificate.
        let empty_extent = if size == 0 { Some(buf_size(0)?) } else { None };
        // Default metadata reserve uses THIS namespace's actual immutable payer,
        // the same account/domain as resident_shared; never invent a model key.
        // Supplied route takes the same existing promise, with no second reserve.
        let default_pass = match context {
            Some(_) => None,
            None => Some(
                self.allocation_namespace.native_domain.reserve(
                    crate::kernels::metal::ColdUnpooled::requested_owned_bytes()
                        .map_err(|e| crate::Error::Msg(format!("upload metadata quote: {e:?}")))?,
                )?,
            ),
        };
        let default_context = default_pass.as_ref().map(|pass| pass.context());
        let metadata = context
            .or(default_context.as_ref())
            .expect("cold upload context");
        metadata.check_open()?;
        self.commands.check_failed().map_err(crate::Error::wrap)?;
        let capsule = unsafe { crate::kernels::metal::ColdUnpooled::cold_in(metadata) }
            .map_err(|e| crate::Error::Msg(format!("upload metadata: {e:?}")))?;
        unsafe {
            capsule.allocate_with(
                metadata,
                || {
                    match empty_extent {
                        // SAFETY: cold private whole-allocation birth, no raw/view escape;
                        // logical extent remains zero and every later native use must
                        // obey the same counted/closed-use contract as nonempty uploads.
                        Some(bytes) => self
                            .backings
                            .resident_shared
                            .alloc_custodied(bytes, context),
                        None => self
                            .backings
                            .resident_shared
                            .alloc_data_custodied(data, context),
                    }
                    .map_err(crate::Error::from)
                },
                self.commands.clock(),
                size,
                Some(Arc::clone(&self.residency_set)),
                &self.commands,
            )
        }
    }

    fn new_buffer_with_data_in<T>(
        &self,
        data: &[T],
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let unpooled_uploads = Self::unpooled;
        let size = core::mem::size_of_val(data);
        // Metal refuses a zero-length `newBufferWithBytes`, and an empty tensor
        // is legal: it gets the pool's minimum allocation, as `allocate_buffer`
        // gives any empty request (`buf_size`), with nothing to copy.
        if size == 0 {
            return self.allocate_buffer_in(0, context);
        }
        // A small upload is served from the buffer pool and filled on the host:
        // an acquired buffer has no GPU reader (reuse waits for completion,
        // §6.3b), so a host write into it is what `newBufferWithBytes` did
        // without the driver allocation, the residency `addAllocation` +
        // `commit` per call and the deferred remove — three Metal calls per
        // upload. The paged FlashDecoding arm made ~50 of these per decode
        // step (its params blocks, chunk table, live count and slot list)
        // where the crown's `setBytes` params make none; the pool's hit is a
        // lock and a hash (§3.3a). The large uploads below (weight staging,
        // whole prompts) keep the unpooled path for the reason it records.
        if size > 0 && size <= SMALL_UPLOAD_BYTES && !unpooled_uploads() {
            let buffer = self.allocate_buffer_in(size, context)?;
            // SAFETY: the pool handed out a buffer at least `size` bytes long
            // whose last GPU reader has completed, and `data` is `size` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), buffer.contents(), size);
            }
            crate::kernels::metal::upload_shadow::record(
                unsafe { buffer.contents() } as usize,
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size) },
            );
            return Ok(buffer);
        }
        let allocation = self.backings.resident_shared.alloc_data(data, context)?;

        // Deliberately not pooled.
        //
        // This path always calls `newBufferWithBytes` and never consults the
        // free list, so a buffer parked here is only reachable by an
        // `allocate_buffer` request for the same size -- and `allocate_buffer`
        // has three callers, all on quantized dequantize paths. Nothing in an
        // f16 forward pass ever asks for one.
        //
        // In practice these are weight tensors staged from safetensors, which
        // `to_dtype` immediately converts into `private_buffers`; the staging
        // buffer is then dropped and never wanted again. Measured on an LFM2
        // load that is 276 buffers holding 5145 MB. The old allocator reclaimed
        // them only as a side effect of its sweep, so with the sweep off the
        // allocation path, pooling them would double the footprint and undo
        // issue #8's 7731 -> 5509 MB.
        //
        // Freeing on drop keeps that reclamation without reintroducing a scan.
        // Note this is a property of the *path*, not of buffer size: these
        // average 18.6 MB, so a size threshold does not separate them from
        // activations.
        //
        // And it leaves the residency set on drop: the set retains what it
        // holds, so an unpooled upload that stayed in it was never freed at
        // all — neither these staging buffers nor a per-round `Tensor::new`.
        Ok(BufferHandle::new(
            PooledBuffer::unpooled_resident(
                allocation.buffer,
                size,
                Arc::clone(&self.residency_set),
            )
            .with_reservation(allocation.reservation)
            .held_in_flight(self.commands.in_flight_hold()),
        ))
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope
                .view
                .allocate_zeros_in(size_in_bytes, Some(&scope.context));
        }
        self.allocate_zeros_in(size_in_bytes, None)
    }

    fn allocate_zeros_in(
        &self,
        size_in_bytes: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer_in(size_in_bytes, context)?;
        if self.base_mode() {
            // A compute fill, inside the barrier chain (see `base_mode`).
            if size_in_bytes > 0 {
                let encoder = self.command_encoder()?;
                // SAFETY: `buffer` was just allocated here and is the caller's
                // storage from now on (binding authority, `mod.rs` head).
                unsafe {
                    crate::kernels::kernels::call_const_fill(
                        &self.device,
                        &encoder,
                        &self.kernels,
                        "fill_u8",
                        size_in_bytes,
                        &buffer,
                        0u8,
                    )
                }
                .map_err(crate::Error::wrap)?;
            }
            return Ok(buffer);
        }
        let mut blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        // SAFETY: as the compute fill above.
        unsafe { blit.fill_buffer(&buffer, (0, buffer.length()), 0) };
        Ok(buffer)
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<BufferHandle> {
        if let Some(scope) = super::allocation_scope::active(self)? {
            return scope.view.allocate_buffer_in(size, Some(&scope.context));
        }
        self.allocate_buffer_in(size, None)
    }

    fn allocate_buffer_in(
        &self,
        size: usize,
        context: Option<&crate::memory_admission::AllocationContext>,
    ) -> Result<BufferHandle> {
        let allocated = buf_size(size)?;
        self.note_pool_request(0);
        let origin = self.declared_allocation_origin();
        let miss = match self.buffers.acquire_for_allocation_in(size, &origin) {
            Ok(b) => return Ok(b),
            Err(miss) => miss,
        };
        // `miss` retains the requested size before rounding, so adoption can
        // account for the overhang (§9.2, #47).
        let size = allocated;
        // §9.5k's branch, on the pool miss and before the allocation -- see
        // `new_buffer` above. This is the pool the KV reserve is served from
        // (`KvSlot::append` -> `Tensor::zeros` -> here), which is why the
        // subtracted `planned` figure includes KV.
        self.buffers
            .check_residual(size)
            .map_err(AllocationError::from)?;
        let split = self.residency_set.policy().is_split();
        let new_buffer =
            Self::allocate_backing(self.backings.for_miss(false, split), size, context)?;
        Ok(self
            .buffers
            .adopt_after_miss_in(new_buffer, size, miss, &origin)?)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }

        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }
}

// `BUFFER_ALIGNMENT` (128), the axis's default, lives in `crate::tuning`
// beside the knob that overrides it: one definition.

/// The lowest alignment the axis will accept.
///
/// `float4`/`half4` are 16 bytes and quantized block structures are wider, so
/// **16 is the floor at which every dtype this backend uses is still correctly
/// aligned**. Below it the arm would not be a granularity choice but a
/// correctness bug, and §3.5 makes a misaligned vector load silent rather than
/// an error. The refusal names this number.
const MIN_BUFFER_ALIGNMENT: usize = 16;

/// The buffer alignment this process allocates at (`DESIGN.md` §9.2, #586).
///
/// # Why this is an axis and not a constant
///
/// §9.2 records the 128 as load-bearing in one direction — *"lowering the
/// alignment to 64 would silently introduce false sharing between adjacent
/// arena slots"* — and closes with *"do not change it without measuring
/// both"*. **There was no instrument to measure either half**: the value was a
/// bare `const` with no selector, so #443's paired A/B could not run and its
/// incumbent figure (+0.49 ms/token to save 2.2 GB) was measured against a
/// free-list scan removed the same day. This makes the arm exist; it does not
/// flip the default (§7.1a) and takes no timing.
///
/// # What this axis is NOT
///
/// **It is not §9.2l's padding**, and conflating the two is the trap that
/// section names. `CachePadded<T>` is `#[repr(align(128))]` over *control
/// state* — struct fields two threads write — and lowering **that** to 64 is a
/// compile error by design. This is *data* alignment: the granularity
/// `buf_size` rounds an allocation to. §9.2l's own words are that its rule is
/// *"not licence for any change to `buf_size` / `BUFFER_ALIGNMENT`, which is
/// #355's and #443's"* — so the two quantities are separated deliberately, and
/// this axis moves only the second.
///
/// # Read once, not per allocation
///
/// A `OnceLock`, for §15.2 #10's reason: `buf_size` is on the allocation path
/// at §6.3b's ~11.6 destroys/token, and an env read per call would put a
/// syscall-shaped cost on it. Reading once also makes the arm a property of the
/// process, which is what lets the harness report it (§7.1a-i: read the arm
/// back from the mechanism that decides it, never from the flag).
fn buffer_alignment() -> usize {
    // A power of two >= `MIN_BUFFER_ALIGNMENT`, refused otherwise by the
    // parser (`tuning.rs`); `BUFFER_ALIGNMENT` (128) is the default.
    crate::tuning::tuning()
        .pool
        .buffer_alignment
        .max(MIN_BUFFER_ALIGNMENT)
}

/// Size to allocate for a request of `size` bytes.
///
/// Previously this rounded up to the next power of two, which wastes up to 2x
/// per buffer. That is not hypothetical: for shapes that sit just above a
/// power of two -- a 292-token KV cache, or an FFN intermediate of 10752 --
/// the rounding costs 75% and 52% respectively, and the pool holds the excess
/// for the process lifetime.
///
/// Aligning instead keeps the guarantees that matter (vector alignment, and a
/// size the memory controller likes) without the waste. Reuse is unaffected:
/// `find_available_buffer` accepts any pooled buffer at least as large as the
/// request, so a slightly larger buffer still satisfies a smaller one.
fn buf_size(size: usize) -> Result<usize> {
    crate::allocation_geometry::aligned_nonzero_bytes(size, buffer_alignment())
}

/// Applies the [`BufferBuilder`] label, clearing any stale label on a reused pooled buffer.
#[cfg(feature = "metal-debug-labels")]
#[inline]
fn buffer_label(buffer: &Buffer, label: Option<&str>) {
    buffer.set_label(label.unwrap_or("unlabeled"));
}
#[cfg(not(feature = "metal-debug-labels"))]
#[inline]
fn buffer_label(_buffer: &Buffer, _label: Option<&str>) {}

type DataUpload<'a> = Box<dyn FnOnce(&MetalDevice) -> Result<BufferHandle> + 'a>;

enum BufferInit<'a> {
    Typed { elem_count: usize, dtype: DType },
    Size(usize),
    Zeros(usize),
    Data(DataUpload<'a>),
}

/// Builder for `MTLBuffer` allocations; pool reuse handled by [`MetalDevice`].
pub struct BufferBuilder<'a> {
    device: &'a MetalDevice,
    label: Option<&'a str>,
}

/// [`BufferBuilder`] with an init kind set; `build()` lives here.
pub struct ReadyBufferBuilder<'a> {
    device: &'a MetalDevice,
    init: BufferInit<'a>,
    label: Option<&'a str>,
}

impl<'a> BufferBuilder<'a> {
    fn new(device: &'a MetalDevice) -> Self {
        Self {
            device,
            label: None,
        }
    }

    /// Allocate elem_count * dtype size bytes, uninitialized, private storage.
    pub fn with_size_for(self, elem_count: usize, dtype: DType) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Typed { elem_count, dtype })
    }

    /// Allocate size bytes, uninitialized, shared storage.
    pub fn with_size(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Size(size))
    }

    /// Allocate size bytes, zero-filled, shared storage. Pool rounding may make
    /// the allocation larger than size; the extra bytes are also zeroed.
    pub fn with_zeros(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Zeros(size))
    }

    /// Allocate a shared buffer initialized from data. Always allocates; does not
    /// reuse the pool.
    pub fn with_data<T>(self, data: &'a [T]) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Data(Box::new(move |device| {
            device.new_buffer_with_data(data)
        })))
    }

    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[inline]
    fn ready(self, init: BufferInit<'a>) -> ReadyBufferBuilder<'a> {
        ReadyBufferBuilder {
            device: self.device,
            init,
            label: self.label,
        }
    }
}

impl<'a> ReadyBufferBuilder<'a> {
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn build(self) -> Result<BufferHandle> {
        let buffer = match self.init {
            BufferInit::Typed { elem_count, dtype } => {
                self.device.new_buffer(elem_count, dtype, "")?
            }
            BufferInit::Size(size) => self.device.allocate_buffer(size)?,
            BufferInit::Zeros(size) => self.device.allocate_zeros(size)?,
            BufferInit::Data(upload) => upload(self.device)?,
        };
        buffer_label(&buffer, self.label);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounding, as a pure function of the granularity.
    ///
    /// `buf_size` reads a process-wide `OnceLock`, so a test cannot vary the arm
    /// in-process without the first read winning — the same property that keeps
    /// the alignment off the per-allocation path. This mirrors the arithmetic so
    /// every arm is exercised, and
    /// `the_default_arm_agrees_with_buf_size` pins the mirror to the real
    /// function at the shipped arm so the two cannot drift (§3.1b rule 1: an
    /// absolute reference, not a shape).
    fn round_to(size: usize, alignment: usize) -> usize {
        size.max(1).next_multiple_of(alignment)
    }

    #[test]
    fn test_buf_size_is_aligned() {
        let alignment = buffer_alignment();
        for size in [1usize, 2, 3, 127, 128, 129, 1000, 4096, 21504, 299008] {
            let got = buf_size(size).unwrap();
            assert!(got >= size, "buf_size({size}) = {got} is too small");
            assert_eq!(
                got % alignment,
                0,
                "buf_size({size}) = {got} is not {alignment}-aligned"
            );
            assert!(
                got - size < alignment,
                "buf_size({size}) = {got} wastes a whole alignment unit"
            );
        }
    }

    #[test]
    fn test_buf_size_never_zero() {
        // Metal rejects zero-length buffers, and an empty tensor asks for one.
        assert_eq!(buf_size(0).unwrap(), buffer_alignment());
    }

    #[test]
    fn test_buf_size_avoids_power_of_two_waste() {
        // Shapes that sit just above a power of two are the ones that suffered:
        // a 292-token KV cache and an FFN intermediate both rounded up hard.
        // Both are multiples of 128, so they are exact at every arm at or below
        // the default -- which is the arithmetic §9.2l relies on when it says
        // padding is free at tensor granularity "by arithmetic rather than by
        // luck".
        assert_eq!(buf_size(299008).unwrap(), 299008); // was 524288, 75% waste
        assert_eq!(buf_size(21504).unwrap(), 21504); // was 32768, 52% waste
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element; a scalar tensor requests a
        // 2-byte buffer. It must not be rounded down.
        assert!(buf_size(2).unwrap() >= 2);
    }

    #[test]
    fn the_default_arm_agrees_with_buf_size() {
        // The mirror above is only worth having if it is the same arithmetic.
        // Pinned at the arm the test process actually resolved, so this holds
        // whether or not the environment selects one.
        let alignment = buffer_alignment();
        for size in [1usize, 2, 127, 128, 129, 21504, 299008] {
            assert_eq!(
                buf_size(size).unwrap(),
                round_to(size, alignment),
                "the mirror and buf_size disagree at {size} (alignment {alignment})"
            );
        }
    }

    /// The arm's *effect*, at an absolute reference rather than a shape.
    ///
    /// §3.1b rule 1: `[3, 3, 3]` is a stronger assertion than *all equal*. A
    /// test asserting only *"a larger alignment wastes at least as much"* would
    /// pass on a `buf_size` that ignored the arm entirely, because a constant
    /// sequence is monotone. These are the exact byte counts.
    #[test]
    fn each_arm_rounds_to_its_own_granularity() {
        // 4104 is deliberately awkward: a multiple of 8 and of none of the
        // arms, so every arm rounds it somewhere different.
        assert_eq!(round_to(4104, 16), 4112);
        assert_eq!(round_to(4104, 64), 4160);
        assert_eq!(round_to(4104, 128), 4224);
        assert_eq!(round_to(4104, 256), 4352);
        assert_eq!(round_to(4104, 4096), 8192);
    }

    #[test]
    fn a_page_sized_request_is_exact_at_every_arm_at_or_below_it() {
        // The reason the canonical digests cannot move under this axis: every
        // LFM2 decode activation shape is already a multiple of 128 (§9.2l), so
        // an arm at or below the default changes no allocation size at all. The
        // arena's five slots, measured by #489.
        for &size in &[21504usize, 21504, 21504, 4096, 1024] {
            for &alignment in &[16usize, 32, 64, 128, 256, 512, 1024] {
                if size % alignment == 0 {
                    assert_eq!(
                        round_to(size, alignment),
                        size,
                        "arena slot {size} should be exact at alignment {alignment}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_floor_is_the_widest_vector_this_backend_loads() {
        // Not a round number chosen for neatness: `float4`/`half4` are 16 bytes
        // and are what the kernels load. The refusal below this value is a
        // correctness guard, so the constant must not drift away from the
        // reason for it.
        assert_eq!(MIN_BUFFER_ALIGNMENT, 16);
        assert_eq!(MIN_BUFFER_ALIGNMENT, std::mem::size_of::<f32>() * 4);
    }
}

/// Whether weight uploads are wired (`LLOOM_WIRE_WEIGHTS`, read once; `1`
/// / `on` wires, the default does not — the A/B of #749 P29: the dense
/// weights are Shared uploads, and the lm_head's 437 MB read at 5–35 ms a
/// token with the machine at free ≈ 0.1 GB).
pub fn wire_weights_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().metal.wire_weights)
}

/// Closed initialized F32 input, created only from the actual rich upload.
/// Geometry is logical input length; raw SDK access remains unsafe separately.
pub(crate) struct PreparedAffineInput {
    handle: BufferHandle,
    elements: usize,
}
impl PreparedAffineInput {
    pub(crate) fn handle(&self) -> &BufferHandle {
        &self.handle
    }
    pub(crate) fn elements(&self) -> usize {
        self.elements
    }
}

#[cfg(all(test, any(debug_assertions, feature = "profile")))]
mod attribution_build_tests {
    #[test]
    fn served_attribution_collects_a_real_backtrace_without_a_device() {
        let symbol_control = std::backtrace::Backtrace::force_capture().to_string();
        assert!(
            symbol_control
                .contains("served_attribution_collects_a_real_backtrace_without_a_device"),
            "symbolization positive control: {symbol_control}"
        );
        eprintln!("symbolization positive control:\n{symbol_control}");
        super::set_served_request_trace(false).unwrap();
        assert!(!super::served_request_trace_on());
        super::set_served_request_trace(true).unwrap();
        assert!(super::served_request_trace_on());
        let _ = super::served_pass_request_sites();
        super::note_served_request_site(0);
        super::note_served_request_site(0);
        super::note_served_request_site(1);
        let sites = super::served_pass_request_sites();
        assert_eq!(sites.len(), 2, "{sites:?}");
        assert_eq!(sites[0].0, 0);
        assert_eq!(sites[0].2, 2);
        assert_eq!(
            sites[0].1, "(no lloom frame)",
            "this host-only caller is in lloom_tensor, not a model"
        );
        assert_eq!(sites[1].0, 1);
        assert_eq!(sites[1].2, 1);
        assert!(super::served_pass_request_sites().is_empty());
        super::set_served_request_trace(false).unwrap();
    }
}
