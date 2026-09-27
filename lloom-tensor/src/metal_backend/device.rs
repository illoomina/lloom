use crate::{DType, Result};

use crate::kernels::metal::CommandMarker;
#[cfg(feature = "ug")]
use crate::kernels::metal::ComputePipeline;
use crate::kernels::{
    metal::{
        session_region::{RegionCounters, RegionPlan},
        transient_heaps_on, Arena, ArenaCounters, ArenaCursor, ArenaDecline, ArenaLayout,
        ArenaOffsets, ArenaRecorder, BlitCommandsGuard, Buffer, BufferPool, Commands,
        CommandsGuard, Device, FreeSummary, HeapChunks, HeapStats, MTLResourceOptions, PassKey,
        PassKind, PassPlanCounters, PoolCounters, PoolOccupancySnapshot, PooledBuffer,
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

/// Runs the residency-set teardown when the **last** [`MetalDevice`] handle
/// drops (`DESIGN.md` §6.3c, issue #163).
///
/// # Why this is a separate type rather than `impl Drop for MetalDevice`
///
/// `MetalDevice` is `Clone` and every field is shared, so every `Tensor` holds a
/// clone through its `MetalStorage`. A `Drop` written on `MetalDevice` itself
/// would therefore fire on **every tensor drop** -- hundreds of times per decode
/// token -- and each firing would wait for the GPU. That is not a teardown
/// guard, it is a synchronize on the hot path, and §11.2's 6.1 % non-GPU budget
/// makes it a regression rather than a subtlety.
///
/// Holding the guard behind an `Arc` moves the trigger to the instant the last
/// handle goes, which is what the panic log shows actually happening:
/// `main+5200` -> `drop_in_place<MetalDevice>` -> `Arc::drop_slow`, on the main
/// thread past the `RESULT` line.
///
/// # Why it is declared first
///
/// Fields drop in declaration order, and this must run **before** `commands`,
/// `buffers` and `private_buffers` -- the pools are what free the `MTLBuffer`s
/// the set still lists, and removing an object that is already gone is what
/// `IOGPUGroupMemory::remove_memory_object()` aborts the machine over. The
/// original defect *is* a declaration-order property, so the fix has to be one
/// too.
pub(crate) struct DeviceTeardown {
    commands: Arc<Commands>,
    residency_set: Arc<ResidencySet>,
}

/// Unregisters an evicted buffer from the residency set, so Metal releases the
/// allocation the pool has just destroyed (`DESIGN.md` §6.3c, §6.3e).
///
/// The pool bounds its free list by **destroying** buffers whose size will not
/// be asked for again (§6.3b), which is a second place a buffer's existence
/// ends -- and unlike a trim, nothing was telling the residency set about it.
///
/// # Why this calls `removeAllocation` as of #210, having retired the key before
///
/// #166 retired the membership key **without** the Metal call, on a measurement:
/// eager unregistration cost **+0.062 ms/token even batched** against §11.2's
/// whole 6.1 % non-GPU budget, and the residual was named *"a retention, not a
/// dangling reference"*. Both halves were right. **What was never asked is what
/// the retention weighs**, and the answer is **48 GB** (§6.3e, #206): the
/// pricing was taken at §6.3b's **11.6 evictions per token**, and under
/// `KvAppend=Cat` at long context the rate is **~220 per token**, at which the
/// shipping default exhausts a 64 GB device at 97 % of
/// `recommendedMaxWorkingSetSize`.
///
/// **The structural fact that decides the shape, and it is read from this file
/// rather than argued.** §6.3e offers three: eager always, eager above a
/// retention threshold, or a sweep folded into the per-command-buffer
/// completion handler. **The first and the third are the same code.**
/// `on_evict` is reached only from `PoolInner::drain_completed`, which is
/// subscribed to `Commands::on_command_buffer_complete` -- so every
/// steady-state eviction **already runs inside** that handler, once per command
/// buffer, batched over everything it freed. There is no third site to fold
/// anything into and no sweep to add; `evict_over_budget`'s only other caller is
/// `set_free_budget`, which is setup-time.
///
/// **And the commit is not additional in the sense the cost argument assumed.**
/// `ResidencySet::insert` is one buffer and one `commit()`, called from every
/// pool miss -- so a residency commit is *already* on the per-token path at the
/// allocation rate, and what a batched eager release adds is **one more per
/// eviction round**, not one per evicted buffer. §6.3e's *"the `commit()` may
/// not be additional"* is right about the mechanism and located it in the wrong
/// place: it is the *allocation* commit that was already there.
///
/// **The threshold shape is rejected on evidence rather than on taste.** It
/// needs a retention figure nobody has measured a threshold against, and §9.1a
/// is the worked case of what that costs -- three policies, all built, none
/// chosen, because the separating axis was never exercised. A threshold here
/// would also be a knob whose wrong setting reproduces exactly this defect
/// silently, where the unconditional form has no setting to get wrong.
///
/// # The measurement that put it here
///
/// #166's figures, which stand and are the bar this had to beat:
///
/// ```text
///   baseline                                        0.928  0.936  0.932
///   unregister eagerly, one call per buffer         1.009  1.016  1.050   +0.087
///   unregister eagerly, one call per batch          0.993  0.998  0.993   +0.062
///   retire the key, remove at teardown              0.941  0.928  0.945    +0.00
/// ```
///
/// **A per-allocation Metal call is forbidden and this is not one** (§9.5e's
/// premise, from #167's own ablation: *accounting is cheap, calling into Metal
/// is not*). The call is per eviction **round**, batched by
/// `BufferEvictionObserver::on_evict`'s contract, which exists for exactly this
/// reason -- #166 measured the per-buffer form at +0.087 and the batched form at
/// +0.062, and the difference between them is the commit count.
pub(crate) struct ResidencyEvictionObserver {
    residency_set: Arc<ResidencySet>,
    /// The transient chunks (§6.3k): an evicted sub-buffer's range goes
    /// back to its chunk instead of out of the residency set.
    heaps: Arc<(HeapChunks, HeapChunks)>,
}

impl ResidencyEvictionObserver {
    pub(crate) fn new(
        residency_set: Arc<ResidencySet>,
        heaps: Arc<(HeapChunks, HeapChunks)>,
    ) -> Self {
        Self {
            residency_set,
            heaps,
        }
    }
}

impl crate::kernels::metal::BufferEvictionObserver for ResidencyEvictionObserver {
    fn on_evict(&self, buffers: &[Buffer]) {
        // One Metal call per eviction ROUND, never per buffer: `on_evict` is
        // handed everything one round destroys, and `remove_batch` issues one
        // `commit()` for the lot.
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
            .filter(|b| !(self.heaps.0.release(b) || self.heaps.1.release(b)))
            .collect();
        if rest.is_empty() {
            return;
        }
        if retain_evicted_allocations() {
            self.residency_set.retire_batch(rest.into_iter());
        } else {
            self.residency_set.remove_batch(rest.into_iter());
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
fn retain_evicted_allocations() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // `CANDLE_METAL_EAGER_RESIDENCY_RELEASE=0` is #206's probe spelled the
        // other way round, and it selected the arm that is now the default.
        // Honoured so an existing script that pins the arm explicitly still
        // selects the arm it names rather than silently getting the new default
        // -- an A/B whose arms both moved is worse than one that fails.
        if let Ok(v) = std::env::var("CANDLE_METAL_EAGER_RESIDENCY_RELEASE") {
            return !(v == "1" || v.eq_ignore_ascii_case("true"));
        }
        crate::tuning::tuning().pool.retain_evicted
    })
}

impl DeviceTeardown {
    pub(crate) fn new(commands: Arc<Commands>, residency_set: Arc<ResidencySet>) -> Self {
        Self {
            commands,
            residency_set,
        }
    }

    /// How many `MetalDevice` handles share this guard.
    ///
    /// The guard fires when this reaches zero, so a test can assert *when* it
    /// will run rather than that it did -- the defect's natural expression is a
    /// machine panic, which is not a testable assertion, so the mechanism is
    /// what gets tested (issue #166).
    pub(crate) fn handles(self: &Arc<Self>) -> usize {
        Arc::strong_count(self)
    }
}

impl Drop for DeviceTeardown {
    fn drop(&mut self) {
        // 1. Wait. This is the load-bearing step and the one `Drop for Commands`
        //    does not do: it calls `flush()`, which commits *without* waiting
        //    (`flush_and_wait` is the one that waits), so without this the pools
        //    are destroyed while GPU work is still in flight -- §6.3c's
        //    aggravating condition, and §6.7 L4's rule that a decision about a
        //    buffer's existence taken on a different clock from the one that
        //    orders execution is a correctness bug.
        //
        //    A failure here is reported and not propagated: this is a
        //    destructor, and unwinding out of one during teardown would abort.
        //    Proceeding to unregister after a failed wait is still strictly
        //    better than not unregistering at all -- the set is emptied either
        //    way, and it is the *absent object* that panics the kernel, not an
        //    outstanding command buffer.
        if let Err(e) = self.commands.wait_until_completed() {
            // `eprintln!` rather than a panic or a log dependency: candle-core
            // takes no logger, and a destructor must not unwind.
            eprintln!("candle: MetalDevice teardown: waiting for GPU work failed: {e}");
        }

        // 2. Unregister, while the pools still hold the buffers. Emptying the
        //    set here is what makes the later teardown of the Metal object a
        //    no-op instead of a removal of objects that no longer exist.
        self.residency_set.remove_all();

        // 3. The pools drop after this, freeing buffers the set no longer
        //    lists. That ordering is the whole fix and it is a property of
        //    where this field is declared, not of anything written here.
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Argument blocks and small tables keyed by their bytes, so a value a
    /// kernel is handed every step — a params block, a chunk table, a live
    /// count — is uploaded once and bound by the same buffer until it
    /// changes. Declared first so it drops first: its buffers return to the
    /// pool before the teardown below waits the GPU out.
    pub(crate) constants: Arc<ConstantsByValue>,
    /// Empties the residency set, after waiting for the GPU, when the last
    /// handle to this device drops. Declared **first** so it runs before the
    /// pools free the buffers the set lists (`DESIGN.md` §6.3c).
    pub(crate) teardown: Arc<DeviceTeardown>,

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

    /// Where a pool miss's bytes come from when `transient_heaps_on()`: heap
    /// chunks in the residency set once (`DESIGN.md` §6.3k), one list per
    /// storage mode (the Shared pool's, the Private pool's).
    pub(crate) transient_heaps: Arc<(HeapChunks, HeapChunks)>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`crate::kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
    /// Residency set registered on the command queue.
    pub(crate) residency_set: Arc<ResidencySet>,

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
    /// `finish_pass`). Bit 0 = the cache is on, bit 1 = a pass is open. The
    /// recorder takes an allocation only while a pass is open: one made
    /// BETWEEN passes (a sampler's ids, a staged input) is nobody's ordinal
    /// — recorded, it would be charged to the pass before it and, read by
    /// the pass after it, counted as that pass's escape (measured: exactly
    /// one per batched step, the argmax's ids).
    pub(crate) pass_state: Arc<std::sync::atomic::AtomicU8>,
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
    layout: ArenaLayout,
    plans: HashMap<PassKey, StepPlan>,
    /// Arenas built over the current base, per key; rebuilt lazily after a
    /// base grow (an `Arena` holds views of the base it was built on).
    arenas: HashMap<PassKey, Arena>,
    base: Option<Buffer>,
    base_generation: u64,
    arena_generation: HashMap<PassKey, u64>,
    /// A key recorded on its first sight, whose plan awaits the next pass.
    pending: Option<PassKey>,
    /// What the pass in progress is doing.
    current: Option<PassState>,
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
            layout,
            plans: HashMap::new(),
            arenas: HashMap::new(),
            base: None,
            base_generation: 0,
            arena_generation: HashMap::new(),
            pending: None,
            current: None,
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
    map: HashMap<Vec<u8>, Arc<PooledBuffer>>,
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
    pub fn trim_unused_buffers(&self) {
        for buffer in self.buffers.trim() {
            if !self.transient_heaps.0.release(&buffer) {
                self.residency_set.remove(&buffer);
            }
        }
        for buffer in self.private_buffers.trim() {
            if !self.transient_heaps.1.release(&buffer) {
                self.residency_set.remove(&buffer);
            }
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

    /// Live and free occupancy for both pools, shared first.
    pub fn pool_occupancy(&self) -> (PoolOccupancySnapshot, PoolOccupancySnapshot) {
        (self.buffers.occupancy(), self.private_buffers.occupancy())
    }

    /// The transient heap chunks' `(shared, private)` stats (§6.3k).
    pub fn heap_stats(&self) -> (HeapStats, HeapStats) {
        (
            self.transient_heaps.0.stats(),
            self.transient_heaps.1.stats(),
        )
    }

    pub fn command_encoder<'a>(&'a self) -> Result<CommandsGuard<'a>> {
        let command_encoder = self.commands.command_encoder().map_err(MetalError::from)?;
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
        let command_encoder = self
            .commands
            .blit_command_encoder()
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    /// The command stream, for a subsystem that orders its own queue against
    /// it (`kernels::metal::io_queue`).
    pub fn commands(&self) -> &Arc<Commands> {
        &self.commands
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
        self.commands.bind_eos_gate_current();
        self.kernels.set_eos_gated(true);
        Ok(())
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
    /// The speculative round's readback as a publish (`words_publish`): `n`
    /// u32 words of `src` copied into the caller's CPU-mapped `out`, then
    /// `generation` stamped into `step` behind a device barrier. The host
    /// polls the stamp and reads the words in place — no completion-handler
    /// wait, no `to_cpu` (#148).
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
        let encoder = self.command_encoder()?;
        crate::kernels::call_words_publish(
            &self.device,
            &encoder,
            &self.kernels,
            bo(src.0, src.1),
            n,
            bo(out.0, out.1),
            bo(step.0, step.1),
            generation,
        )
        .map_err(MetalError::from)?;
        Ok(())
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
        crate::kernels::call_ring_stamp(
            &self.device,
            &encoder,
            &self.kernels,
            bo(step.0, step.1),
            generation,
            certified.map(|(s, off)| bo(s, off)),
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    pub fn eos_gate_end(&self) -> crate::EosGateReport {
        self.kernels.set_eos_gated(false);
        self.commands.eos_gate().end()
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
        let plan = RegionPlan::reserve(reservations);
        let bytes = plan.bytes().max(1);
        let base = self
            .device
            .new_buffer(bytes, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&base);
        let region = SessionRegion::new(base, plan)
            .map_err(|e| MetalError::Message(format!("session region rejected: {e}")))?;
        {
            let mut slot = self.session_region.write();
            // Unregister the region being replaced while it still owns its base
            // — #489's ordering argument, which applies here for the same reason
            // it applies to the arena.
            if let Some(previous) = slot.as_ref() {
                self.residency_set.remove(previous.base());
            }
            *slot = Some(region);
        }
        Ok(bytes)
    }

    /// Remove the session region. Subsequent allocations take the pool path.
    ///
    /// Ordered as [`Self::clear_arena`] is: the base leaves the residency set
    /// **while the region still owns it**, so `removeAllocation` names an object
    /// that provably still exists (§6.3c, #489).
    pub fn clear_session_region(&self) {
        let mut slot = self.session_region.write();
        if let Some(region) = slot.as_ref() {
            self.residency_set.remove(region.base());
        }
        *slot = None;
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
        let bytes = plan.bump_capacity().max(plan.arena_bytes()).max(1);
        let base = self
            .device
            .new_buffer(bytes, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&base);
        let arena = Arena::new(base, plan, layout)
            .map_err(|e| MetalError::Message(format!("arena plan rejected: {e}")))?;
        {
            let mut slot = self.arena.write();
            // Unregister the arena being *replaced*, on the same argument as
            // `clear_arena` and while it still owns its base (issue #489). This
            // is latent today because nothing in the shipping engine replaces an
            // arena -- one is installed and lives for the process -- but under
            // #483's ring or any per-phase arena (§9.2's sizing discussion) the
            // leak would be per replacement rather than one-off.
            if let Some(previous) = slot.as_ref() {
                self.residency_set.remove(previous.base());
            }
            *slot = Some(arena);
        }
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
        }
        *slot = None;
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
        self.arena_auto_report.lock().clone()
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

    pub fn pass_plans_enabled(&self) -> bool {
        self.pass_plans.lock().is_some()
    }

    /// Whether a plan for `key` is cached: the pass will be served from the
    /// base, allocating none of its activations (its admission owes no
    /// transient).
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

    /// Open a forward pass of shape `key`. Under a cached plan the pass is
    /// served from the base; on a key's first sight, and on the pass after
    /// it, the pass runs on the pool under the recorder. Falls back to
    /// [`Self::begin_decode_step`] when the cache is off.
    /// Whether a planned pass is open (`begin_pass` without its
    /// `finish_pass`): a model that brackets its own passes skips the
    /// bracket when a driver already holds one.
    pub fn pass_open(&self) -> bool {
        self.pass_plans
            .lock()
            .as_ref()
            .is_some_and(|pp| pp.current.is_some())
    }

    pub fn begin_pass(&self, key: PassKey) -> Result<()> {
        let mut g = self.pass_plans.lock();
        let Some(pp) = g.as_mut() else {
            drop(g);
            self.begin_decode_step();
            return Ok(());
        };
        if pp.current.is_some() {
            return Err(MetalError::Message(
                "begin_pass: a pass is already open (finish_pass was not called)".to_string(),
            )
            .into());
        }
        self.pass_state
            .fetch_or(0b10, std::sync::atomic::Ordering::Relaxed);
        // Whatever this pass does, the arena the last pass installed must not
        // serve it: a pass is served only under its own key's plan.
        self.clear_arena_keep_base();
        let state = match (pp.pending, pp.plans.contains_key(&key)) {
            (Some(closes), cached) => {
                // The recorder still holds the pending key's pass as its
                // current step; this pass becomes the step that says what
                // escaped it.
                self.next_arena_recording_step();
                pp.counters.observed += 1;
                if cached {
                    PassState::Observing { closes }
                } else {
                    PassState::Recording {
                        key,
                        closes: Some(closes),
                    }
                }
            }
            (None, true) => {
                let generation = pp.base_generation;
                let stale = pp.arena_generation.get(&key) != Some(&generation);
                if stale || !pp.arenas.contains_key(&key) {
                    let plan = pp.plans.get(&key).expect("cached").clone();
                    let base = pp.base.clone().expect("a cached plan has a base");
                    let arena = Arena::new(base, plan, pp.layout)
                        .map_err(|e| MetalError::Message(format!("pass plan rejected: {e}")))?;
                    pp.arenas.insert(key, arena);
                    pp.arena_generation.insert(key, generation);
                }
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
                PassState::Served(key)
            }
            (None, false) => {
                self.begin_arena_recording();
                pp.counters.recorded += 1;
                PassState::Recording { key, closes: None }
            }
        };
        pp.current = Some(state);
        Ok(())
    }

    /// Close the pass `begin_pass` opened: build the plan a pending key was
    /// waiting for, grow the base if it needs more, and register this pass's
    /// key as pending if it was recorded. Falls back to
    /// [`Self::finish_decode_step`] when the cache is off.
    pub fn finish_pass(&self) -> Result<()> {
        let mut g = self.pass_plans.lock();
        let Some(pp) = g.as_mut() else {
            drop(g);
            return self.finish_decode_step();
        };
        let Some(state) = pp.current.take() else {
            return Err(MetalError::Message("finish_pass: no pass is open".to_string()).into());
        };
        self.pass_state
            .fetch_and(!0b10, std::sync::atomic::Ordering::Relaxed);
        // A plan closed this pass: the recording pass's transients sit in
        // the pool's free list while the base now holds the pass's shape —
        // trimmed after the lock, or a budget that admitted one copy of the
        // transient carries two (measured: the served second prefill
        // refused at 36.62 GB against 36.18, #749 P28).
        let mut closed_a_plan = false;
        match state {
            PassState::Served(key) => {
                self.end_decode_step();
                if let Some(arena) = self.arena_slot() {
                    if arena.take_drifted() {
                        // The sequence moved under this key: the plan is wrong
                        // about it. Drop it; the next sight re-records.
                        pp.plans.remove(&key);
                        pp.arenas.remove(&key);
                        pp.counters.evicted += 1;
                    }
                }
                self.clear_arena_keep_base();
            }
            PassState::Recording { key, closes } => {
                if let Some(closes) = closes {
                    Self::close_pending(self, pp, closes)?;
                    closed_a_plan = true;
                }
                if pp.plans.contains_key(&key) {
                    // The key it closed was its own (two sightings back to
                    // back): planned now, nothing left to observe.
                    pp.pending = None;
                    self.finish_arena_recording(pp.layout);
                } else {
                    // This pass's allocations are the recorder's current
                    // step; the next pass says which of them escape.
                    pp.pending = Some(key);
                }
            }
            PassState::Observing { closes } => {
                Self::close_pending(self, pp, closes)?;
                closed_a_plan = true;
                pp.pending = None;
                self.finish_arena_recording(pp.layout);
            }
        }
        drop(g);
        if closed_a_plan {
            self.trim_unused_buffers();
        }
        Ok(())
    }

    /// Build `closes`'s plan from the recorder's previous step (its first
    /// sight) and this step's binds against it, and cache it.
    fn close_pending(&self, pp: &mut PassPlans, closes: PassKey) -> Result<()> {
        let (plan, undeclared) = {
            let g = self.arena_recorder.lock();
            let rec = g.as_ref().ok_or_else(|| {
                MetalError::Message("finish_pass: no recorder while a key was pending".to_string())
            })?;
            let r = rec.lock();
            (r.plan_previous(pp.layout), r.previous_undeclared_ordinals())
        };
        pp.counters.escapes += undeclared.len() as u64;
        pp.escapes.insert(closes, undeclared);
        let Some(plan) = plan else {
            // The previous pass allocated nothing worth planning; forget it.
            return Ok(());
        };
        let needed = plan.bump_capacity().max(plan.arena_bytes()).max(1);
        let have = pp.base.as_ref().map_or(0, |b| b.length());
        if needed > have {
            // Admitted before it is allocated (`DESIGN.md` §9.5): the base is
            // held for the run, so a grow the machine cannot take must stop
            // the run here rather than after the allocation. The OLD base is
            // still held at this point and is released just below, so the
            // grow charged is the difference — what the process's footprint
            // actually rises by.
            if pp.admit_fraction > 0.0 {
                let grow = needed.saturating_sub(have);
                let allocated = self.device.current_allocated_size();
                let working_set = self.device.recommended_max_working_set_size();
                let budget = (working_set as f64 * pp.admit_fraction) as usize;
                if allocated.saturating_add(grow) > budget {
                    let gb = |b: usize| b as f64 / 1e9;
                    return Err(MetalError::Message(format!(
                        "pass plan base grow refused for {:?}: it needs {:.3} GB more, and the \
                         process already holds {:.2} GB of the {:.2} GB it may ({:.2} x \
                         recommendedMaxWorkingSetSize {:.2} GB). Nothing was allocated and no \
                         plan was cached; the pass runs on the pool as it did before \
                         (DESIGN.md 9.5, 3.5). Raise the fraction, narrow the pass (a smaller \
                         prefill tile), or bound the run.",
                        closes.kind,
                        gb(grow),
                        gb(allocated),
                        gb(budget),
                        pp.admit_fraction,
                        gb(working_set),
                    ))
                    .into());
                }
            }
            let base = self
                .device
                .new_buffer(needed, PRIVATE_RESOURCE_OPTIONS)
                .map_err(MetalError::from)?;
            self.residency_set.insert(&base);
            if let Some(previous) = pp.base.take() {
                self.residency_set.remove(&previous);
            }
            pp.base = Some(base);
            pp.base_generation += 1;
            pp.arenas.clear();
            pp.counters.base_grows += 1;
        }
        pp.plans.insert(closes, plan);
        pp.counters.plans += 1;
        if matches!(closes.kind, PassKind::Ragged { .. }) {
            pp.ragged_order.push_back(closes);
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
        Ok(())
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
        // The encoder observes binds, which is where an interval ends, so it
        // needs the same recorder this device is filling.
        crate::kernels::metal::arena::set_bind_observer(Some(Arc::clone(&rec)));
        {
            let mut g = self.arena_recorder.lock();
            *g = Some(rec);
        }
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
        crate::kernels::metal::arena::set_bind_observer(None);
        let mut g = self.arena_recorder.lock();
        let rec = g.take()?;
        let r = rec.lock();
        if r.is_empty() {
            return None;
        }
        Some(r.plan(layout))
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
    ) -> Result<Arc<PooledBuffer>> {
        let size = element_count * dtype.size_in_bytes();
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
        let buffer = match self.private_buffers.acquire(size) {
            Some(b) => b,
            None => {
                let alloc = buf_size(size);
                // `DESIGN.md` §9.5k's one branch, on the derived residual. It
                // sits on the pool MISS, after `acquire` has declined, because
                // a hit allocates nothing and there is nothing to bound; and
                // BEFORE `new_buffer`, because reporting an overrun after the
                // allocation is committed is useless. Inert with no cap
                // installed, which is every process that has not run admission.
                self.private_buffers
                    .check_residual(alloc)
                    .map_err(|e| MetalError::Message(e.to_string()))?;
                let new_buffer = if transient_heaps_on() {
                    // From a heap chunk: resident with the chunk, no residency
                    // call here and none at the cut (§6.3k).
                    self.transient_heaps
                        .1
                        .alloc(alloc)
                        .map_err(MetalError::from)?
                } else {
                    let b = self
                        .device
                        .new_buffer(alloc, PRIVATE_RESOURCE_OPTIONS)
                        .map_err(MetalError::from)?;
                    // `add`, not `insert`: stage the allocation (one `addAllocation`)
                    // and let the next command-buffer cut commit the residency set
                    // once, rather than committing per missed buffer. A cold-pool
                    // prefill misses ~a hundred buffers; §6.3b prices each residency
                    // commit at ~0.087 ms, so batching them to one commit is the
                    // whole of the first-forward residency cost (`commit_swap_locked`
                    // calls `commit_pending` before it commits, so the buffer is
                    // resident before any dispatch that binds it executes, §3.7).
                    self.residency_set.add(&b);
                    b
                };
                // `size` is the pre-rounding request and `alloc` is what Metal
                // was given; `adopt_requesting` counts both so §9.2's residual
                // overhang is derivable. The rounding happened at `alloc`
                // above, which is the last point where the two are both in
                // scope (#47).
                self.private_buffers
                    .adopt_requesting(new_buffer, size, alloc)
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
    fn note_for_recording(&self, buffer: Arc<PooledBuffer>, size: usize) -> Arc<PooledBuffer> {
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
    ) -> Result<Arc<PooledBuffer>> {
        let size = element_count * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&buffer);
        // Deliberately not pooled: this exists for persistent buffers, so it
        // returns a handle with no pool behind it. Dropping it frees the buffer
        // outright rather than offering it for reuse.
        Ok(Arc::new(PooledBuffer::unpooled(buffer, size)))
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
    pub fn constant_buffer(&self, bytes: &[u8]) -> Result<Arc<PooledBuffer>> {
        if bytes.len() > CONSTANT_MAX_BYTES {
            return self.new_buffer_with_data(bytes);
        }
        {
            let mut g = self.constants.inner.lock();
            if let Some(b) = g.map.get(bytes) {
                let b = Arc::clone(b);
                g.hits += 1;
                return Ok(b);
            }
        }
        let buf = self.new_buffer_with_data(bytes)?;
        let mut g = self.constants.inner.lock();
        g.misses += 1;
        if g.map.len() >= CONSTANT_ENTRIES {
            if let Some(old) = g.order.pop_front() {
                g.map.remove(&old);
            }
        }
        g.map.insert(bytes.to_vec(), Arc::clone(&buf));
        g.order.push_back(bytes.to_vec());
        Ok(buf)
    }

    /// [`Self::constant_buffer`] for one plain-data value (a params block).
    pub fn constant_of<T: Copy>(&self, v: &T) -> Result<Arc<PooledBuffer>> {
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
    pub fn new_unpooled_shared(&self, size: usize) -> Result<Arc<PooledBuffer>> {
        let buffer = self
            .device
            .new_buffer(size.max(1), RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&buffer);
        Ok(Arc::new(PooledBuffer::unpooled_resident(
            buffer,
            size,
            Arc::clone(&self.residency_set),
        )))
    }

    /// A Private buffer of `size` bytes that lives as long as its handle —
    /// outside every pool, registered in the residency set and unregistered
    /// when the handle drops. For a persistent structure of known geometry
    /// (a streamed tier's slot pool) that must not be a pool bucket.
    /// As [`Self::new_persistent_private`], Shared: a persistent buffer the
    /// CPU writes into place (the streamed tier's slot pools under the
    /// direct read, `DESIGN.md` §20.4a).
    pub fn new_persistent_shared(&self, size: usize) -> Result<Arc<PooledBuffer>> {
        let buffer = self
            .device
            .new_buffer(size.max(1), RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&buffer);
        // A Shared buffer's pages are the process's, and under memory
        // pressure the kernel compresses them whatever the residency set
        // says (measured: 28 GB of Shared slot pools un-wired and 7.5 GB
        // of them compressed mid-prefill, #749 P27); a Private buffer's are
        // not. Pin them.
        wire_pages(buffer.contents(), size);
        Ok(Arc::new(PooledBuffer::unpooled_resident(
            buffer,
            size,
            Arc::clone(&self.residency_set),
        )))
    }

    pub fn new_persistent_private(&self, size: usize) -> Result<Arc<PooledBuffer>> {
        let buffer = self
            .device
            .new_buffer(size.max(1), PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&buffer);
        Ok(Arc::new(PooledBuffer::unpooled_resident(
            buffer,
            size,
            Arc::clone(&self.residency_set),
        )))
    }

    /// Pin a Shared buffer's pages (`mlock`) so the kernel cannot compress
    /// them under memory pressure. A no-op for a buffer the CPU cannot see
    /// (Private storage: its pages are not the process's to compress).
    pub fn wire_buffer(&self, buffer: &PooledBuffer) {
        // `contents()` is null for a Private buffer (`Buffer::new`).
        wire_pages(buffer.contents(), buffer.size());
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<PooledBuffer>> {
        let unpooled_uploads = Self::unpooled;
        let size = core::mem::size_of_val(data);
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
            let buffer = self.allocate_buffer(size)?;
            // SAFETY: the pool handed out a buffer at least `size` bytes long
            // whose last GPU reader has completed, and `data` is `size` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), buffer.contents(), size);
            }
            crate::kernels::metal::upload_shadow::record(buffer.contents() as usize, unsafe {
                std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size)
            });
            return Ok(buffer);
        }
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        self.residency_set.insert(&new_buffer);

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
        Ok(Arc::new(PooledBuffer::unpooled_resident(
            new_buffer,
            size,
            Arc::clone(&self.residency_set),
        )))
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<PooledBuffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        if self.base_mode() {
            // A compute fill, inside the barrier chain (see `base_mode`).
            if size_in_bytes > 0 {
                let encoder = self.command_encoder()?;
                crate::kernels::kernels::call_const_fill(
                    &self.device,
                    &encoder,
                    &self.kernels,
                    "fill_u8",
                    size_in_bytes,
                    &buffer,
                    0u8,
                )
                .map_err(crate::Error::wrap)?;
            }
            return Ok(buffer);
        }
        let mut blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        Ok(buffer)
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<PooledBuffer>> {
        if let Some(b) = self.buffers.acquire(size) {
            return Ok(b);
        }
        // Kept before `buf_size` shadows `size`, so the overhang has both terms
        // (§9.2, #47).
        let requested = size;
        let size = buf_size(size);
        // §9.5k's branch, on the pool miss and before the allocation -- see
        // `new_buffer` above. This is the pool the KV reserve is served from
        // (`KvSlot::append` -> `Tensor::zeros` -> here), which is why the
        // subtracted `planned` figure includes KV.
        self.buffers
            .check_residual(size)
            .map_err(|e| MetalError::Message(e.to_string()))?;
        let new_buffer = if transient_heaps_on() {
            // From a heap chunk: resident with the chunk (§6.3k).
            self.transient_heaps
                .0
                .alloc(size)
                .map_err(MetalError::from)?
        } else {
            let b = self
                .device
                .new_buffer(size, RESOURCE_OPTIONS)
                .map_err(MetalError::from)?;
            // `add`, not `insert`: the deferred-commit batching the private pool's
            // miss already uses (§6.3b). Stage the allocation (one `addAllocation`)
            // and let the next command-buffer cut commit the residency set once via
            // `commit_pending`, rather than one `commit` per missed buffer. This is
            // the Shared pool (`self.buffers`) that serves the KV reserve
            // (`KvSlot::append` -> `Tensor::zeros` -> here); the earlier
            // residency-batching fix converted only the private pool and left this
            // path committing per miss, so a cold-pool prefill still paid a residency
            // commit for every Shared miss (§6.3b prices each commit at ~0.087 ms and
            // a cold prefill misses on the order of a hundred buffers).
            // `commit_pending` (`Commands::commit_swap_locked`) publishes the staged
            // buffers before any dispatch that binds them executes (§3.7), so the
            // residency invariant holds and the result is byte-identical.
            self.residency_set.add(&b);
            b
        };
        Ok(self.buffers.adopt_requesting(new_buffer, requested, size))
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

/// Alignment applied to every buffer allocation.
///
/// This is the Apple Silicon cache line. Aligning to it means every buffer
/// starts on a line boundary, so no two buffers share a line and the memory
/// controller can coalesce accesses cleanly. It also covers the alignment any
/// dtype could need: the kernels use `float4`/`half4` (16 bytes) widely and
/// quantized block structures are wider still, all well under 128.
///
/// **This is the DEFAULT and no longer the only reachable value** — see
/// [`buffer_alignment`], which reads `LLOOM_586_BUFFER_ALIGNMENT` once. The
/// constant is kept as the name every comment and test refers to, and as what
/// the axis reports when nothing selects an arm.
const BUFFER_ALIGNMENT: usize = 128;

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
    crate::tuning::tuning().pool.buffer_alignment.max(MIN_BUFFER_ALIGNMENT)
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
fn buf_size(size: usize) -> usize {
    // Never return 0: Metal rejects zero-length buffers, and callers may ask
    // for an empty tensor.
    size.max(1).next_multiple_of(buffer_alignment())
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

type DataUpload<'a> = Box<dyn FnOnce(&MetalDevice) -> Result<Arc<PooledBuffer>> + 'a>;

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

    pub fn build(self) -> Result<Arc<PooledBuffer>> {
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
            let got = buf_size(size);
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
        assert_eq!(buf_size(0), buffer_alignment());
    }

    #[test]
    fn test_buf_size_avoids_power_of_two_waste() {
        // Shapes that sit just above a power of two are the ones that suffered:
        // a 292-token KV cache and an FFN intermediate both rounded up hard.
        // Both are multiples of 128, so they are exact at every arm at or below
        // the default -- which is the arithmetic §9.2l relies on when it says
        // padding is free at tensor granularity "by arithmetic rather than by
        // luck".
        assert_eq!(buf_size(299008), 299008); // was 524288, 75% waste
        assert_eq!(buf_size(21504), 21504); // was 32768, 52% waste
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element; a scalar tensor requests a
        // 2-byte buffer. It must not be rounded down.
        assert!(buf_size(2) >= 2);
    }

    #[test]
    fn the_default_arm_agrees_with_buf_size() {
        // The mirror above is only worth having if it is the same arithmetic.
        // Pinned at the arm the test process actually resolved, so this holds
        // whether or not the environment selects one.
        let alignment = buffer_alignment();
        for size in [1usize, 2, 127, 128, 129, 21504, 299008] {
            assert_eq!(
                buf_size(size),
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

/// `mlock` `size` bytes at `ptr` (a Shared buffer's own mapping); a failure
/// is reported, not fatal — the buffer works unwired, only slower under
/// pressure.
fn wire_pages(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr..ptr+size` is the buffer's own mapping, alive for as
    // long as the buffer; `munlock` is implicit at unmap.
    let rc = unsafe { libc::mlock(ptr as *const libc::c_void, size.max(1)) };
    if rc != 0 {
        let e = std::io::Error::last_os_error();
        eprintln!("mlock of a {size}-byte Shared buffer failed: {e}");
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
