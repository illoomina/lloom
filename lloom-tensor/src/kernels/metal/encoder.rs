use crate::kernels::metal::{
    arena,
    contention::{Censused, Side},
    executor::{DispatchAction, DispatchRecord, ExecutorSlot, Grid},
    late_bind, trace, Buffer, ComputePipeline, DeviceListing, Fence,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLComputeCommandEncoder,
    MTLResourceUsage, MTLSize,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
};

/// Shared cross-encoder output map: maps buffer pointer -> fence of the last encoder that wrote it.
/// Used by subsequent encoders to call waitForFence before reading those buffers.
///
/// Read by the encoding thread once per buffer per encoder session and written
/// at each encoder close, where the entries of sessions whose command buffers
/// have completed are also removed. No Metal completion thread takes this
/// lock (`Commands`' `FenceBook`), so a bind never waits on one descheduled
/// while holding it. A `parking_lot::RwLock` so that reads never serialise
/// against each other; wrapped in [`Censused`] so that any acquisition that
/// does wait is counted (`DESIGN.md` §6.4d).
pub type PrevCeOutputs = Arc<Censused<parking_lot::RwLock<HashMap<usize, Arc<Fence>>>>>;

fn size_tuple(size: MTLSize) -> (usize, usize, usize) {
    (size.width, size.height, size.depth)
}

/// How intra-encoder hazards are keyed (`DESIGN.md` §9.2e).
///
/// Metal barriers are resource-granular and always will be (§3.5) -- this
/// changes the *decision* to emit one, never the barrier itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HazardKey {
    /// The buffer pointer alone. What candle has always done, and correct:
    /// two values in one allocation are one resource, so ordering them is
    /// conservative rather than wrong.
    ///
    /// Its cost is that it cannot tell "the same bytes" from "the same
    /// allocation". Under the pool those coincide often enough not to matter;
    /// under an arena every activation shares one pointer, so every
    /// write-then-read pair inside it becomes a false dependency.
    #[default]
    Pointer,
    /// `(pointer, offset, length)`, with overlap decided by an interval test --
    /// §9.2e's route (a).
    ///
    /// Strictly more precise than [`HazardKey::Pointer`]: two bindings that
    /// overlap in bytes still collide, and two that do not no longer do. So it
    /// can only ever *remove* barriers, never add one, and it cannot introduce
    /// a missing dependency -- an edge it drops is one where the two bindings
    /// provably touch disjoint bytes.
    Range,
}

/// The hazard keying every new encoder session adopts.
///
/// Process-global for the same reason the dispatch trace's switch is: the
/// choice is made by a harness in `candle-core` and consumed in the encoder,
/// and threading it through would add a parameter to
/// `compute_command_encoder` and to everything that calls it. It is read once
/// per encoder session -- 14 times per decode token -- not per bind, so the
/// atomic is nowhere near a hot path.
///
/// `Pointer` is the default and is the behaviour that shipped, so an
/// unconfigured process is byte-for-byte what it was.
static HAZARD_KEY: AtomicBool = AtomicBool::new(false);

/// Whether `CANDLE_METAL_HAZARD_KEY` has been consulted yet.
static HAZARD_KEY_FROM_ENV: OnceLock<()> = OnceLock::new();

/// Whether the key was chosen EXPLICITLY — by the environment or by
/// [`set_hazard_key`] — as opposed to being the shipped default. An arena
/// install selects `Range` only when nothing explicit has, so a harness that
/// pins `pointer` to measure the wrong key on purpose keeps it.
static HAZARD_KEY_EXPLICIT: AtomicBool = AtomicBool::new(false);

/// Apply `CANDLE_METAL_HAZARD_KEY=range` if it is set, once per process.
///
/// An environment switch rather than a parameter, so that **any** harness gets
/// the A/B without being taught about it -- including the determinism probe,
/// which is the one that matters here. Route (a) *removes* ordering edges, and
/// under `HazardTrackingModeUntracked` a wrongly-removed edge is silent
/// corruption rather than an error (§3.5), so it has to be run against the
/// §15.1 #7 gate and not only against a barrier count. A probe that cannot
/// select the mode cannot gate it.
fn init_hazard_key_from_env() {
    HAZARD_KEY_FROM_ENV.get_or_init(|| {
        match crate::tuning::tuning().metal.hazard_key.as_deref() {
            Some("range") => {
                HAZARD_KEY.store(true, Ordering::Relaxed);
                HAZARD_KEY_EXPLICIT.store(true, Ordering::Relaxed);
            }
            Some("pointer") | Some("ptr") => HAZARD_KEY_EXPLICIT.store(true, Ordering::Relaxed),
            // The parser (`tuning.rs`) refused anything else, loudly.
            _ => {}
        }
    });
}

/// Select how new encoder sessions key hazards (`DESIGN.md` §9.2e).
///
/// Takes effect at the next encoder session rather than immediately: hazard
/// state is per session and the two keyings hold different state, so switching
/// mid-session would compare bindings recorded under one rule against a lookup
/// under the other.
pub fn set_hazard_key(key: HazardKey) {
    // Mark the environment as consulted, so an explicit call wins over it
    // rather than being overwritten at the next session.
    let _ = HAZARD_KEY_FROM_ENV.set(());
    HAZARD_KEY.store(key == HazardKey::Range, Ordering::Relaxed);
    HAZARD_KEY_EXPLICIT.store(true, Ordering::Relaxed);
}

/// Select `key` unless the environment or a [`set_hazard_key`] call already
/// chose one. Returns the key in force afterwards.
///
/// The arena's install calls this with [`HazardKey::Range`]: an arena puts
/// every activation behind one pointer, so under the `Pointer` key every
/// write-then-read pair inside it is a false dependency (`DESIGN.md` §9.2e —
/// 497 / 475 / 446 barriers a token against the pool's 436 / 36 / 0), and
/// route (a) is the key an arena is correct *and* cheap under. A key that
/// only makes sense with the arena is selected by the arena; the explicit
/// switches stay for the A/B that measures the wrong one on purpose.
pub fn set_hazard_key_unless_explicit(key: HazardKey) -> HazardKey {
    init_hazard_key_from_env();
    if !HAZARD_KEY_EXPLICIT.load(Ordering::Relaxed) {
        HAZARD_KEY.store(key == HazardKey::Range, Ordering::Relaxed);
    }
    hazard_key()
}

/// The keying new encoder sessions will use.
pub fn hazard_key() -> HazardKey {
    init_hazard_key_from_env();
    if HAZARD_KEY.load(Ordering::Relaxed) {
        HazardKey::Range
    } else {
        HazardKey::Pointer
    }
}

/// Which of the three orderable hazards a binding just detected.
///
/// **The direction is computed at both emission sites and was discarded into a
/// `bool`** (issue #185). `set_input_buffer` tests one set and can only be
/// read-after-write; `set_output_buffer` tests two and is write-after-write or
/// write-after-read. Read-after-read is absent by construction -- two reads
/// never conflict -- which is why 5.394 GB of weights, bound on every dispatch,
/// contributes **zero** of the barriers.
///
/// Retaining it costs nothing on the default path: it is produced inside the
/// branch that has already decided a barrier is needed, and it is consumed only
/// by [`trace`] and by the `hazard-audit` feature, both of which are off unless
/// asked for.
///
/// This is `DESIGN.md` §6.7 **L2** -- *if a mechanism records information it
/// does not consume, either consume it or stop recording it* -- applied to a
/// value that was computed and then not even recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HazardKind {
    /// Read-after-write: this dispatch reads bytes an earlier one wrote.
    Raw,
    /// Write-after-write: this dispatch overwrites bytes an earlier one wrote.
    Waw,
    /// Write-after-read: this dispatch overwrites bytes an earlier one read.
    War,
}

impl HazardKind {
    /// A stable lowercase tag, for traces and assertion messages.
    pub fn as_str(self) -> &'static str {
        match self {
            HazardKind::Raw => "raw",
            HazardKind::Waw => "waw",
            HazardKind::War => "war",
        }
    }

    fn bit(self) -> u8 {
        match self {
            HazardKind::Raw => 1,
            HazardKind::Waw => 2,
            HazardKind::War => 4,
        }
    }
}

/// Which directions a **read** binding conflicts in, given the state before it.
///
/// A read can only ever be RAW: read-after-read is not a hazard, which is why
/// `prev_inputs` is not consulted and why the weights -- 5.394 GB, read on every
/// dispatch and never written -- contribute **zero** barriers.
///
/// Split out of [`ComputeCommandEncoder::set_input_buffer`] so that a test can
/// exercise **this** function rather than a re-implementation of it. #8.1d
/// records what the alternative costs: a script that reproduced the intended
/// name instead of asking the compiler *"validated the intent instead of the
/// artifact"*, and every one of the 48 variants was absent from the metallib.
#[inline]
pub(crate) fn read_hazards(
    prev_outputs: &BoundSet,
    key: HazardKey,
    range: &BoundRange,
) -> HazardKinds {
    let mut kinds = HazardKinds::NONE;
    if prev_outputs.conflicts(key, range) {
        kinds.insert(HazardKind::Raw);
    }
    kinds
}

/// Which directions a **write** binding conflicts in, given the state before it.
///
/// **Both tests run; this must not become `||`.** The original site was
/// `if prev_outputs.conflicts(..) || prev_inputs.conflicts(..)`, which is
/// correct for a `bool` and lossy for an attribution: `||` stops at the first
/// true operand, so at any position where WAW fires WAR is never tested, and WAR
/// is under-reported exactly where the two coincide (issue #185).
///
/// `conflicts` takes `&self` and only reads, so evaluating both is
/// behaviour-identical for the barrier decision -- the extra lookup happens only
/// on positions already emitting a barrier, where the barrier dominates.
#[inline]
pub(crate) fn write_hazards(
    prev_outputs: &BoundSet,
    prev_inputs: &BoundSet,
    key: HazardKey,
    range: &BoundRange,
) -> HazardKinds {
    let mut kinds = HazardKinds::NONE;
    if prev_outputs.conflicts(key, range) {
        kinds.insert(HazardKind::Waw);
    }
    if prev_inputs.conflicts(key, range) {
        kinds.insert(HazardKind::War);
    }
    kinds
}

/// The set of hazard directions behind one pending barrier.
///
/// A three-bit set rather than a collection: a barrier is a latch that any
/// number of bindings may set before it fires, so the honest attribution is
/// *which directions contributed*, not *which one was last*. One `u8` in
/// `EncoderState` and three bit operations per detected hazard, so retaining the
/// direction is free where it is produced -- which is the property that lets
/// this be on by default rather than behind the audit feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HazardKinds(u8);

impl HazardKinds {
    /// The empty set.
    pub const NONE: HazardKinds = HazardKinds(0);

    /// Add `kind` to the set.
    #[inline]
    pub fn insert(&mut self, kind: HazardKind) {
        self.0 |= kind.bit();
    }

    /// Add everything in `other`.
    #[inline]
    pub fn merge(&mut self, other: HazardKinds) {
        self.0 |= other.0;
    }

    /// Whether `kind` contributed.
    #[inline]
    pub fn contains(self, kind: HazardKind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// Whether nothing has been recorded.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Empty the set and return what it held.
    #[inline]
    pub fn take(&mut self) -> HazardKinds {
        std::mem::replace(self, HazardKinds::NONE)
    }

    /// The directions present, in RAW, WAW, WAR order.
    pub fn iter(self) -> impl Iterator<Item = HazardKind> {
        [HazardKind::Raw, HazardKind::Waw, HazardKind::War]
            .into_iter()
            .filter(move |&k| self.contains(k))
    }
}

/// One buffer binding, as the hazard tracking sees it.
///
/// Carries the range even under [`HazardKey::Pointer`], where the extra fields
/// are simply not compared. Keeping one type means the two modes cannot drift
/// apart in what they record, only in how they compare it -- the same reason
/// §11.3d gives for building the packed block out of the classical argument
/// list rather than beside it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundRange {
    pub ptr: usize,
    pub offset: usize,
    pub len: usize,
}

impl BoundRange {
    /// Whether these two bindings can touch the same byte.
    ///
    /// A zero length is treated as covering the whole allocation rather than
    /// nothing: an unknown extent must fail toward ordering, since the cost of
    /// a spurious barrier is throughput and the cost of a missing one is silent
    /// corruption (§3.5).
    #[inline]
    fn overlaps(&self, other: &BoundRange) -> bool {
        if self.ptr != other.ptr {
            return false;
        }
        if self.len == 0 || other.len == 0 {
            return true;
        }
        self.offset < other.offset + other.len && other.offset < self.offset + self.len
    }
}

/// The set of bindings seen in one phase, comparable under either keying.
///
/// `Pointer` mode keeps a pointer set and answers in O(1). `Range` mode keeps
/// the ranges bucketed *by pointer*, so a lookup scans only the bindings that
/// share an allocation -- one bucket, not the encoder's whole history. Under
/// the pool that bucket holds one or two entries; under an arena it holds the
/// slots, which is 6.
#[derive(Default)]
pub struct BoundSet {
    ptrs: HashSet<usize>,
    ranges: HashMap<usize, Vec<BoundRange>>,
}

impl BoundSet {
    fn insert(&mut self, key: HazardKey, b: BoundRange) {
        match key {
            HazardKey::Pointer => {
                self.ptrs.insert(b.ptr);
            }
            HazardKey::Range => {
                let bucket = self.ranges.entry(b.ptr).or_default();
                if !bucket.contains(&b) {
                    bucket.push(b);
                }
            }
        }
    }

    fn conflicts(&self, key: HazardKey, b: &BoundRange) -> bool {
        match key {
            HazardKey::Pointer => self.ptrs.contains(&b.ptr),
            HazardKey::Range => self
                .ranges
                .get(&b.ptr)
                .is_some_and(|bucket| bucket.iter().any(|seen| seen.overlaps(b))),
        }
    }

    /// Fold `other`'s bindings into this set, leaving `other` empty and
    /// **keeping its capacity** for the next dispatch to refill.
    ///
    /// # Why this deduplicates, and why that is not an optimisation of
    /// convenience (lloom #727)
    ///
    /// This runs once per DISPATCH from `auto_barrier`'s no-barrier branch.
    /// `append` re-inserted every range without the containment check
    /// `insert` applies, so a bucket grew by every binding made in the
    /// encoder — and under an arena or pass-plan base every activation shares
    /// ONE pointer, so one bucket takes all of it and every later `insert`
    /// and `conflicts` scans the lot.
    ///
    /// **Measured as a NULL on the shipping config, and the reason is the cut
    /// period** (lloom #727): `CANDLE_METAL_COMPUTE_PER_BUFFER` defaults to
    /// 50, so an encoder holds ~50 dispatches, not a step's 338 — the bucket
    /// reaches a few hundred ranges, where the linear scan is cheap. An
    /// interleaved A/B on LFM2 showed 62.5/62.5/62.6 before and
    /// 62.5/62.3/62.5 after, streams byte-identical.
    ///
    /// It is kept because it is correct and unconditionally cheaper —
    /// duplicates cannot change what `conflicts` answers, so keeping them was
    /// pure cost — and because the growth it removes is UNBOUNDED in the cut
    /// period: an arm that widens the period (`pipelined_cut` is 512) or a
    /// model with more dispatches per encoder pays the old shape
    /// quadratically. The null is on this config, not on the mechanism.
    ///
    /// Deduplicating here is SEMANTICS-PRESERVING because `conflicts` asks
    /// only whether some retained range overlaps the probe: a duplicate can
    /// never change that answer, and `insert` already relies on exactly this
    /// property (it skips a range the bucket already holds). What changes is
    /// only how many copies of the same fact are kept.
    ///
    /// `drain()` rather than `mem::take` on the far side, and no reallocation
    /// on this one: the buckets keep their capacity across dispatches instead
    /// of being freed and regrown 338 times a step.
    fn absorb(&mut self, other: &mut BoundSet) {
        self.ptrs.extend(other.ptrs.drain());
        for (ptr, v) in other.ranges.iter_mut() {
            let bucket = self.ranges.entry(*ptr).or_default();
            for b in v.drain(..) {
                if !bucket.contains(&b) {
                    bucket.push(b);
                }
            }
        }
    }

    /// Empty this set, keeping its allocated buckets for reuse.
    ///
    /// The counterpart to [`Self::absorb`]'s capacity discipline: the
    /// per-dispatch rotation in `auto_barrier` used `mem::take`, which hands
    /// the allocation away and leaves a fresh empty map to be regrown on the
    /// next dispatch. Clearing keeps the buckets — the same pointers recur
    /// every dispatch, so their capacity is exactly what is wanted next.
    ///
    /// The entries are RETAINED empty rather than removed: a pointer bound on
    /// one dispatch is overwhelmingly bound on the next (the weights, the KV
    /// slab, the arena's base), so keeping the `Vec` is what makes
    /// [`Self::insert`] allocation-free in the steady state. The map's own
    /// size is bounded by the distinct buffers an encoder touches, which is
    /// small and does not grow with dispatches.
    fn clear(&mut self) {
        self.ptrs.clear();
        for v in self.ranges.values_mut() {
            v.clear();
        }
    }

    /// Bytes this set is holding, for the test that pins the steady state.
    #[cfg(test)]
    fn bucket_capacity(&self) -> usize {
        self.ranges.values().map(Vec::capacity).sum()
    }

    fn take(&mut self) -> BoundSet {
        BoundSet {
            ptrs: std::mem::take(&mut self.ptrs),
            ranges: std::mem::take(&mut self.ranges),
        }
    }
}

/// Barrier tracking state for one encoder session.
/// Owned by ComputeCommandEncoder via Arc<Mutex<>> so clones share state.
pub struct EncoderState {
    recording: Option<super::command_buffer::EncoderAdmission>,
    ended: bool,
    close_succeeded: bool,
    /// How hazards are keyed this session.
    pub hazard_key: HazardKey,
    /// Bindings written since last barrier (RAW/WAW detection).
    pub prev_outputs: BoundSet,
    pub next_outputs: BoundSet,
    /// Bindings read since last barrier (WAR detection).
    pub prev_inputs: BoundSet,
    pub next_inputs: BoundSet,
    pub needs_barrier: bool,
    /// Which hazard directions contributed to the currently pending barrier
    /// (issue #185).
    ///
    /// A **set rather than one value**, because `needs_barrier` is a latch: any
    /// number of bindings may set it before the next dispatch fires the barrier,
    /// and they need not agree on direction. A single field would report
    /// whichever binding happened to be last, which is an arbitrary choice
    /// dressed as an attribution.
    ///
    /// Drained with `needs_barrier`, so a *deferred* barrier (§11.3r's
    /// suppression arm) keeps its kinds exactly as it keeps `prev_*` -- the two
    /// must roll together or the eventual emission is attributed to only the
    /// directions seen since the suppression.
    pub pending_kinds: HazardKinds,
    /// All outputs seen this encoder session (registered in global map at end_encoding).
    ///
    /// **There is deliberately no `all_inputs` beside this** — removed by issue
    /// #579, and the asymmetry is the point rather than an oversight.
    ///
    /// `all_outputs` is *read*: `Commands::end_encoding` walks it to register
    /// this encoder's fence against every buffer it wrote, so a later encoder
    /// reading one of them can wait. `all_inputs` was written at every input
    /// binding and **read nowhere** — its doc comment claimed *"cross-encoder
    /// fence coordination"*, which is a job `wait_for_buffer` already does, per
    /// buffer, at bind time (§6.4's fix). A set of every input a session saw
    /// expresses nothing that wait has not already acted on.
    ///
    /// It cost **1123 inserts per decode token** (counted from a dispatch
    /// trace, not from the source) at **27.0 ns per binding — 0.0303 ms/token,
    /// 3.82 % of §6.6b's non-GPU floor**, of which **66 % was the fresh
    /// `HashSet` allocation per encoder session** rather than the inserts. §6.7
    /// **L2** names this exact field as the canonical *"records information it
    /// does not consume"*, and §15.2 #11's dispositions are *consume it*,
    /// *delete it*, or *keep it as evidence* — the third is for probe results,
    /// not for a per-token write, and the first has no consumer to write.
    pub all_outputs: HashSet<usize>,
    /// Fences already waited on this session, so a buffer bound repeatedly does
    /// not re-emit the same wait.
    pub waited_fences: HashSet<usize>,
    /// Buffers whose last writer this session has already been ordered after,
    /// so a buffer bound repeatedly does not re-read the cross-encoder map.
    ///
    /// The second bind of a buffer in one session cannot find a different
    /// writer than the first: sessions on one `Commands` are sequential (the
    /// guard holds the outer lock for the whole dispatch), and the map changes
    /// only at an encoder close -- registering the closing session's outputs,
    /// removing completed ones -- so it does not change while a session is
    /// open. Counted on a real
    /// decode step before this set existed: 1101 map reads, 46 that could
    /// matter (lloom #727 §27; #732 measured the change).
    pub probed: HashSet<usize>,
    /// Cross-encoder map reads, hits, and `waitForFence` calls this session.
    /// Plain counters under the lock the bind already holds, summed into the
    /// profiler once at encoder close -- no per-bind atomic.
    pub fence_probes: u32,
    pub fence_hits: u32,
    pub fence_waits: u32,
    /// Name of the pipeline most recently set, so a dispatch can be attributed
    /// to a kernel. Only maintained when an executor is installed or profiling
    /// is on -- Metal has no way to read the bound pipeline back, and doing so
    /// per dispatch on the classical path would be cost for nobody.
    pub current_pipeline: Option<Arc<str>>,
    /// The largest input buffer bound since the last dispatch, in bytes. For a
    /// GEMV that is its weight matrix, which is how the counter profile prices a
    /// kernel in GB/s. Only maintained under `profile::counters_enabled()`.
    pub dispatch_input_max: usize,
}

impl Default for EncoderState {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderState {
    pub fn new() -> Self {
        Self::with_hazard_key(HazardKey::default())
    }

    pub fn with_hazard_key(hazard_key: HazardKey) -> Self {
        EncoderState {
            recording: None,
            ended: false,
            close_succeeded: false,
            hazard_key,
            prev_outputs: BoundSet::default(),
            next_outputs: BoundSet::default(),
            prev_inputs: BoundSet::default(),
            next_inputs: BoundSet::default(),
            needs_barrier: false,
            pending_kinds: HazardKinds::NONE,
            all_outputs: HashSet::new(),
            waited_fences: HashSet::new(),
            probed: HashSet::new(),
            fence_probes: 0,
            fence_hits: 0,
            fence_waits: 0,
            current_pipeline: None,
            dispatch_input_max: 0,
        }
    }
}

/// Shared allocation: the mutex remains the existing hazard-state lock. The
/// external count is touched only by wrapper construction/clone/drop, not binds.
pub(crate) struct SharedEncoderState {
    state: parking_lot::Mutex<EncoderState>,
    external: std::sync::atomic::AtomicUsize,
    domain: std::sync::OnceLock<Arc<EncoderDomain>>,
    wake: std::sync::OnceLock<Arc<crate::completion_wake::Wake>>,
}
impl std::ops::Deref for SharedEncoderState {
    type Target = parking_lot::Mutex<EncoderState>;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

/// All escaped encoders created by one Commands, including older recordings.
/// Only atomics and Wake are owned here: no Core, native resource or user capture
/// can be finally destroyed by a foreign permit releasing this metadata.
pub(super) struct EncoderDomain {
    external: std::sync::atomic::AtomicUsize,
    wake: Arc<crate::completion_wake::Wake>,
}
impl EncoderDomain {
    pub(super) fn new(wake: Arc<crate::completion_wake::Wake>) -> Arc<Self> {
        Arc::new(Self {
            external: std::sync::atomic::AtomicUsize::new(0),
            wake,
        })
    }
    pub(super) fn has_external_handles(&self) -> bool {
        self.external.load(Ordering::Acquire) != 0
    }
}
struct DomainEncoderHandle(Arc<EncoderDomain>);
impl DomainEncoderHandle {
    fn acquire(domain: &Arc<EncoderDomain>) -> Self {
        let domain = Arc::clone(domain);
        // Each counted permit owns this Arc before increment and through release.
        domain.external.fetch_add(1, Ordering::Relaxed);
        Self(domain)
    }
}
impl Drop for DomainEncoderHandle {
    fn drop(&mut self) {
        if self.0.external.fetch_sub(1, Ordering::Release) == 1 {
            self.0.wake.notify();
        }
    }
}
/// Declared LAST in the wrapper: release follows every other field destructor.
/// This retains shared encoder state, including its recording parent until
/// final capture extraction takes it. Count zero proves external operation
/// exclusion, not final metadata/physical destruction. It also rolls back a
/// partly constructed Clone after earlier cloned fields unwind.
struct ExternalEncoderHandle {
    state: Arc<SharedEncoderState>,
    // LAST: shared-state custody is released before aggregate capability.
    _domain: Option<DomainEncoderHandle>,
}
impl ExternalEncoderHandle {
    fn acquire(state: &Arc<SharedEncoderState>) -> Self {
        // The state Arc is retained before counting, and released after decrement.
        let state = Arc::clone(state);
        let domain = state.domain.get().map(DomainEncoderHandle::acquire);
        state.external.fetch_add(1, Ordering::Relaxed);
        Self {
            state,
            _domain: domain,
        }
    }
}
impl Drop for ExternalEncoderHandle {
    fn drop(&mut self) {
        if self.state.external.fetch_sub(1, Ordering::Release) == 1 {
            if let Some(wake) = self.state.wake.get() {
                wake.notify();
            }
        }
    }
}

pub struct ComputeCommandEncoder {
    finalizer: Option<Arc<super::commands::EncoderFinalizer>>,
    pub(crate) raw: super::raw_encoder::Compute,
    /// Retained so we can register completion handlers on this CB.
    pub(crate) command_buffer: super::CommandBuffer,
    /// Per-encoder-session fence. Updated at end_encoding.
    pub(crate) fence: Arc<Fence>,
    /// Hazard tracking state. CommandsGuard borrows the canonical EntryState
    /// wrapper under the outer mutex. Explicit public clones share this allocation
    /// and carry independently counted external handles.
    pub(crate) state: Arc<SharedEncoderState>,
    /// Buffer -> fence of its last writer, so a bind can wait on just that
    /// buffer instead of on every live fence.
    pub(crate) prev_ce_outputs: PrevCeOutputs,
    /// How dispatches reach the GPU (`DESIGN.md` §11.1).
    ///
    /// `ExecutorSlot::Classical` is the default and forwards to `self.raw`
    /// exactly as before this field existed, so the default path is unchanged
    /// rather than merely equivalent.
    pub(crate) executor: Arc<ExecutorSlot>,
    /// Open packed-params block, when a caller is capturing scalars instead of
    /// binding them inline (`DESIGN.md` §11.3b, issue #38).
    ///
    /// `AtomicBool` beside the buffer, rather than testing an `Option` under
    /// the lock, so the classical path pays a relaxed load and nothing else.
    /// Following #35's shape deliberately: that change kept "the classical path
    /// must not regress" *structural* by branching before doing any work, and a
    /// mutex per scalar bind would have given that up -- `DESIGN.md` §6.4a
    /// measured per-bind bookkeeping at 29.1 ns and the whole fence probe at
    /// 5.1 % of non-GPU time, so per-bind additions are exactly the shape worth
    /// not paying by default.
    pub(crate) capturing: Arc<AtomicBool>,
    pub(crate) param_capture: Arc<parking_lot::Mutex<ParamCapture>>,
    /// The clock the pooled constants buffers decide rewriting against
    /// (`ConstantsPool`). `None` on encoders built outside `Commands`, which
    /// have no in-flight work to order a CPU write behind.
    pub(crate) clock: Option<Arc<crate::kernels::metal::GpuClock>>,
    /// The device's own dispatch census, when profiling is on
    /// (`profile::DeviceCensus`). `None` on encoders built outside `Commands`.
    pub(crate) census: Option<Arc<crate::kernels::metal::profile::DeviceCensus>>,
    /// The device's own dispatch trace, when the trace switch is on
    /// (`trace::DeviceTrace`). `None` on encoders built outside `Commands`.
    pub(crate) trace: Option<Arc<trace::DeviceTrace>>,
    // Keep LAST. None only for the canonical EntryState-owned wrapper.
    external: Option<ExternalEncoderHandle>,
}

impl Clone for ComputeCommandEncoder {
    fn clone(&self) -> Self {
        // A clone of even the internal canonical wrapper is an external escape.
        // Keep this rollback owner local until all other field clones succeed.
        let external = Some(ExternalEncoderHandle::acquire(&self.state));
        Self {
            finalizer: self.finalizer.clone(),
            raw: self.raw.clone(),
            command_buffer: self.command_buffer.clone(),
            fence: self.fence.clone(),
            state: self.state.clone(),
            prev_ce_outputs: self.prev_ce_outputs.clone(),
            executor: self.executor.clone(),
            capturing: self.capturing.clone(),
            param_capture: self.param_capture.clone(),
            clock: self.clock.clone(),
            census: self.census.clone(),
            trace: self.trace.clone(),
            external,
        }
    }
}

/// Scalars accumulated for a packed params block, and the buffer renumbering
/// that has to accompany them.
///
/// Diverting a scalar out of the argument list leaves a hole in the buffer
/// indices: `call_rms_norm` binds `(length, elements_to_sum, src, dst, alpha,
/// eps)` at 0..5, and the packed kernel takes `(params, src, dst, alpha)` at
/// 0..3. So capture cannot only collect bytes -- it must also renumber the
/// bindings that remain, or every buffer lands one or two slots too high and
/// the kernel reads whatever was left at that index. Under
/// `HazardTrackingModeUntracked` that is a silent wrong answer (`DESIGN.md`
/// §3.5), which is the same class of failure as a bad struct offset and is
/// caught by the same bit-identical test.
///
/// `next_buffer` starts at 1 because slot 0 is the params buffer itself.
#[derive(Default)]
pub struct ParamCapture {
    bytes: Vec<u8>,
    next_buffer: usize,
    /// Buffers allocated to hold arrays that the classical path binds with
    /// `setBytes`. Handed to the caller at capture close, so their lifetime is
    /// the dispatch's rather than the capture's.
    staged: Vec<Buffer>,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

/// Already-evaluated inline constant borrow. This is a byte representation,
/// not an arbitrary EncoderParam callback; preparation invokes no native effect.
/// Packed/ICB uses a separate genuine staged-block binding, not this inline mode.
pub(crate) struct InlineArgument<'a> {
    pub(crate) index: usize,
    pub(crate) bytes: &'a [u8],
}

// Cold-owned immutable shader access is separate from the unique operation
// writer. A Plan moves whole into original-generation slots before any effect.
enum InlinePipelineOwner<'a> {
    Borrowed(&'a ComputePipeline),
    Plan(crate::metal_backend::CatalogPlanLease),
}
enum InstalledInlinePipeline<'a> {
    Borrowed(&'a ComputePipeline),
    Plan(super::native_generation_bindings::InstalledPlan),
}
impl InstalledInlinePipeline<'_> {
    fn pipeline(&self) -> &ComputePipeline {
        match self {
            Self::Borrowed(pipeline) => pipeline,
            Self::Plan(plan) => plan.affine_shader().native_pipeline(),
        }
    }
}
/// Original encoder Ticket, not current Commands lookup. Existing state mutex
/// is held only to borrow the Ticket; executor/native lowering runs offguard.
struct BindingEdit<'a> {
    buffer: &'a super::CommandBuffer,
    ticket: Option<crate::command_lifecycle::Ticket>,
    entered_effects: bool,
}
impl BindingEdit<'_> {
    fn finish(mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.buffer
            .finish_binding_edit(self.ticket.as_mut().expect("binding edit"))?;
        self.ticket = None;
        Ok(())
    }
}
impl Drop for BindingEdit<'_> {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.as_mut() else {
            return;
        };
        let result = if self.entered_effects || std::thread::panicking() {
            self.buffer.abandon_binding_edit(ticket)
        } else {
            self.buffer.finish_binding_edit(ticket)
        };
        if result.is_err() || self.entered_effects || std::thread::panicking() {
            if let Ok(generation) = self.buffer.binding_generation() {
                // Quarantine is not terminality/Drop. Root polling later wakes
                // SAME queue waiters outside registry and ready/active locks.
                generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            }
        }
    }
}
impl ComputeCommandEncoder {
    fn binding_edit(&self) -> Result<BindingEdit<'_>, crate::kernels::MetalKernelError> {
        let state = self.state.lock(); // existing mutex only
        if state.ended {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "encoder already ended",
            ));
        }
        let owner =
            state
                .recording
                .as_ref()
                .ok_or(crate::kernels::MetalKernelError::QueueUnavailable(
                    "raw encoder requires unsafe interoperability",
                ))?;
        if !owner.same_buffer(&self.command_buffer) {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "foreign encoder admission",
            ));
        }
        let ticket = owner.begin_binding_edit()?;
        drop(state);
        Ok(BindingEdit {
            buffer: &self.command_buffer,
            ticket: Some(ticket),
            entered_effects: false,
        })
    }
    /// All tuple arguments and counted mints are preflighted before this private
    /// lowering entry. Installed capability roots exact generation+NativeUse;
    /// untracked legacy/foreign/replaced generations never take this safe route.
    pub(crate) fn set_buffer_tracked(
        &self,
        index: usize,
        binding: &super::native_generation_bindings::PreparedBuffer,
        offset: usize,
        write: bool,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        self.set_buffer_batch_tracked(&[(index, binding, offset, write)])
    }

    /// Buffer-only batch primitive, preserving one exact operation permit.
    /// Caller evaluates argument expressions ONCE and premints EVERY lease
    /// before this call. Full heterogeneous scalar/packed/ICB lowering remains a
    /// caller migration obligation; this does not claim all set_params coverage.
    pub(crate) fn set_buffer_batch_tracked<const N: usize>(
        &self,
        bindings: &[(
            usize,
            &super::native_generation_bindings::PreparedBuffer,
            usize,
            bool,
        ); N],
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let mut edit = self.binding_edit()?;
        let generation = self.command_buffer.binding_generation()?;
        let mut prepared: [Option<super::native_generation_bindings::PreparedBuffer>; N] =
            std::array::from_fn(|_| None);
        for (slot, &(_, binding, _, _)) in prepared.iter_mut().zip(bindings) {
            *slot = Some(binding.try_clone()?);
        }
        // Keep the actual packed cursor under its EXISTING mutex across whole
        // preflight/once-only commit, but release before any custom/native lower.
        let mut capture = self
            .capturing
            .load(Ordering::Relaxed)
            .then(|| self.param_capture.lock());
        let mut next_buffer = capture.as_ref().map(|capture| capture.next_buffer);
        let mut preview_effects = false;
        let plan = crate::native_binding_preflight::preflight_buffer_indices(
            next_buffer.as_mut(),
            bindings,
            |entry| entry.0,
            |_, &(_, binding, offset, _)| {
                if !binding.matches(generation) {
                    return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                        "stale native binding generation",
                    ));
                }
                let buffer = binding.lease().buffer_for_binding().map_err(|_| {
                    crate::kernels::MetalKernelError::QueueUnavailable(
                        "native binding is not counted External custody",
                    )
                })?;
                crate::native_binding_preflight::checked_bind_offset(
                    buffer.base_offset(),
                    buffer.length(),
                    offset,
                )
                .ok_or(crate::kernels::MetalKernelError::QueueUnavailable(
                    "native binding view offset refused",
                ))?;
                Ok(())
            },
            crate::kernels::MetalKernelError::QueueUnavailable,
            &mut preview_effects,
        )?;
        // All ordinary checks/clones complete. Existing exact EDIT is the
        // sole slot-writer permit; capacity refusal commits nothing above/below.
        let installed = unsafe {
            generation
                .install_prepared(prepared.map(|binding| binding.expect("prepared tuple member")))
        }?;
        edit.entered_effects |= preview_effects;
        if let (Some(capture), Some(next)) = (capture.as_mut(), next_buffer) {
            capture.next_buffer = next;
        }
        drop(capture);
        for (position, &(index, _, offset, write)) in bindings.iter().enumerate() {
            let binding = &installed[position];
            let index = plan.resolve(position, index);
            let lease = binding.lease();
            // Already checked immutable External role; borrow spans same head.
            let buffer = lease.buffer_for_binding().map_err(|_| {
                crate::kernels::MetalKernelError::QueueUnavailable(
                    "installed native binding changed role",
                )
            })?;
            // SAFETY: installed counted lease and exact generation head survive
            // native terminality and cached references; raw does not escape.
            let raw = unsafe { lease.raw_unchecked() };
            if write {
                self.set_output_buffer_native_resolved(index, Some(buffer), offset, Some(raw));
            } else {
                self.set_input_buffer_native_resolved(index, Some(buffer), offset, Some(raw));
            }
        }
        edit.finish()
    }

    /// Genuine mint-only output custody, before pipeline/scalar/dispatch effects.
    /// Capacity is caller-preadmitted/refilled; no legacy lease coercion occurs.
    pub(crate) fn prepare_fill_output(
        &self,
        buffer: &Buffer,
    ) -> Result<super::native_generation_bindings::PreparedBuffer, crate::kernels::MetalKernelError>
    {
        self.command_buffer
            .binding_generation()?
            .prepare_buffer(buffer)
    }

    /// Preserved prior captured-fill intermediate; not selected by inline adapter. Exact original encoder Ticket
    /// spans preflight, packed bytes, pipeline, output, scalars AND dispatch.
    /// # Safety
    /// Caller binds the exact fill shader to this primitive's width/alignment,
    /// validates output extent and dispatch geometry, and owns every captured
    /// array/packed block through terminal native use AND escaped emission.
    /// A packed block may be CPU-rewritten only under its genuine exclusive
    /// physical occupancy; clock identity or a cursor reset is not that proof.
    /// All arguments are evaluated once before any encoder effects.
    pub(crate) unsafe fn dispatch_fill_captured_intermediate(
        &self,
        pipeline: &ComputePipeline,
        output: &Buffer,
        value: &[u8],
        value_align: usize,
        length: usize,
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        let mut edit = self.binding_edit()?;
        let generation = self.command_buffer.binding_generation()?;
        let buffer = output;
        if value.is_empty() || value.len() > 8 || !value_align.is_power_of_two() || value_align > 8
        {
            return Err(Error::QueueUnavailable("invalid sealed fill scalar layout"));
        }
        let extent = length
            .checked_mul(value.len())
            .ok_or(Error::QueueUnavailable("fill byte extent overflow"))?;
        if extent > buffer.length() {
            return Err(Error::QueueUnavailable("fill exceeds output view"));
        }
        let length_bytes = length.to_ne_bytes();
        let mut capture = self
            .capturing
            .load(Ordering::Relaxed)
            .then(|| self.param_capture.lock());
        // Exact existing packed order: Output(0), scalar(1), usize length(2).
        // Preflight all byte geometry/capacity BEFORE effect latch/cursor commit.
        let scalar_layout = if let Some(cap) = capture.as_ref() {
            let value_start = cap
                .bytes
                .len()
                .checked_add(value_align - 1)
                .map(|end| end & !(value_align - 1))
                .ok_or(Error::QueueUnavailable(
                    "fill packed scalar offset overflow",
                ))?;
            let value_end = value_start
                .checked_add(value.len())
                .ok_or(Error::QueueUnavailable("fill packed scalar end overflow"))?;
            let align = core::mem::align_of::<usize>();
            let length_start = value_end
                .checked_add(align - 1)
                .map(|end| end & !(align - 1))
                .ok_or(Error::QueueUnavailable(
                    "fill packed length offset overflow",
                ))?;
            let end = length_start
                .checked_add(length_bytes.len())
                .ok_or(Error::QueueUnavailable("fill packed length end overflow"))?;
            if end > cap.bytes.capacity() {
                return Err(Error::QueueUnavailable(
                    "fill packed storage needs cold refill",
                ));
            }
            Some((value_start, value_end, length_start, end))
        } else {
            None
        };
        // Preview against local cursor/effects only. No shared capture mutation,
        // pin reservation or native effect until every ordinary preflight succeeds.
        let mut next_buffer = capture.as_ref().map(|cap| cap.next_buffer);
        let mut preview_effects = false;
        let entries = [0usize];
        let plan = crate::native_binding_preflight::preflight_buffer_indices(
            next_buffer.as_mut(),
            &entries,
            |entry| *entry,
            |_, _| {
                crate::native_binding_preflight::checked_bind_offset(
                    buffer.base_offset(),
                    buffer.length(),
                    0,
                )
                .map(|_| ())
                .ok_or(Error::QueueUnavailable(
                    "fill output absolute offset overflow",
                ))
            },
            Error::QueueUnavailable,
            &mut preview_effects,
        )?;
        // Exact original EDIT admission excludes native close/submit. Counted
        // mint/refusal runs before consuming slot, through shared actual helper.
        let prepared = generation.prepare_buffer(output)?;
        // SAFETY: original EDIT is held; complete scalar/geometry/capture
        // preflight above. No ordinary failure can consume a partial tuple.
        let [installed] = unsafe { generation.install_prepared([prepared]) }?;
        let buffer = installed.lease().buffer_for_binding().map_err(|_| {
            generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            Error::QueueUnavailable("installed fill lease violated External role")
        })?;
        edit.entered_effects |= preview_effects;
        if let (Some(cap), Some(next)) = (capture.as_mut(), next_buffer) {
            cap.next_buffer = next;
        }
        if let (Some(cap), Some((value_start, value_end, length_start, end))) =
            (capture.as_mut(), scalar_layout)
        {
            // All bounds/capacity checked. u8 resize fits retained allocation;
            // no fallible clone/allocation/user callback between arm and commit.
            cap.bytes.resize(end, 0);
            cap.bytes[value_start..value_end].copy_from_slice(value);
            cap.bytes[length_start..end].copy_from_slice(&length_bytes);
        }
        drop(capture);
        self.set_compute_pipeline_state(pipeline);
        // SAFETY: genuine output's same-generation head remains throughout this
        // call and independently through raw terminality/cached-reference closure.
        let raw = unsafe { installed.lease().raw_unchecked() };
        self.set_output_buffer_native_resolved(plan.resolve(0, 0), Some(buffer), 0, Some(raw));
        if scalar_layout.is_none() {
            self.set_bytes_directly(1, value.len(), value.as_ptr().cast());
            self.set_bytes(2, &length);
        }
        self.dispatch_thread_groups(groups, threads);
        edit.finish()
    }

    /// Shared complete inline operation: one original EDIT covers every buffer
    /// mint, whole credit plan/install, scalar, pipeline and dispatch. This is
    /// an intermediate caller seam, not legacy or Packed/ICB activation.
    /// # Safety
    /// Caller validates this pipeline's exact shader schema, scalar widths and
    /// accessed buffer extents/grid. All arguments are evaluated once before
    /// this call. Inline mode has no active old parameter capture; Packed/ICB
    /// callers must supply genuine plan-owned staging/rewrite custody separately.
    pub(crate) unsafe fn dispatch_inline_arguments_tracked<const N: usize>(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[(usize, &Buffer, usize, bool)],
        scalars: &[InlineArgument<'_>],
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        // SAFETY: unchanged exact schema/geometry caller contract above.
        unsafe {
            self.dispatch_inline_arguments_impl::<N>(
                InlinePipelineOwner::Borrowed(pipeline),
                buffers,
                scalars,
                groups,
                threads,
            )
        }
    }

    /// Complete actual affine tuple with terminal-owned SAME-node writer/root.
    /// No method-return unlock, pipeline retain or operation Arc allocation.
    /// # Safety
    /// Sealed payload preparation validates shader/native integer geometry and
    /// originating input/output/staging extents. ALL parameter writes ended
    /// before into_plan. All buffer arguments were evaluated once. Escaped
    /// native/executor/ICB emissions retain this SAME generation's page head.
    pub(crate) unsafe fn dispatch_affine_arguments_tracked<const N: usize>(
        &self,
        plan: crate::metal_backend::CatalogPlanLease,
        buffers: &[(usize, &Buffer, usize, bool)],
        scalars: &[InlineArgument<'_>],
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        // SAFETY: no mutable plan projection survives consumption above; the
        // common actual EDIT/mint/preflight body publishes N+1 before effects.
        unsafe {
            self.dispatch_inline_arguments_impl::<N>(
                InlinePipelineOwner::Plan(plan),
                buffers,
                scalars,
                groups,
                threads,
            )
        }
    }

    unsafe fn dispatch_inline_arguments_impl<const N: usize>(
        &self,
        pipeline: InlinePipelineOwner<'_>,
        buffers: &[(usize, &Buffer, usize, bool)],
        scalars: &[InlineArgument<'_>],
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        let mut edit = self.binding_edit()?;
        let generation = self.command_buffer.binding_generation()?;
        if buffers.len() != N {
            return Err(Error::QueueUnavailable(
                "inline operation tuple arity mismatch",
            ));
        }
        // Existing Packed/ICB route remains callable; this new inline route is
        // not used to erase it or claim its independent staging ownership closed.
        if self.capturing.load(Ordering::Acquire) {
            return Err(Error::QueueUnavailable(
                "inline operation requires prepared noncapture mode",
            ));
        }
        // A complete shader argument tuple has one binding per native index.
        // Reject scalar/buffer and same-kind collisions before any mint, credit
        // reservation or effect; otherwise setBytes could replace a live buffer.
        let mut occupied = 0u32;
        for &(index, _, _, _) in buffers {
            if index >= 31 {
                return Err(Error::QueueUnavailable(
                    "inline buffer native index refused",
                ));
            }
            let bit = 1u32 << index;
            if occupied & bit != 0 {
                return Err(Error::QueueUnavailable("inline argument index collision"));
            }
            occupied |= bit;
        }
        for scalar in scalars {
            // Apple setBytes contract: data smaller than 4KiB. The prepared
            // caller stages larger arrays as ordinary counted buffer members.
            if scalar.bytes.len() >= 4096 {
                return Err(Error::QueueUnavailable(
                    "inline argument needs genuine staged buffer",
                ));
            }
            if scalar.index >= 31 {
                return Err(Error::QueueUnavailable(
                    "inline scalar native index refused",
                ));
            }
            let bit = 1u32 << scalar.index;
            if occupied & bit != 0 {
                return Err(Error::QueueUnavailable("inline argument index collision"));
            }
            occupied |= bit;
        }
        let mut prepared: [Option<super::native_generation_bindings::PreparedBuffer>; N] =
            std::array::from_fn(|_| None);
        for (slot, &(_, buffer, _, _)) in prepared.iter_mut().zip(buffers) {
            *slot = Some(generation.prepare_buffer(buffer)?);
        }
        let mut preview_effects = false;
        let plan = crate::native_binding_preflight::preflight_buffer_indices(
            None,
            buffers,
            |entry| entry.0,
            |_, &(_, buffer, offset, _)| {
                crate::native_binding_preflight::checked_bind_offset(
                    buffer.base_offset(),
                    buffer.length(),
                    offset,
                )
                .map(|_| ())
                .ok_or(Error::QueueUnavailable("inline buffer view offset refused"))
            },
            Error::QueueUnavailable,
            &mut preview_effects,
        )?;
        // SAFETY: SAME original EDIT spans this entire call and is the sole
        // slot writer. Every ordinary schema/geometry/mint check precedes this.
        let prepared = prepared.map(|binding| binding.expect("prepared inline member"));
        let (installed, pipeline) = match pipeline {
            InlinePipelineOwner::Borrowed(pipeline) => (
                unsafe { generation.install_prepared(prepared) }?,
                InstalledInlinePipeline::Borrowed(pipeline),
            ),
            InlinePipelineOwner::Plan(plan) => {
                let (buffers, plan) =
                    unsafe { generation.install_prepared_with_plan(prepared, plan) }?;
                (buffers, InstalledInlinePipeline::Plan(plan))
            }
        };
        // No effect precedes every counted pin and optional plan LIVE. Any
        // unwind/refused finish quarantines original generation; writer/root,
        // native leases and initialized slots remain in SAME terminal custody.
        edit.entered_effects = true;
        self.set_compute_pipeline_state(pipeline.pipeline());
        for (position, &(index, _, offset, write)) in buffers.iter().enumerate() {
            let lease = installed[position].lease();
            let buffer = lease.buffer_for_binding().map_err(|_| {
                generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
                Error::QueueUnavailable("installed inline lease violated External role")
            })?;
            let raw = unsafe { lease.raw_unchecked() };
            let index = plan.resolve(position, index);
            if write {
                self.set_output_buffer_native_resolved(index, Some(buffer), offset, Some(raw));
            } else {
                self.set_input_buffer_native_resolved(index, Some(buffer), offset, Some(raw));
            }
        }
        for scalar in scalars {
            self.set_bytes_directly(
                scalar.index,
                scalar.bytes.len(),
                scalar.bytes.as_ptr().cast(),
            );
        }
        self.dispatch_thread_groups(groups, threads);
        edit.finish()
    }

    /// Sealed primitive fill is the first actual common-bundle caller. Dynamic
    /// shader/type pairs never enter through this method's sealed outer adapter.
    pub(crate) fn dispatch_fill_tracked(
        &self,
        pipeline: &ComputePipeline,
        output: &Buffer,
        value: &[u8],
        value_align: usize,
        length: usize,
        groups: MTLSize,
        threads: MTLSize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        if value.is_empty() || value.len() > 8 || !value_align.is_power_of_two() || value_align > 8
        {
            return Err(Error::QueueUnavailable("invalid sealed fill scalar layout"));
        }
        let extent = length
            .checked_mul(value.len())
            .ok_or(Error::QueueUnavailable("fill byte extent overflow"))?;
        if extent > output.length() {
            return Err(Error::QueueUnavailable("fill exceeds output view"));
        }
        let length_bytes = length.to_ne_bytes();
        // SAFETY: sole caller is the sealed TrackedFillValue adapter: matching
        // fill shader/primitive width, output extent checked above; native
        // groups/threads come from its checked nonzero pipeline width.
        unsafe {
            self.dispatch_inline_arguments_tracked::<1>(
                pipeline,
                &[(0, output, 0, true)],
                &[
                    InlineArgument {
                        index: 1,
                        bytes: value,
                    },
                    InlineArgument {
                        index: 2,
                        bytes: &length_bytes,
                    },
                ],
                groups,
                threads,
            )
        }
    }

    /// Import an already active native compute encoder and its exact parent.
    ///
    /// # Safety
    ///
    /// `raw` must be the unique active encoder of `command_buffer`, from Apple
    /// Metal or a separately audited equivalent native implementation. The
    /// caller must satisfy [`super::CommandBuffer::new`]'s import obligations.
    /// Do not independently import or externally encode/end this same encoder
    /// while this wrapper owns it. Keep its parent and bound resources alive
    /// through actual native use, including unretained resources through GPU use.
    /// End encoding successfully before external parent submission; coordinate
    /// that submission without concurrent wrapper/native access. All operations
    /// must obey documented native thread/phase rules, including callbacks and
    /// final release. Importing does not establish engine-tracked ownership.
    pub unsafe fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> ComputeCommandEncoder {
        // SAFETY: the caller supplies the same exact-parent, native lifetime
        // and exclusive raw-import obligations; only the executor is selected.
        unsafe {
            Self::with_executor(
                raw,
                command_buffer,
                fence,
                prev_ce_outputs,
                Arc::new(ExecutorSlot::Classical),
            )
        }
    }

    /// As [`Self::new`], but submitting dispatches through `executor`.
    ///
    /// # Safety
    ///
    /// All native-origin, matching-parent, exclusivity, resource-lifetime and
    /// thread/phase obligations of [`Self::new`] apply. The executor does not
    /// establish native lifetime, tracked ownership or thread safety.
    pub unsafe fn with_executor(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
        executor: Arc<ExecutorSlot>,
    ) -> ComputeCommandEncoder {
        Self::with_backend(
            super::raw_encoder::Compute::Metal3(raw),
            // SAFETY: the raw importer promises this is the encoder's exact
            // parent, with one authoritative family and native lifetimes held.
            unsafe { super::CommandBuffer::new(command_buffer) },
            fence,
            prev_ce_outputs,
            executor,
        )
    }

    pub(crate) fn with_backend(
        raw: super::raw_encoder::Compute,
        command_buffer: super::CommandBuffer,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
        executor: Arc<ExecutorSlot>,
    ) -> ComputeCommandEncoder {
        let state = Arc::new(SharedEncoderState {
            state: parking_lot::Mutex::new(EncoderState::with_hazard_key(hazard_key())),
            external: std::sync::atomic::AtomicUsize::new(0),
            domain: std::sync::OnceLock::new(),
            wake: std::sync::OnceLock::new(),
        });
        let external = Some(ExternalEncoderHandle::acquire(&state));
        crate::kernels::metal::hazard_audit::record_pass_open(
            raw.audit_id(),
            command_buffer.identity(),
        );
        ComputeCommandEncoder {
            finalizer: None,
            raw,
            command_buffer,
            fence,
            state,
            prev_ce_outputs,
            executor,
            capturing: Arc::new(AtomicBool::new(false)),
            param_capture: Arc::new(parking_lot::Mutex::new(ParamCapture::default())),
            clock: None,
            census: None,
            trace: None,
            external,
        }
    }

    /// Commands alone converts a freshly constructed, never-published wrapper
    /// into its canonical EntryState owner under EntryState. Borrowed guards do
    /// not increment the count; cloning through one always does.
    pub(super) fn into_canonical(mut self, domain: Arc<EncoderDomain>) -> Self {
        let _ = self.state.wake.set(domain.wake.clone());
        let _ = self.state.domain.set(domain);
        self.external.take();
        self
    }

    pub(super) fn has_external_handles(&self) -> bool {
        self.state.external.load(Ordering::Acquire) != 0
    }

    /// Final Core extraction only; no native access is exposed by this check.
    pub(super) fn retirement_closed(&self) -> bool {
        self.external.is_none()
            && !self.has_external_handles()
            && self
                .state
                .try_lock()
                .is_some_and(|state| state.close_succeeded)
    }

    pub(super) fn with_finalizer(
        mut self,
        finalizer: Arc<super::commands::EncoderFinalizer>,
    ) -> Self {
        self.finalizer = Some(finalizer);
        self
    }

    pub(super) fn with_recording(self, admission: super::command_buffer::EncoderAdmission) -> Self {
        self.state.lock().recording = Some(admission);
        self
    }

    pub(super) fn end_encoding_with(
        &self,
        close: impl FnOnce(
            super::command_buffer::EndingEncoder<'_>,
            &mut EncoderState,
        ) -> Result<(), crate::kernels::MetalKernelError>,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        self.end_encoding_prepared(false, || Ok(()), |scope, state, ()| close(scope, state))
    }

    /// Acquire all fallible/nonblocking preparation before ENDING and any field
    /// mutation. A pending poll does not notify itself; actual phase release or
    /// exceptional native exit wakes only after state/preparation guards unwind.
    pub(super) fn end_encoding_prepared<P>(
        &self,
        nonblocking: bool,
        prepare: impl FnOnce() -> Result<P, crate::kernels::MetalKernelError>,
        close: impl FnOnce(
            super::command_buffer::EndingEncoder<'_>,
            &mut EncoderState,
            P,
        ) -> Result<(), crate::kernels::MetalKernelError>,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let mut wake = super::command_buffer::EncoderEndWake::pending(&self.command_buffer);
        let mut state = if nonblocking {
            self.state
                .try_lock()
                .ok_or(crate::kernels::MetalKernelError::RecordingAdmission(
                    crate::command_lifecycle::LifecycleError::Busy,
                ))?
        } else {
            self.state.lock()
        };
        if state.close_succeeded {
            return Ok(());
        }
        if state.ended {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "encoder already ended",
            ));
        }
        // self.command_buffer remains a live parent anchor even if unwind
        // drops this nonfinal lease before the state guard. The wake guard was
        // declared first and notifies only after the state lock is released.
        let prepared = prepare()?;
        let mut recording = state.recording.take();
        let mut ended = false;
        let result = match recording.as_mut() {
            Some(admission) => admission.end(&mut ended, false, |scope| {
                wake.arm();
                state.ended = true;
                close(scope, &mut state, prepared)
            }),
            None => {
                wake.arm();
                state.ended = true;
                close(
                    super::command_buffer::EndingEncoder::raw_compat(&self.command_buffer),
                    &mut state,
                    prepared,
                )
            }
        };
        state.recording = recording;
        if result.is_ok() {
            state.close_succeeded = true;
        }
        result
    }

    /// The GPU clock of the queue this encoder's command buffer is committed on.
    pub fn with_clock(mut self, clock: Arc<crate::kernels::metal::GpuClock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Count this encoder's dispatches into `census` as well as the
    /// process-wide census.
    pub fn with_census(
        mut self,
        census: Arc<crate::kernels::metal::profile::DeviceCensus>,
    ) -> Self {
        self.census = Some(census);
        self
    }

    /// Record this encoder's dispatches into `trace` as well as the
    /// process-wide trace.
    pub fn with_trace(mut self, trace: Arc<trace::DeviceTrace>) -> Self {
        self.trace = Some(trace);
        self
    }

    /// The clock a CPU write into a buffer this encoder's predecessors may still
    /// be reading must consult, when the encoder has one.
    pub fn clock(&self) -> Option<&crate::kernels::metal::GpuClock> {
        self.clock.as_deref()
    }

    /// Order this session after the last writer of `ptr`, if any -- reading the
    /// cross-encoder map once per buffer per session, and waiting once per
    /// fence.
    ///
    /// Called under the session's state lock, which the bind takes anyway for
    /// its hazard bookkeeping: one acquisition per bind where there were two
    /// (the map's `RwLock` and this one on a hit). The map read is skipped
    /// for a buffer this session has already probed -- see
    /// `EncoderState::probed` for why a repeat cannot find a different writer.
    /// The `waitForFence` is issued with the lock held; the guard already
    /// holds the outer `Commands` lock for the whole dispatch, so nothing
    /// contends, and the call does not re-enter this state.
    fn wait_for_buffer(&self, s: &mut EncoderState, ptr: usize) {
        if !s.probed.insert(ptr) {
            return;
        }
        s.fence_probes += 1;
        let fence = {
            // Per-buffer-per-session read: the frequent access, and why the map
            // is an `RwLock`. `parking_lot` guards are infallible -- no `unwrap`.
            let map = self.prev_ce_outputs.read(Side::Host);
            map.get(&ptr).cloned()
        };
        // A completed writer is ordered already; its entry awaits the next
        // encoder close to be removed (`Commands`' `FenceBook`).
        let Some(fence) = fence.filter(|f| !f.is_completed()) else {
            return;
        };
        s.fence_hits += 1;
        if s.waited_fences.insert(Arc::as_ptr(&fence) as usize) {
            s.fence_waits += 1;
            self.raw.waitForFence(fence.raw());
        }
    }

    /// `(map reads, hits, waits)` this session has made -- the engagement
    /// counters the fence-probe gate (`fence_probe_tests`) asserts absolutely.
    /// Test-only: the shipping consumer is the encoder close, which reads the
    /// fields under the lock it already holds and sums them into the profiler.
    #[cfg(test)]
    pub(crate) fn fence_stats(&self) -> (u32, u32, u32) {
        let s = self.state.lock();
        (s.fence_probes, s.fence_hits, s.fence_waits)
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.auto_barrier();
        trace::record_dispatch(
            self.trace.as_deref(),
            size_tuple(threads_per_grid),
            size_tuple(threads_per_threadgroup),
            false,
        );
        arena::note_dispatch();
        self.record_dispatch();
        if !self.offer_to_executor(
            Grid::Threads(threads_per_grid.into()),
            threads_per_threadgroup,
        ) {
            return;
        }
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.auto_barrier();
        trace::record_dispatch(
            self.trace.as_deref(),
            size_tuple(threadgroups_per_grid),
            size_tuple(threads_per_threadgroup),
            true,
        );
        arena::note_dispatch();
        self.record_dispatch();
        if !self.offer_to_executor(
            Grid::Threadgroups(threadgroups_per_grid.into()),
            threads_per_threadgroup,
        ) {
            return;
        }
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    /// Offer this dispatch to the executor; `true` means encode it normally.
    ///
    /// The early return on the classical path is what keeps `DESIGN.md` §11.1's
    /// "must not regress" structural: with no executor installed this is one
    /// predictable branch, and the `DispatchRecord` -- which costs an `Arc<str>`
    /// clone and a lock acquisition -- is never built.
    #[inline(always)]
    fn offer_to_executor(&self, grid: Grid, threads_per_threadgroup: MTLSize) -> bool {
        if self.executor.is_classical() {
            return true;
        }
        let kernel = self.state.lock().current_pipeline.clone();
        match self.executor.dispatch_action(&DispatchRecord {
            kernel,
            grid,
            threads_per_threadgroup: threads_per_threadgroup.into(),
        }) {
            DispatchAction::Encode => true,
            DispatchAction::Suppress => false,
            DispatchAction::ExecuteIcb {
                icb,
                location,
                length,
            } => {
                self.execute_icb_range(&icb, location, length);
                false
            }
            DispatchAction::ExecuteIcbThenEncode {
                icb,
                location,
                length,
            } => {
                self.execute_icb_range(&icb, location, length);
                self.barrier_after_icb();
                true
            }
        }
    }

    /// Order the dispatch about to be encoded after a range issued a moment
    /// ago, at a point its own `auto_barrier` has already passed.
    ///
    /// The hazard sets are left as they are rather than rolled: they already
    /// hold this dispatch's bindings (bound before `auto_barrier` ran), and
    /// the next position must still be checked against them. Only the latch
    /// is discharged, because the fence just emitted is the one it asked for.
    fn barrier_after_icb(&self) {
        let mut s = self.state.lock();
        self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        let kinds = s.pending_kinds.take();
        trace::record_barrier(self.trace.as_deref(), kinds);
        crate::kernels::metal::hazard_audit::record_barrier_kinds(self.raw.audit_id(), kinds);
        s.needs_barrier = false;
    }

    /// Run `length` pre-encoded ICB commands from `location`, here.
    ///
    /// Called at the position where a run of replayed dispatches begins, so the
    /// commands land between the same two classically-encoded neighbours the
    /// originals had. `metal::icb::Run` carries why that placement is the whole
    /// of the correctness argument.
    pub(crate) fn execute_icb_range(
        &self,
        range: &crate::kernels::metal::executor::IcbRange,
        location: usize,
        length: usize,
    ) {
        // The range is ordered against earlier encoder sessions here, on the
        // encoder that executes it, and not only through the binds its
        // members made: a run validated to its tail executes after every
        // member's bind, and a cut between head and tail would leave those
        // binds' fence waits on an encoder that has already ended. Waiting
        // once per buffer is the same per-buffer wait a classical bind makes,
        // and registering the writes is what lets the next session wait on
        // this one.
        {
            let mut s = self.state.lock();
            for buf in &range.resident {
                self.wait_for_buffer(&mut s, buf.raw_ptr() as usize);
            }
            for &ptr in &range.written {
                s.all_outputs.insert(ptr);
            }
        }
        // SAFETY: every resource a command in this run binds is in the run's
        // `resident` -- the plan collects them from the bindings as it encodes
        // each command, so the set cannot fall behind the commands -- and
        // residency has to be declared on *this* encoder, which is why it is
        // done at every execute rather than once at install.
        // Omitting it is silent corruption rather than an error (`DESIGN.md`
        // §3.7), and unified memory can mask the omission.
        unsafe {
            // One batched residency declaration for the whole run, from the array
            // cached once at record (`IcbRange::resident_ptrs`) — a single
            // `useResources:count:usage:` msgSend rather than one `useResource:`
            // per buffer every replay. Skip when empty (a count-0 call takes a
            // pointer nothing backs).
            if let Some(first) = range.resident_ptrs.first() {
                self.raw.useResources_count_usage(
                    core::ptr::NonNull::from(first),
                    range.resident_ptrs.len(),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            self.raw.executeCommandsInBuffer_withRange(
                &range.icb,
                objc2_foundation::NSRange { location, length },
            );
        }
        trace::record_execute_range(range.resident.len());
    }

    /// Attribute this dispatch to the currently bound pipeline, for the
    /// profiler and for the hazard audit.
    ///
    /// Both dispatch entry points funnel through here so neither consumer's
    /// count can drift from the number of dispatches actually encoded.
    ///
    /// **The two consumers are asked separately, and the profiler's early
    /// return is deliberately not shared.** Folding the audit in behind
    /// `profile::enabled()` would make it silently record nothing whenever
    /// profiling happened to be off -- an instrument reporting a different
    /// quantity than its name, which §11.3n records live in this very file
    /// (`record_dispatch`'s own doc comment was falsified by #115 adding
    /// suppression without editing it). With the `hazard-audit` feature off,
    /// `hazard_audit::record_dispatch` is an empty inline function, so the
    /// separation costs nothing.
    ///
    /// **The profiler is handed the `Arc<str>`, not a `&str`, and that is what
    /// keeps the census allocation-free** (issue #579). Its per-label map is
    /// keyed on `Arc<str>`, so a kernel already in the map costs a hash and a
    /// compare; only a label seen for the *first* time clones the handle into
    /// the key. Steady-state decode reaches 17 distinct kernels at 675
    /// dispatches per token (measured, §11.2), so after the first token every
    /// one of them is a lookup. The audit keeps taking `&str` because it builds
    /// an owned `AuditDispatch` record per dispatch regardless, and it is
    /// compiled out entirely by default.
    #[inline]
    fn record_dispatch(&self) {
        let profiling = crate::kernels::metal::profile::enabled();
        if !profiling && !cfg!(feature = "hazard-audit") {
            return;
        }
        let (name, input_bytes) = {
            let mut s = self.state.lock();
            (
                s.current_pipeline.clone(),
                std::mem::take(&mut s.dispatch_input_max),
            )
        };
        if profiling {
            crate::kernels::metal::profile::record_dispatch(name.as_ref(), input_bytes);
            if let Some(census) = &self.census {
                census.record(name.as_ref());
            }
        }
        crate::kernels::metal::hazard_audit::record_dispatch(
            self.raw.audit_id(),
            name.as_deref().unwrap_or("<unnamed>"),
        );
    }

    fn auto_barrier(&self) {
        // Issue #144: ask whether the executor will replay this position as a
        // non-head member of a run already in flight, in which case the ordering
        // this fence would express is already encoded on the ICB command that
        // replaces it (`ReplayBarriers`, and
        // `measurements/issue-144-predicate.md` for the edge-level argument).
        //
        // **The query is non-advancing, and that is the whole seam.**
        // `dispatch_action` cannot be used here: it drains `pending`,
        // `pending_retained` and `pending_pipeline` with three `mem::take`s, so
        // a speculative call would record the next dispatch with no bindings.
        // `Executor::disposition` is the decision half split out from that
        // drain -- a phase check and a window test, touching neither the
        // bindings nor the cursor.
        //
        // `Classical` answers `Unknown` without a virtual call, so on the
        // default path this is one predictable branch and the behaviour is
        // byte-for-byte what shipped.
        let suppress = matches!(
            self.executor.disposition(),
            crate::kernels::metal::executor::Disposition::ReplayedInFlightRun
        );

        let mut s = self.state.lock();
        if s.needs_barrier {
            if suppress {
                // Do NOT emit, and do NOT discharge. The state is rolled exactly
                // as the no-barrier arm below rolls it, so `needs_barrier` stays
                // set and `prev_*` keeps accumulating.
                //
                // **This is the correctness centre of the change, and the
                // alternative is a silent wrong answer.** Clearing
                // `needs_barrier` and replacing `prev_*` -- i.e. doing
                // everything the emitting arm does except the emit -- would
                // discharge the pending edge for every *later* dispatch as well,
                // including the uncovered gap positions between the runs. Those
                // are encoded classically and depend on candle's fence; §11.3m
                // is exactly this asymmetry, and §11.3p obstacle 2 is the
                // population it would break. Suppressing the emit while keeping
                // the state is what confines the change to the position whose
                // ordering the ICB has genuinely re-expressed.
                //
                // So the invariant is: a suppressed barrier is *deferred*, not
                // dropped. If the very next position is a gap dispatch, it still
                // sees `needs_barrier` and emits -- which is why the 123
                // interleaved positions keep their ordering.
                let mut next_out = s.next_outputs.take();
                s.prev_outputs.absorb(&mut next_out);
                let mut next_in = s.next_inputs.take();
                s.prev_inputs.absorb(&mut next_in);
                // `pending_kinds` is deliberately NOT drained here, for exactly
                // the reason `needs_barrier` and `prev_*` are not: a suppressed
                // barrier is *deferred, not dropped*. Draining the kinds while
                // keeping the latch would attribute the eventual emission to
                // only the directions seen after this point (issue #185).
                trace::record_barrier_suppressed();
                // The audit must know a barrier was *not* emitted here: a
                // suppressed barrier is not a surviving fence, and whether the
                // ICB's `setBarrier` and the session seams cover what it would
                // have ordered is the whole question #144 poses (issue #185).
                crate::kernels::metal::hazard_audit::record_barrier_suppressed(self.raw.audit_id());
                return;
            }
            self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            // Observed here rather than reconstructed downstream: the barrier
            // count `DESIGN.md` §9.2e requires is a property of this branch, and
            // deriving it from a trace of bindings would additionally have to
            // model where each encoder session began.
            //
            // The kinds go with it, at the same site and for the same reason
            // (issue #185): which directions this barrier is *for* is known here
            // and nowhere downstream, because `prev_*` is replaced two lines
            // below and the evidence is gone.
            let kinds = s.pending_kinds.take();
            trace::record_barrier(self.trace.as_deref(), kinds);
            crate::kernels::metal::hazard_audit::record_barrier_kinds(self.raw.audit_id(), kinds);
            s.needs_barrier = false;
            // Replaced, not extended: everything bound before the barrier is
            // now ordered against everything after it.
            s.prev_outputs = s.next_outputs.take();
            s.prev_inputs = s.next_inputs.take();
        } else {
            // Absorb IN PLACE and clear, rather than `take` + absorb: the
            // rotation happens once per dispatch, and handing the maps'
            // allocations away each time made every dispatch regrow them
            // (lloom #727, `BoundSet::absorb`).
            // `&mut *s` once, then split the fields: the guard's `DerefMut`
            // cannot be called twice in one expression.
            let st = &mut *s;
            st.prev_outputs.absorb(&mut st.next_outputs);
            st.next_outputs.clear();
            st.prev_inputs.absorb(&mut st.next_inputs);
            st.next_inputs.clear();
        }
    }

    /// The binding a hazard check should compare, including its extent.
    ///
    /// `offset` is the byte the kernel starts at -- already including the
    /// arena's `base_offset` where there is one -- and `length()` is what the
    /// handle addresses from there, which for an arena view is its slot rather
    /// than the whole arena.
    #[inline]
    fn bound_range(buf: &Buffer, offset: usize) -> BoundRange {
        BoundRange {
            ptr: buf.raw_ptr() as usize,
            offset,
            len: buf.length(),
        }
    }

    /// # Safety
    /// Explicit legacy/raw interoperability: caller retains genuine originating
    /// reuse authority and native/accounting owners through actual terminal work,
    /// cached native references and every escaped future emission. Bare derived
    /// Buffer/native pointer retention is NOT reuse authority. No tracked pin is
    /// installed by this method; use the admitted typed operation instead.
    pub unsafe fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        // Existing legacy spelling remains until coordinated raw API/caller
        // migration. It is not enrolled by this additive tracked route.
        self.set_input_buffer_native(
            index,
            buffer,
            offset,
            buffer.map(|buffer| unsafe { buffer.raw_unchecked() }),
        );
    }
    fn set_input_buffer_native(
        &self,
        index: usize,
        buffer: Option<&Buffer>,
        offset: usize,
        native: Option<&ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) {
        let index = self.capture_buffer_index(index);
        self.set_input_buffer_native_resolved(index, buffer, offset, native);
    }
    // The tracked batch has already reserved its actual packed indices. Legacy
    // single bindings preserve their original capture behavior through wrapper.
    fn set_input_buffer_native_resolved(
        &self,
        index: usize,
        buffer: Option<&Buffer>,
        offset: usize,
        native: Option<&ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) {
        // An arena slot is a region of a shared allocation, so the byte the
        // kernel must start at is the slot's base plus the layout offset the
        // caller computed. `base_offset` is 0 for every ordinary allocation,
        // which is why this is a no-op everywhere except the arena -- see
        // `Buffer::base_offset` for why the addition belongs here and not at
        // the call sites.
        let offset = offset + buffer.map_or(0, Buffer::base_offset);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(
                self.trace.as_deref(),
                index,
                ptr,
                offset,
                buf.length(),
                false,
            );
            // The clock a liveness recording must use: a value is live until the
            // last dispatch that binds it (`DESIGN.md` §6.7 L4).
            arena::note_bind(ptr);
            let range = Self::bound_range(buf, offset);
            let mut s = self.state.lock();
            late_bind::on_bind(ptr, false, s.current_pipeline.as_deref().unwrap_or("?"));
            if crate::kernels::metal::profile::counters_enabled() {
                s.dispatch_input_max = s.dispatch_input_max.max(buf.length());
            }
            // Read-after-write against an earlier encoder: order against that
            // buffer's last writer only, under the lock this bind holds anyway.
            self.wait_for_buffer(&mut s, ptr);
            let key = s.hazard_key;
            // Read-after-write within this encoder. See `read_hazards` for why a
            // read has exactly one possible direction (issue #185).
            let kinds = read_hazards(&s.prev_outputs, key, &range);
            if !kinds.is_empty() {
                s.needs_barrier = true;
                s.pending_kinds.merge(kinds);
            }
            s.next_inputs.insert(key, range);
            crate::kernels::metal::hazard_audit::record_binding(self.raw.audit_id(), &range, false);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, false);
            }
        }
        let listing = buffer.map_or(DeviceListing::No, Buffer::device_listing);
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(native, offset, index, listing)
        }
    }

    /// # Safety
    /// Explicit legacy/raw interoperability: caller retains genuine originating
    /// reuse authority and native/accounting owners through actual terminal work,
    /// cached native references and every escaped future emission. Bare derived
    /// Buffer/native pointer retention is NOT reuse authority. No tracked pin is
    /// installed by this method; use the admitted typed operation instead.
    pub unsafe fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        self.set_output_buffer_native(
            index,
            buffer,
            offset,
            buffer.map(|buffer| unsafe { buffer.raw_unchecked() }),
        );
    }
    fn set_output_buffer_native(
        &self,
        index: usize,
        buffer: Option<&Buffer>,
        offset: usize,
        native: Option<&ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) {
        let index = self.capture_buffer_index(index);
        self.set_output_buffer_native_resolved(index, buffer, offset, native);
    }
    // The tracked batch has already reserved its actual packed indices. Legacy
    // single bindings preserve their original capture behavior through wrapper.
    fn set_output_buffer_native_resolved(
        &self,
        index: usize,
        buffer: Option<&Buffer>,
        offset: usize,
        native: Option<&ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) {
        // See `set_input_buffer`: the arena's `base + offset`, added once at the
        // choke point every binding passes through.
        let offset = offset + buffer.map_or(0, Buffer::base_offset);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(
                self.trace.as_deref(),
                index,
                ptr,
                offset,
                buf.length(),
                true,
            );
            arena::note_bind(ptr);
            let range = Self::bound_range(buf, offset);
            let mut s = self.state.lock();
            late_bind::on_bind(ptr, true, s.current_pipeline.as_deref().unwrap_or("?"));
            // Write-after-write or write-after-read against an earlier encoder.
            self.wait_for_buffer(&mut s, ptr);
            let key = s.hazard_key;
            // Write-after-write, and write-after-read. See `write_hazards` for
            // why both tests run where the original short-circuited on `||`
            // (issue #185).
            let kinds = write_hazards(&s.prev_outputs, &s.prev_inputs, key, &range);
            if !kinds.is_empty() {
                s.needs_barrier = true;
                s.pending_kinds.merge(kinds);
            }
            s.next_outputs.insert(key, range);
            crate::kernels::metal::hazard_audit::record_binding(self.raw.audit_id(), &range, true);
            s.all_outputs.insert(ptr);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, true);
            }
        }
        let listing = buffer.map_or(DeviceListing::No, Buffer::device_listing);
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(native, offset, index, listing)
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    /// Capture this scalar into the packed-params staging area instead of
    /// binding it inline, when a capture is open.
    ///
    /// Returns `false` when none is, which is the classical path and the
    /// default: the caller then does exactly what it did before. The relaxed
    /// load is the whole of the cost in that case -- no lock is taken.
    ///
    /// This is `DESIGN.md` §11.3b's "one function, not 51 call sites".
    /// `EncoderParam::set_param` is the only place a primitive reaches
    /// `setBytes`, so diverting it here leaves every `set_params!` site and
    /// every `call_*` entry point untouched: the caller still passes a `u32`,
    /// and where it lands becomes the encoder's business.
    #[inline(always)]
    pub(crate) fn capture_scalar(&self, bytes: &[u8], align: usize) -> bool {
        if !self.capturing.load(Ordering::Relaxed) {
            return false;
        }
        let mut cap = self.param_capture.lock();
        // Match the layout the kernel will read: pad to the field's own
        // alignment before appending. `DESIGN.md` §15.1 -- a field at the wrong
        // offset is silent corruption, so the padding rule has to be the one
        // MSL actually applies, and `reduce_params_layout_matches_metal` is
        // what proves it is.
        while !cap.bytes.len().is_multiple_of(align) {
            cap.bytes.push(0);
        }
        cap.bytes.extend_from_slice(bytes);
        true
    }

    /// Promote a `setBytes` array to a device buffer, when capturing.
    ///
    /// `dims` and `strides` cannot join the packed struct -- their length comes
    /// from the tensor's layout -- but they do not need to: an ICB command can
    /// bind a buffer of any length, it just has no `setBytes` at all. So under
    /// capture they become a real buffer and keep their own argument slot.
    ///
    /// Allocating per call is deliberate for this change and is not what a
    /// decode path would do; see `with_packed_params` in `kernels/reduce.rs`
    /// for why that is acceptable here and what a plan-owned buffer would look
    /// like instead.
    #[inline]
    /// # Safety
    /// Original packed staging interoperability. Caller closes actual staging
    /// native references, reuse/accounting and future emission; this method is
    /// not a counted generation/ICB ownership proof.
    pub(crate) unsafe fn capture_array(&self, len_bytes: usize, bytes: *const c_void) -> bool {
        if !self.capturing.load(Ordering::Relaxed) {
            return false;
        }
        let index = self.capture_buffer_index(usize::MAX);
        let device = self.command_buffer.device();
        let Ok(buffer) =
            device.new_buffer_with_data(bytes, len_bytes, crate::kernels::RESOURCE_OPTIONS)
        else {
            // Allocation failure here would otherwise bind nothing and let the
            // kernel read a stale slot. Reporting it is not possible through
            // `EncoderParam`, which returns unit, so fail loudly rather than
            // silently: this path is behind an opt-in style and is not reached
            // by any classical dispatch.
            panic!("packed-params staging allocation failed for {len_bytes} bytes");
        };
        // Bound directly rather than through `set_input_buffer`, which would
        // renumber a second time.
        let ptr = buffer.raw_ptr() as usize;
        {
            let range = Self::bound_range(&buffer, 0);
            let mut s = self.state.lock();
            late_bind::on_bind(ptr, false, s.current_pipeline.as_deref().unwrap_or("?"));
            self.wait_for_buffer(&mut s, ptr);
            let key = s.hazard_key;
            s.next_inputs.insert(key, range);
            crate::kernels::metal::hazard_audit::record_binding(self.raw.audit_id(), &range, false);
        }
        unsafe {
            self.raw.setBuffer_offset_atIndex(
                Some(buffer.raw_unchecked()),
                0,
                index,
                buffer.device_listing(),
            )
        }
        // The staging buffer must stay alive until the dispatch it feeds has
        // completed, not merely until it is encoded. It is parked here and
        // handed to the caller by `end_param_capture`, which holds it across
        // the dispatch -- releasing it at capture-close would drop it while the
        // GPU may still be reading, which is the in-flight-reuse failure
        // `DESIGN.md` §2.3.8b describes and no fence can see.
        self.param_capture.lock().staged.push(buffer);
        true
    }

    /// The index a buffer should actually bind at, given any scalars already
    /// diverted out of the argument list ahead of it.
    ///
    /// Returns the caller's own index unless a capture is open.
    #[inline(always)]
    fn capture_buffer_index(&self, index: usize) -> usize {
        if !self.capturing.load(Ordering::Relaxed) {
            return index;
        }
        let mut cap = self.param_capture.lock();
        let slot = cap.next_buffer;
        cap.next_buffer += 1;
        slot
    }

    /// Begin capturing scalars into a packed-params block.
    ///
    /// Scoped rather than persistent: [`Self::end_param_capture`] returns the
    /// bytes and closes it, so a capture cannot leak into the next dispatch.
    pub fn begin_param_capture(&self) {
        let mut cap = self.param_capture.lock();
        cap.bytes.clear();
        cap.staged.clear();
        // Slot 0 is the params buffer, bound by the caller after the capture
        // closes, so the first real buffer goes to 1.
        cap.next_buffer = 1;
        drop(cap);
        self.capturing.store(true, Ordering::Relaxed);
    }

    /// Close a capture opened by [`Self::begin_param_capture`], returning the
    /// packed bytes and any buffers staged for arrays.
    ///
    /// The caller must hold the returned buffers until the dispatch is
    /// complete; see [`Self::capture_array`].
    ///
    /// The trailing pad matters: C++ pads a struct up to its own alignment, so
    /// `sizeof` is always a multiple of `alignof`. Without it a `{u64,u32}`
    /// would ship 12 bytes where the kernel reads 16.
    pub fn end_param_capture(&self, align: usize) -> (Vec<u8>, Vec<Buffer>) {
        self.capturing.store(false, Ordering::Relaxed);
        let mut cap = self.param_capture.lock();
        let mut bytes = std::mem::take(&mut cap.bytes);
        let staged = std::mem::take(&mut cap.staged);
        while align != 0 && !bytes.len().is_multiple_of(align) {
            bytes.push(0);
        }
        (bytes, staged)
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        trace::record_pipeline(
            self.trace.as_deref(),
            pipeline.name().unwrap_or("<unnamed>"),
        );
        self.note_pipeline(pipeline);
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    /// Remember which kernel is bound, so the next dispatch can be attributed,
    /// and tell the executor if one is installed.
    ///
    /// Two consumers, one hook: the profiler counts dispatches per kernel, and
    /// an executor needs the same name to validate a recorded sequence. Both
    /// are off by default, so the classical path pays only the two tests.
    #[inline]
    fn note_pipeline(&self, pipeline: &ComputePipeline) {
        let has_executor = !self.executor.is_classical();
        if has_executor {
            self.executor.will_set_pipeline(pipeline);
        }
        if !has_executor && !crate::kernels::metal::profile::enabled() && !late_bind::armed() {
            return;
        }
        // `name_handle` clones the `Arc<str>` the pipeline already holds. The
        // form this replaced built a fresh one from the `&str` at every call --
        // once per dispatch, ~677 per decode token under
        // `CANDLE_METAL_PROFILE=1` (`DESIGN.md` §11.2), which is the profiler
        // perturbing the quantity it exists to measure (§2.4). The field it is
        // stored into is `Option<Arc<str>>` already, so this is a refcount bump
        // into a slot that was always shaped for one (issue #579).
        let name = pipeline.name_handle();
        self.state.lock().current_pipeline = name;
    }

    /// Insert a memory barrier at buffers scope.
    pub fn insert_memory_barrier(&self) {
        self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// Wait for a fence before continuing execution.
    pub fn wait_for_fence(&self, fence: &Fence) {
        self.raw.waitForFence(fence.raw());
    }

    /// Update a fence after commands complete.
    pub fn update_fence(&self, fence: &Fence) {
        self.raw.updateFence(fence.raw());
    }

    pub fn end_encoding(&self) -> Result<(), crate::kernels::MetalKernelError> {
        if let Some(finalizer) = &self.finalizer {
            return finalizer.end_encoding(self);
        }
        self.end_encoding_with(|_, _| {
            self.raw.updateFence(self.fence.raw());
            self.raw.endEncoding();
            Ok(())
        })
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        trace::record_pipeline(
            self.trace.as_deref(),
            pipeline.name().unwrap_or("<unnamed>"),
        );
        self.note_pipeline(pipeline);
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

/// RAII guard that pops a Metal debug group on drop. Debug groups are a stack
/// scoped to the push/pop range, so each dispatch is attributed correctly on
/// the shared concurrent encoder where `set_label` cannot.
#[cfg(feature = "debug-labels")]
pub struct DebugGroupGuard<'a> {
    encoder: &'a ComputeCommandEncoder,
}

#[cfg(feature = "debug-labels")]
impl Drop for DebugGroupGuard<'_> {
    fn drop(&mut self) {
        self.encoder.raw.popDebugGroup();
    }
}

#[cfg(feature = "debug-labels")]
impl ComputeCommandEncoder {
    /// Push a Metal debug group scoped to the returned guard.
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> DebugGroupGuard<'_> {
        self.raw.pushDebugGroup(&NSString::from_str(label));
        DebugGroupGuard { encoder: self }
    }
}

/// One exact copy2d_u8 rectangle, using already admitted tuple positions.
/// Positions identify installed owners, not native buffer indices. The fixed
/// shader schema supplies those; no Packed cursor or argument callback runs.
pub(crate) struct ByteCopyStep {
    pub(crate) source: usize,
    pub(crate) destination: usize,
    pub(crate) source_offset: usize,
    pub(crate) destination_offset: usize,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) packed: Option<usize>,
}
impl ComputeCommandEncoder {
    /// Lower all rectangles and optional overlap scratch phases under ONE
    /// original EDIT. Every native user is minted/installed before ANY effect.
    /// # Safety
    /// Pipeline is exact nongated copy2d_u8[_packed] (constant777=false). Split
    /// scalars below match four int64 fields; every Packed owner contains that
    /// step's initialized four-field block. Originating owners are genuine rich
    /// handles; all source bytes are initialized. Overlap must use independent
    /// placement scratch, write its complete range before reading it, and honor
    /// actual executor barrier semantics. Escaped ICB/emission/IO custody is an
    /// independent terminal owner obligation, not established by these bounds.
    pub(crate) unsafe fn dispatch_byte_copy_steps<const N: usize>(
        &self,
        pipeline: &ComputePipeline,
        owners: &[&super::BufferHandle; N],
        steps: &[ByteCopyStep],
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        let invalid = Error::QueueUnavailable;
        if steps.is_empty() {
            return Ok(());
        }
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        if max_threads == 0 {
            return Err(invalid("byte copy pipeline has zero thread width"));
        }
        // Check the ENTIRE operation before original EDIT/mint/credit/effect.
        // Grid IDs are exact uint2, address/stride arithmetic exact signed64.
        for step in steps {
            if step.rows == 0
                || step.columns == 0
                || step.rows > u32::MAX as usize
                || step.columns > u32::MAX as usize
            {
                return Err(invalid("copy rectangle uint grid refused"));
            }
            let bytes = step
                .rows
                .checked_mul(step.columns)
                .filter(|bytes| *bytes <= i64::MAX as usize)
                .ok_or(invalid("copy rectangle signed address extent refused"))?;
            let source = owners
                .get(step.source)
                .ok_or(invalid("copy source tuple position refused"))?;
            let destination = owners
                .get(step.destination)
                .ok_or(invalid("copy destination tuple position refused"))?;
            crate::native_binding_preflight::checked_byte_range(
                source.base_offset(),
                source.length(),
                step.source_offset,
                bytes,
            )
            .ok_or(invalid("copy source originating range refused"))?;
            crate::native_binding_preflight::checked_byte_range(
                destination.base_offset(),
                destination.length(),
                step.destination_offset,
                bytes,
            )
            .ok_or(invalid("copy destination originating range refused"))?;
            if let Some(position) = step.packed {
                let packed = owners
                    .get(position)
                    .ok_or(invalid("copy Packed tuple position refused"))?;
                crate::native_binding_preflight::checked_byte_range(
                    packed.base_offset(),
                    packed.length(),
                    0,
                    32,
                )
                .ok_or(invalid("copy Packed block range refused"))?;
            }
        }
        let mut edit = self.binding_edit()?;
        if self.capturing.load(Ordering::Acquire) {
            return Err(invalid(
                "byte copy requires explicit prepared Packed ownership",
            ));
        }
        let generation = self.command_buffer.binding_generation()?;
        let mut prepared: [Option<super::native_generation_bindings::PreparedBuffer>; N] =
            std::array::from_fn(|_| None);
        for (slot, owner) in prepared.iter_mut().zip(owners) {
            *slot = Some(generation.prepare_buffer(owner.as_ref())?);
        }
        // SAFETY: exact original EDIT spans ALL native-user publication and
        // both phases; no other writer can consume this generation's cursor.
        let installed = unsafe {
            generation.install_prepared(prepared.map(|p| p.expect("preminted copy owner")))
        }?;
        edit.entered_effects = true;
        self.set_compute_pipeline_state(pipeline);
        for step in steps {
            let (input_index, output_index) = if let Some(position) = step.packed {
                let lease = installed[position].lease();
                let buffer = lease
                    .buffer_for_binding()
                    .map_err(|_| invalid("installed copy Packed role refused"))?;
                self.set_input_buffer_native_resolved(
                    0,
                    Some(buffer),
                    0,
                    Some(unsafe { lease.raw_unchecked() }),
                );
                (1, 2)
            } else {
                let values = [
                    step.rows as i64,
                    step.columns as i64,
                    step.columns as i64,
                    step.columns as i64,
                ];
                for (index, value) in values.iter().enumerate() {
                    self.set_bytes_directly(
                        index,
                        std::mem::size_of::<i64>(),
                        std::ptr::from_ref(value).cast(),
                    );
                }
                (4, 5)
            };
            let source = installed[step.source].lease();
            let destination = installed[step.destination].lease();
            let source_buffer = source
                .buffer_for_binding()
                .map_err(|_| invalid("installed copy source role refused"))?;
            let destination_buffer = destination
                .buffer_for_binding()
                .map_err(|_| invalid("installed copy destination role refused"))?;
            self.set_input_buffer_native_resolved(
                input_index,
                Some(source_buffer),
                step.source_offset,
                Some(unsafe { source.raw_unchecked() }),
            );
            self.set_output_buffer_native_resolved(
                output_index,
                Some(destination_buffer),
                step.destination_offset,
                Some(unsafe { destination.raw_unchecked() }),
            );
            // EXACT grid, no rounded uint IDs. The existing dispatch path keeps
            // hazard/trace/executor behavior and real write->read barriers.
            self.dispatch_threads(
                MTLSize {
                    width: step.rows,
                    height: step.columns,
                    depth: 1,
                },
                MTLSize {
                    width: max_threads.min(step.rows),
                    height: 1,
                    depth: 1,
                },
            );
        }
        edit.finish()
    }
}

pub struct BlitCommandEncoder {
    pub(crate) raw: super::raw_encoder::Blit,
    recording: Option<super::command_buffer::EncoderAdmission>,
    ended: bool,
    close_succeeded: bool,
    /// Per-encoder fence, updated at end_encoding.
    fence: Arc<Fence>,
    /// Shared global cross-encoder output map.
    prev_ce_outputs: PrevCeOutputs,
    /// Buffer pointers written by this blit encoder (registered in global map at end_encoding).
    tracked_outputs: Vec<usize>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    /// Import an already active native blit encoder without a tracked parent.
    ///
    /// # Safety
    ///
    /// `raw` must come from Apple Metal or a separately audited equivalent and
    /// be its parent's unique active encoder. No independent wrapper or external
    /// alias may encode/end it while this wrapper owns it. The caller retains
    /// the exact parent and all bound resources through their native use (also
    /// unretained resources), ends this encoder before parent submission, and
    /// obeys documented native thread/phase, callback and final-release rules.
    /// This import does not acquire engine ownership or a parent lifetime pin.
    pub unsafe fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> BlitCommandEncoder {
        Self::with_backend(
            super::raw_encoder::Blit::Metal3(raw),
            fence,
            prev_ce_outputs,
        )
    }
    pub(crate) fn with_backend(
        raw: super::raw_encoder::Blit,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder {
            raw,
            recording: None,
            ended: false,
            close_succeeded: false,
            fence,
            prev_ce_outputs,
            tracked_outputs: Vec::new(),
        }
    }

    /// Wait for a fence before continuing execution.
    pub fn wait_for_fence(&self, fence: &Fence) {
        self.raw.waitForFence(fence.raw());
    }

    /// Update a fence after commands complete.
    pub fn update_fence(&self, fence: &Fence) {
        self.raw.updateFence(fence.raw());
    }

    pub(super) fn with_recording(
        mut self,
        admission: super::command_buffer::EncoderAdmission,
    ) -> Self {
        crate::kernels::metal::hazard_audit::record_pass_open(
            self.raw.audit_id(),
            admission.command_buffer().identity(),
        );
        self.recording = Some(admission);
        self
    }

    pub fn end_encoding(&mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.end_encoding_mode(false)
    }
    pub(super) fn retirement_closed(&self) -> bool {
        self.close_succeeded
    }

    pub(super) fn try_end_encoding(&mut self) -> Result<bool, crate::kernels::MetalKernelError> {
        match self.end_encoding_mode(true) {
            Ok(()) => Ok(true),
            Err(crate::kernels::MetalKernelError::RecordingAdmission(
                crate::command_lifecycle::LifecycleError::Busy,
            )) => Ok(false),
            Err(error) => Err(error),
        }
    }
    fn end_encoding_mode(
        &mut self,
        nonblocking: bool,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let Self {
            raw,
            recording,
            ended,
            close_succeeded,
            fence,
            prev_ce_outputs,
            tracked_outputs,
        } = self;
        // A confirmed close is idempotent for explicit end followed by guard
        // Drop. An unknown native exit is not a successful close.
        if *close_succeeded {
            return Ok(());
        }
        if *ended {
            return Err(crate::kernels::MetalKernelError::QueueUnavailable(
                "blit encoder already ended",
            ));
        }
        let prepare = || {
            if nonblocking {
                prev_ce_outputs.try_write().ok_or(
                    crate::kernels::MetalKernelError::RecordingAdmission(
                        crate::command_lifecycle::LifecycleError::Busy,
                    ),
                )
            } else {
                Ok(prev_ce_outputs.write(Side::Host))
            }
        };
        let close = |mut map: parking_lot::RwLockWriteGuard<'_, HashMap<usize, Arc<Fence>>>| {
            raw.updateFence(fence.raw());
            raw.endEncoding();
            for &out in tracked_outputs.iter() {
                map.insert(out, Arc::clone(fence));
            }
            Ok(())
        };
        let result = if let Some(admission) = recording.as_mut() {
            admission.end_prepared(ended, true, prepare, |_, map| close(map))
        } else {
            // Raw constructor compatibility has no engine recording lease.
            let map = prepare()?;
            *ended = true;
            close(map)
        };
        if result.is_ok() {
            *close_succeeded = true;
        }
        result
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    /// Wait on the last writer of each of `ptrs`, emitting each distinct fence
    /// once.
    ///
    /// `Commands::blit_command_encoder` already waits on every fence in
    /// `live_fences` before handing this encoder out, which covers every
    /// *compute* encoder that has ended. It does not cover a prior *blit*:
    /// `end_encoding` below registers this encoder's outputs in
    /// `prev_ce_outputs` but never adds its fence to `live_fences`. So a
    /// blit-after-blit dependency is recorded only in the map, and consulting
    /// the map here is what closes that gap.
    ///
    /// Metal tolerates a repeated `waitForFence`, but `copy_from_buffer` passes
    /// two buffers that often share a writer, so the dedup avoids emitting the
    /// same wait twice on the common path.
    fn wait_for_last_writers(&self, ptrs: &[usize]) {
        let fences: Vec<Arc<Fence>> = {
            // A read of the ordering map (blit-after-blit); see `PrevCeOutputs`.
            let map = self.prev_ce_outputs.read(Side::Host);
            let mut out: Vec<Arc<Fence>> = Vec::new();
            for ptr in ptrs {
                if let Some(f) = map.get(ptr).filter(|f| !f.is_completed()) {
                    if !out.iter().any(|seen| Arc::ptr_eq(seen, f)) {
                        out.push(Arc::clone(f));
                    }
                }
            }
            out
        };
        for fence in fences {
            self.raw.waitForFence(fence.raw());
        }
    }

    /// Only the already-selected legal aligned blit arm. A complete transfer
    /// caller selects its tracked compute/staged arm BEFORE acquiring this
    /// encoder for U8 misalignment/overlap; this method is not that dispatcher.
    /// Ordinary geometry/count/page refusal occurs before native effects.
    pub(crate) fn copy_from_handles_aligned(
        &mut self,
        source: &super::BufferHandle,
        source_offset: usize,
        destination: &super::BufferHandle,
        destination_offset: usize,
        size: usize,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        use crate::native_binding_preflight::{byte_copy_route, ByteCopyRoute};
        let (source_offset, destination_offset) = match byte_copy_route(
            source.base_offset(),
            source.length(),
            source_offset,
            destination.base_offset(),
            destination.length(),
            destination_offset,
            size,
            source.raw_ptr() == destination.raw_ptr(),
        )
        .ok_or(Error::QueueUnavailable(
            "blit copy exceeds originating view",
        ))? {
            ByteCopyRoute::Empty => return Ok(()),
            ByteCopyRoute::AlignedBlit {
                source,
                destination,
            } => (source, destination),
            ByteCopyRoute::Compute { .. } => {
                return Err(Error::QueueUnavailable(
                    "byte transfer requires its prepared compute/staged arm",
                ))
            }
        };
        let owner = self.recording.as_ref().ok_or(Error::QueueUnavailable(
            "raw blit encoder requires unsafe interoperability",
        ))?;
        if self.ended {
            return Err(Error::QueueUnavailable("blit encoder already ended"));
        }
        let buffer = owner.command_buffer();
        let ticket = owner.begin_binding_edit()?;
        let mut edit = BindingEdit {
            buffer,
            ticket: Some(ticket),
            entered_effects: false,
        };
        let generation = buffer.binding_generation()?;
        // Authentic rich handles span both counted mints. Legacy refusal occurs
        // before native retain/member clone; no bare-pointer lifetime inference.
        let source_prepared = generation.prepare_buffer(source)?;
        let destination_prepared = generation.prepare_buffer(destination)?;
        // SAFETY: SAME original encoder EDIT and complete immutable geometry;
        // shared whole-batch planner either installs BOTH pins or consumes none.
        let installed =
            unsafe { generation.install_prepared([source_prepared, destination_prepared]) }?;
        let source_lease = installed[0].lease();
        let destination_lease = installed[1].lease();
        let source = source_lease.buffer_for_binding().map_err(|_| {
            generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            Error::QueueUnavailable("installed blit source lost External custody")
        })?;
        let destination = destination_lease.buffer_for_binding().map_err(|_| {
            generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            Error::QueueUnavailable("installed blit destination lost External custody")
        })?;
        // Keep every existing diagnostic and writer dependency. No new lock is
        // added; existing map/diagnostic/Vec warm costs are NOT discharged here.
        edit.entered_effects = true;
        let source_ptr = source.raw_ptr() as usize;
        let destination_ptr = destination.raw_ptr() as usize;
        arena::note_bind(source_ptr);
        arena::note_bind(destination_ptr);
        late_bind::on_bind(source_ptr, false, "blit copy");
        late_bind::on_bind(destination_ptr, true, "blit copy");
        self.wait_for_last_writers(&[source_ptr, destination_ptr]);
        self.tracked_outputs.push(destination_ptr);
        // SAFETY: checked absolute ranges and macOS alignment; both installed
        // External pins survive exact native terminal + escaped emission/refs.
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source_lease.raw_unchecked(),
                    source_offset,
                    destination_lease.raw_unchecked(),
                    destination_offset,
                    size,
                );
        }
        edit.finish()
    }

    /// Selected legal aligned fill arm; unaligned ranges select their sealed
    /// u8 compute fill before obtaining this blit encoder. No adjacent-view
    /// rounding or old raw setter is used by this ownership-tracked entry.
    pub(crate) fn fill_handle_aligned(
        &mut self,
        destination: &super::BufferHandle,
        range: (usize, usize),
        value: u8,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        use crate::kernels::MetalKernelError as Error;
        let absolute = crate::native_binding_preflight::checked_byte_range(
            destination.base_offset(),
            destination.length(),
            range.0,
            range.1,
        )
        .ok_or(Error::QueueUnavailable(
            "blit fill exceeds originating view",
        ))?;
        if range.1 == 0 {
            return Ok(());
        }
        if !absolute.is_multiple_of(4) || !range.1.is_multiple_of(4) {
            return Err(Error::QueueUnavailable(
                "byte fill requires its prepared compute arm",
            ));
        }
        let owner = self.recording.as_ref().ok_or(Error::QueueUnavailable(
            "raw blit encoder requires unsafe interoperability",
        ))?;
        if self.ended {
            return Err(Error::QueueUnavailable("blit encoder already ended"));
        }
        let buffer = owner.command_buffer();
        let ticket = owner.begin_binding_edit()?;
        let mut edit = BindingEdit {
            buffer,
            ticket: Some(ticket),
            entered_effects: false,
        };
        let generation = buffer.binding_generation()?;
        let prepared = generation.prepare_buffer(destination)?;
        // SAFETY: SAME exact EDIT; whole checked range and counted mint precede
        // this single-member install. Full consumes no credit/native effects.
        let [installed] = unsafe { generation.install_prepared([prepared]) }?;
        let lease = installed.lease();
        let destination = lease.buffer_for_binding().map_err(|_| {
            generation.quarantine(crate::retirement_owners::Failure::AdapterFailed);
            Error::QueueUnavailable("installed fill destination lost External custody")
        })?;
        edit.entered_effects = true;
        let ptr = destination.raw_ptr() as usize;
        arena::note_bind(ptr);
        late_bind::on_bind(ptr, true, "blit fill");
        self.wait_for_last_writers(&[ptr]);
        self.tracked_outputs.push(ptr);
        // SAFETY: exact initialized view geometry, legal macOS range and SAME
        // installed External lease through terminal + emission/ref closure.
        self.raw.fillBuffer_range_value(
            unsafe { lease.raw_unchecked() },
            NSRange {
                location: absolute,
                length: range.1,
            },
            value,
        );
        edit.finish()
    }

    /// Copy bytes from src to dst, ordered after the last writer of *either*.
    ///
    /// The source wait is the obvious one: the copy reads it. The destination
    /// wait matters because a copy's destination is typically a buffer the pool
    /// has just recycled, which is where a pending writer is most likely -- and
    /// under `HazardTrackingModeUntracked` a missed dependency corrupts
    /// silently rather than failing (`DESIGN.md` §3.5). The sibling
    /// `fill_buffer` already waited on its destination; this did not, and the
    /// asymmetry was not deliberate.
    ///
    /// Measured on LFM2 decode, the destination has a registered writer in 0 of
    /// its calls, so this closes a hole rather than removing an observed bug.
    ///
    /// This does **not** fix the grouped-convolution corruption: that is the
    /// buffer pool aliasing two tensors onto one in-flight allocation, which no
    /// fence can observe -- at the aliasing instant no encoder has bound the
    /// buffer, and by the time one does it looks freshly allocated
    /// (`DESIGN.md` §2.3.8b). Measured at 11/30 unstable with and without.
    /// # Safety
    /// Explicit legacy/raw interoperability: caller retains genuine originating
    /// reuse authority and native/accounting owners through actual terminal work,
    /// cached native references and every escaped future emission. Bare derived
    /// Buffer/native pointer retention is NOT reuse authority. No tracked pin is
    /// installed by this method; use the admitted typed operation instead.
    pub unsafe fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        // Empty tensor copies are no-ops, not legal zero-byte Metal blits.
        // No byte is read or written, so there is no hazard to register.
        if size == 0 {
            return;
        }
        let src_ptr = src_buffer.raw_ptr() as usize;
        let dst_ptr = dst_buffer.raw_ptr() as usize;
        // A blit reads and writes buffers exactly as a dispatch binds them, so
        // an arena recording must see both: a value whose last reader is a
        // blit would otherwise end its interval early and be aliased while
        // the copy still reads it (`DESIGN.md` §9.2aa).
        arena::note_bind(src_ptr);
        arena::note_bind(dst_ptr);
        late_bind::on_bind(src_ptr, false, "blit copy");
        late_bind::on_bind(dst_ptr, true, "blit copy");
        self.wait_for_last_writers(&[src_ptr, dst_ptr]);

        self.tracked_outputs.push(dst_ptr);

        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.raw_unchecked(),
                    src_offset,
                    dst_buffer.raw_unchecked(),
                    dst_offset,
                    size,
                )
        }
    }

    /// # Safety
    /// Explicit legacy/raw interoperability: caller retains genuine originating
    /// reuse authority and native/accounting owners through actual terminal work,
    /// cached native references and every escaped future emission. Bare derived
    /// Buffer/native pointer retention is NOT reuse authority. No tracked pin is
    /// installed by this method; use the admitted typed operation instead.
    pub unsafe fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        let ptr = buffer.raw_ptr() as usize;
        arena::note_bind(ptr);
        late_bind::on_bind(ptr, true, "blit fill");
        self.wait_for_last_writers(&[ptr]);
        self.tracked_outputs.push(ptr);

        self.raw.fillBuffer_range_value(
            unsafe { buffer.raw_unchecked() },
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(ptr: usize, offset: usize, len: usize) -> BoundRange {
        BoundRange { ptr, offset, len }
    }

    /// The property route (a) rests on: two bindings collide exactly when they
    /// can touch the same byte (`DESIGN.md` §9.2e).
    ///
    /// Adjacency is the case worth pinning. `[0, 128)` and `[128, 256)` share no
    /// byte, so they must not collide -- which is precisely the pair an arena
    /// produces, since slots are laid out end to end at 128 B alignment. Getting
    /// the comparison non-strict would make every neighbouring slot conflict and
    /// silently give back the whole win.
    /// What `auto_barrier`'s no-barrier branch costs per dispatch, against the
    /// REAL `BoundSet` rather than a model of it (lloom #727).
    ///
    /// The branch does `next_*.take()` then `prev_*.absorb(&mut next)`, once
    /// per dispatch. `absorb` drains both maps and re-appends every range into
    /// `prev_*`, so its cost grows with what `prev_*` already holds — and
    /// under a pass-plan or arena base EVERY activation shares one pointer, so
    /// a single bucket accumulates every range and `insert`'s `contains` scans
    /// it linearly. That is the shape this measures: 338 dispatches (an LFM2
    /// decode step) x 6 binds, at 1 / 6 / 60 distinct pointers.
    ///
    /// Not an assertion of a time — a timing test would be flaky and would
    /// pin the machine, not the code. It asserts the STRUCTURAL property,
    /// which is stable: a range re-bound every dispatch is retained once, and
    /// distinct ranges are all retained.
    ///
    /// The 338 here is a decode step's dispatch count, NOT one encoder's: the
    /// cut period bounds an encoder at ~50 dispatches on the shipping config,
    /// which is why the fix measured null there (see `absorb`).
    #[test]
    fn absorb_accumulates_every_range_into_one_bucket_under_a_shared_base() {
        let dispatches = 338usize;
        let binds = 6usize;
        let mut prev = BoundSet::default();
        for d in 0..dispatches {
            let mut next = BoundSet::default();
            for b in 0..binds {
                // One base pointer, distinct 128 B slots — the arena's layout.
                next.insert(HazardKey::Range, r(1, (d * binds + b) * 128, 128));
            }
            let mut taken = next.take();
            prev.absorb(&mut taken);
        }
        // One bucket, and it holds each DISTINCT range once — `absorb`
        // deduplicates as `insert` does. It used to `append` blindly, so this
        // bucket held `dispatches * binds` entries and every later `contains`
        // and `conflicts` scanned all of them (lloom #727).
        let bucket = prev.ranges.get(&1).expect("the shared base's bucket");
        assert_eq!(prev.ranges.len(), 1, "one shared base, one bucket");
        assert_eq!(
            bucket.len(),
            dispatches * binds,
            "these ranges ARE all distinct (disjoint 128 B slots), so none may \
             be dropped: dedup must not lose a range"
        );

        // The property the fix buys, on the population that actually recurs:
        // the SAME range re-bound every dispatch — a weight, a KV slab, the
        // arena's own base — is kept once, not once per dispatch.
        let mut prev = BoundSet::default();
        for _ in 0..dispatches {
            let mut next = BoundSet::default();
            next.insert(HazardKey::Range, r(7, 0, 4096));
            let mut taken = next.take();
            prev.absorb(&mut taken);
        }
        assert_eq!(
            prev.ranges.get(&7).map(Vec::len),
            Some(1),
            "a range re-bound every dispatch must be retained ONCE; \
             appending blindly kept {dispatches} copies and scanned them all"
        );
    }

    /// The per-dispatch rotation is ALLOCATION-FREE in the steady state.
    ///
    /// `set_input_buffer`/`set_output_buffer` run ~2000 times per token, and
    /// each one `insert`s into a `BoundSet`. If the rotation handed the map's
    /// allocation away every dispatch (`mem::take`) the next dispatch would
    /// regrow every bucket — a `HashMap` entry and a `Vec` push per bind,
    /// 2000 times a token, for a population that is the SAME pointers every
    /// dispatch.
    ///
    /// This pins the fix as a property rather than as a timing: after a warm
    /// first dispatch, the buckets' capacity stops growing. It is asserted at
    /// the structure level because a timing assertion here would pin the
    /// machine, not the code — and because the end-to-end effect is a
    /// measured null on the shipping config (lloom #727, measurements §25:
    /// a +2.1 ms/step encode swing moves throughput 0.0, since the GPU step
    /// dominates). The cost is real, bounded, and worth not paying; the
    /// throughput claim is not made.
    #[test]
    fn the_per_dispatch_rotation_stops_allocating_once_warm() {
        let mut prev = BoundSet::default();
        let mut next = BoundSet::default();
        // The steady-state population: a handful of pointers, re-bound every
        // dispatch at the same ranges — weights, the KV slab, an arena base.
        let binds = [
            r(1, 0, 4096),
            r(2, 0, 8192),
            r(3, 128, 128),
            r(1, 4096, 4096),
        ];

        let mut capacity_after = Vec::new();
        for _ in 0..8 {
            for b in binds {
                next.insert(HazardKey::Range, b);
            }
            prev.absorb(&mut next);
            next.clear();
            capacity_after.push((prev.bucket_capacity(), next.bucket_capacity()));
        }

        // Warm after the first pass: neither set grows again.
        let steady = capacity_after[1];
        for (i, c) in capacity_after.iter().enumerate().skip(1) {
            assert_eq!(
                *c, steady,
                "dispatch {i} changed the retained capacity {capacity_after:?} — the rotation \
                 is reallocating per dispatch"
            );
        }
        // And `prev` holds each distinct range once, not once per dispatch.
        assert_eq!(prev.ranges.get(&1).map(Vec::len), Some(2));
        assert_eq!(prev.ranges.get(&2).map(Vec::len), Some(1));
        assert_eq!(prev.ranges.get(&3).map(Vec::len), Some(1));
    }

    #[test]
    fn ranges_overlap_exactly_when_they_share_a_byte() {
        assert!(
            !r(1, 0, 128).overlaps(&r(1, 128, 128)),
            "adjacent ranges collided"
        );
        assert!(
            !r(1, 128, 128).overlaps(&r(1, 0, 128)),
            "adjacency is not symmetric"
        );
        assert!(
            r(1, 0, 129).overlaps(&r(1, 128, 128)),
            "a one-byte overlap was missed"
        );
        assert!(
            r(1, 0, 256).overlaps(&r(1, 64, 64)),
            "containment was missed"
        );
        assert!(
            r(1, 64, 64).overlaps(&r(1, 0, 256)),
            "containment is not symmetric"
        );
        assert!(
            r(1, 0, 128).overlaps(&r(1, 0, 128)),
            "a range did not overlap itself"
        );
        // Different allocations never collide, whatever their offsets say --
        // offsets are only comparable within one buffer.
        assert!(
            !r(1, 0, 128).overlaps(&r(2, 0, 128)),
            "distinct buffers collided"
        );
    }

    /// An unknown extent must fail toward ordering.
    ///
    /// A spurious barrier costs throughput; a missing one is silent corruption
    /// under `HazardTrackingModeUntracked` (§3.5). So a zero length covers the
    /// whole allocation rather than nothing -- the asymmetry is deliberate and
    /// is the one place this logic is allowed to be imprecise.
    #[test]
    fn an_unknown_extent_orders_conservatively() {
        assert!(
            r(1, 0, 0).overlaps(&r(1, 4096, 128)),
            "a zero length did not order"
        );
        assert!(
            r(1, 4096, 128).overlaps(&r(1, 0, 0)),
            "zero-length is not symmetric"
        );
        assert!(
            !r(1, 0, 0).overlaps(&r(2, 0, 0)),
            "zero length crossed buffers"
        );
    }

    /// A barrier is a latch, so its attribution is a **set** of directions.
    ///
    /// Asserted rather than assumed because the single-value alternative is
    /// tempting and silently lossy: it would report whichever binding happened
    /// to be last before the dispatch, which is an arbitrary choice wearing the
    /// clothes of an attribution (issue #185).
    #[test]
    fn a_barrier_can_be_owed_to_more_than_one_direction() {
        let mut k = HazardKinds::NONE;
        assert!(k.is_empty(), "a fresh set was not empty");
        assert_eq!(k.iter().count(), 0);

        k.insert(HazardKind::Waw);
        k.insert(HazardKind::War);
        assert!(k.contains(HazardKind::Waw) && k.contains(HazardKind::War));
        assert!(
            !k.contains(HazardKind::Raw),
            "a direction nothing recorded was reported"
        );
        // Order is RAW, WAW, WAR regardless of insertion order, so a report is
        // stable rather than dependent on which binding fired first.
        assert_eq!(
            k.iter().map(HazardKind::as_str).collect::<Vec<_>>(),
            ["waw", "war"]
        );

        // Idempotent: two WAW hazards before one barrier are one direction, not
        // two. The set counts *which* directions, never how many bindings.
        let mut twice = HazardKinds::NONE;
        twice.insert(HazardKind::Raw);
        twice.insert(HazardKind::Raw);
        assert_eq!(twice.iter().count(), 1);

        // `take` drains, so a barrier cannot inherit the previous one's kinds.
        let drained = k.take();
        assert!(k.is_empty(), "take did not drain");
        assert_eq!(drained.iter().count(), 2, "take did not return the set");
    }

    /// **The write path must evaluate both hazard tests, not short-circuit.**
    ///
    /// `set_output_buffer` was `if prev_outputs.conflicts(..) ||
    /// prev_inputs.conflicts(..)`, which is correct for a `bool` and wrong for
    /// an attribution: at any position where WAW fires, `||` stops and WAR is
    /// never tested, so WAR would be under-reported exactly where the two
    /// coincide. This pins the case, since it is the one a future
    /// simplification back to `||` would silently reintroduce.
    ///
    /// **Calls `write_hazards`, the function the encoder calls.** Asserting over
    /// a re-implementation would validate the intent instead of the artifact --
    /// §8.1d's recorded failure, where a script reproduced the intended
    /// lowercase name rather than asking the preprocessor and reported all 90
    /// variants matching while 48 were absent from the metallib. Verified by
    /// mutation: restoring the `||` short-circuit in production source turns
    /// this red.
    #[test]
    fn a_write_can_be_both_waw_and_war_and_both_must_be_seen() {
        let slot = r(9, 0, 128);

        // One earlier dispatch wrote the slot and another read it, so a later
        // write to the same bytes is WAW *and* WAR at once.
        let mut prev_outputs = BoundSet::default();
        prev_outputs.insert(HazardKey::Range, slot);
        let mut prev_inputs = BoundSet::default();
        prev_inputs.insert(HazardKey::Range, slot);

        let kinds = write_hazards(&prev_outputs, &prev_inputs, HazardKey::Range, &slot);
        assert_eq!(
            kinds.iter().map(HazardKind::as_str).collect::<Vec<_>>(),
            ["waw", "war"],
            "a write that is both WAW and WAR reported only one direction"
        );

        // Each direction alone, so the test cannot pass by always reporting both
        // -- the mutation a "report everything" simplification would introduce.
        let only_waw = write_hazards(&prev_outputs, &BoundSet::default(), HazardKey::Range, &slot);
        assert_eq!(
            only_waw.iter().map(HazardKind::as_str).collect::<Vec<_>>(),
            ["waw"],
            "a pure WAW reported a direction nothing supports"
        );
        let only_war = write_hazards(&BoundSet::default(), &prev_inputs, HazardKey::Range, &slot);
        assert_eq!(
            only_war.iter().map(HazardKind::as_str).collect::<Vec<_>>(),
            ["war"],
            "a pure WAR reported a direction nothing supports"
        );
        // And a write to bytes nobody touched orders nothing.
        assert!(
            write_hazards(
                &prev_outputs,
                &prev_inputs,
                HazardKey::Range,
                &r(9, 4096, 128)
            )
            .is_empty(),
            "a disjoint write manufactured a hazard"
        );
    }

    /// Read-after-read is not a hazard, and that is why the weights are free.
    ///
    /// 5.394 GB is bound on every dispatch and read every time; if RAR ordered,
    /// every dispatch would conflict with every other and the barrier count
    /// would be the dispatch count. `read_hazards` therefore consults
    /// `prev_outputs` **only**, and this pins that asymmetry against a
    /// symmetry-restoring "cleanup" (issue #185).
    #[test]
    fn two_reads_never_conflict() {
        let weight = r(11, 0, 4096);

        // The whole of `prev_inputs` is reads. Whatever it holds, a read against
        // it is RAR and must order nothing -- so `read_hazards` does not take it
        // and cannot be made to consult it by accident.
        let mut prev_outputs = BoundSet::default();
        assert!(
            read_hazards(&prev_outputs, HazardKey::Range, &weight).is_empty(),
            "a read conflicted against an empty write set"
        );

        prev_outputs.insert(HazardKey::Range, r(11, 8192, 128));
        assert!(
            read_hazards(&prev_outputs, HazardKey::Range, &weight).is_empty(),
            "a read conflicted with a write to disjoint bytes"
        );

        // ... but a read of bytes something did write is RAW, and only RAW.
        prev_outputs.insert(HazardKey::Range, weight);
        assert_eq!(
            read_hazards(&prev_outputs, HazardKey::Range, &weight)
                .iter()
                .map(HazardKind::as_str)
                .collect::<Vec<_>>(),
            ["raw"],
            "a genuine read-after-write was missed, or mis-attributed"
        );
    }

    /// `Range` can only ever remove barriers relative to `Pointer`, never add
    /// one. That is what makes it safe to adopt without re-verifying every
    /// dependency: an edge it drops is one where the bindings provably touch
    /// disjoint bytes, and Metal's barrier was resource-granular anyway.
    #[test]
    fn range_keying_is_a_refinement_of_pointer_keying() {
        let a = r(7, 0, 128);
        let b = r(7, 128, 128);

        let mut ptr_set = BoundSet::default();
        ptr_set.insert(HazardKey::Pointer, a);
        assert!(
            ptr_set.conflicts(HazardKey::Pointer, &b),
            "pointer keying stopped ordering two values in one allocation"
        );

        let mut range_set = BoundSet::default();
        range_set.insert(HazardKey::Range, a);
        assert!(
            !range_set.conflicts(HazardKey::Range, &b),
            "range keying ordered two disjoint slots"
        );
        // ... but it still orders anything that genuinely overlaps.
        assert!(
            range_set.conflicts(HazardKey::Range, &r(7, 64, 128)),
            "range keying dropped a real dependency"
        );
    }

    /// The keying is read per encoder session, so switching it is only visible
    /// to sessions opened afterwards. Stated as a test because the alternative
    /// -- switching mid-session -- would compare bindings recorded under one
    /// rule against a lookup under the other.
    #[test]
    fn a_sessions_keying_is_fixed_when_it_opens() {
        let s = EncoderState::with_hazard_key(HazardKey::Range);
        assert_eq!(s.hazard_key, HazardKey::Range);
        let s = EncoderState::with_hazard_key(HazardKey::Pointer);
        assert_eq!(s.hazard_key, HazardKey::Pointer);
        // The default is what shipped, so an unconfigured process is unchanged.
        assert_eq!(EncoderState::new().hazard_key, HazardKey::Pointer);
        assert_eq!(HazardKey::default(), HazardKey::Pointer);
    }

    /// Every set this state carries is READ somewhere (issue #579).
    ///
    /// `all_inputs` was written at 1123 bindings per decode token and read
    /// nowhere -- §6.7 **L2**'s canonical instance -- and it is now gone. This
    /// is what stops it, or a successor, coming back silently: a set on the
    /// per-bind path is 27 ns a token pays 1123 times, and nothing about
    /// adding one announces that it has no reader.
    ///
    /// **A source-level census, because the property is the absence of a
    /// read** -- a runtime test cannot observe a field nobody consults, and
    /// the value it holds is correct either way. Needles are assembled and
    /// comment lines stripped, both of §9.2p's repairs, for its measured
    /// reason.
    #[test]
    fn every_recorded_set_on_the_bind_path_has_a_reader() {
        fn code_only(src: &str) -> String {
            src.lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !(t.starts_with("//") || t.starts_with("/*") || t.starts_with('*'))
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        let encoder = code_only(include_str!("encoder.rs"));
        let commands = code_only(include_str!("commands.rs"));

        let gone = ["all_", "inputs"].concat();
        assert!(
            !encoder.contains(&gone),
            "a write-only per-bind set is back. It cost 1123 inserts and \
             0.0303 ms per decode token and had no reader (issue #579, \
             DESIGN.md §6.7 L2). If it has a consumer now, name the consumer \
             here and delete this assertion"
        );

        // The control, and it is the half that makes the assertion mean
        // something: the SIBLING set is still written AND still read, so the
        // needle above is not passing because the whole mechanism vanished.
        let kept = ["all_", "outputs"].concat();
        assert!(
            encoder.contains(&kept),
            "the sibling set is gone too -- this census is now vacuous"
        );
        assert!(
            commands.contains(&kept),
            "the sibling set has lost its reader in end_encoding, which is the \
             only thing that distinguished it from the removed one"
        );
    }
}

#[cfg(test)]
mod external_handle_tests {
    #[test]
    fn older_recording_keeps_domain_open_after_current_handle_releases() {
        let wake = Arc::new(crate::completion_wake::Wake::default());
        let domain = EncoderDomain::new(wake.clone());
        let older = shared();
        let current = shared();
        for state in [&older, &current] {
            assert!(state.domain.set(domain.clone()).is_ok());
            assert!(state.wake.set(wake.clone()).is_ok());
        }
        let old_handle = ExternalEncoderHandle::acquire(&older);
        let current_handle = ExternalEncoderHandle::acquire(&current);
        assert_eq!(domain.external.load(Ordering::Acquire), 2);
        drop(current_handle);
        assert_eq!(current.external.load(Ordering::Acquire), 0);
        assert!(
            domain.has_external_handles(),
            "older shared-map holder was ignored"
        );
        let seen = wake.snapshot();
        let refused_clone = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _rollback = ExternalEncoderHandle::acquire(&older);
            assert_eq!(domain.external.load(Ordering::Acquire), 2);
            panic!("later clone field refused");
        }));
        assert!(refused_clone.is_err());
        assert_eq!(domain.external.load(Ordering::Acquire), 1);
        assert_eq!(
            wake.snapshot(),
            seen,
            "rollback released another live domain handle"
        );
        drop(old_handle);
        assert!(!domain.has_external_handles());
        assert_ne!(
            wake.snapshot(),
            seen,
            "aggregate last release did not notify"
        );
    }

    use super::*;

    fn shared() -> Arc<SharedEncoderState> {
        Arc::new(SharedEncoderState {
            state: parking_lot::Mutex::new(EncoderState::new()),
            external: std::sync::atomic::AtomicUsize::new(0),
            domain: std::sync::OnceLock::new(),
            wake: std::sync::OnceLock::new(),
        })
    }

    #[test]
    fn last_external_release_notifies_after_nonfinal_releases() {
        let state = shared();
        let wake = Arc::new(crate::completion_wake::Wake::default());
        assert!(state.wake.set(wake.clone()).is_ok());
        let seen = wake.snapshot();
        let first = ExternalEncoderHandle::acquire(&state);
        let second = ExternalEncoderHandle::acquire(&state);
        assert_eq!(state.external.load(Ordering::Acquire), 2);
        assert_eq!(Arc::strong_count(&state), 3);
        drop(first);
        assert_eq!(state.external.load(Ordering::Acquire), 1);
        assert_eq!(wake.snapshot(), seen, "nonfinal release invented readiness");
        drop(second);
        assert_eq!(state.external.load(Ordering::Acquire), 0);
        assert_ne!(wake.snapshot(), seen, "last release failed to notify");
        assert_eq!(Arc::strong_count(&state), 1);
    }

    #[test]
    fn partial_clone_unwind_rolls_back_without_releasing_live_handle() {
        let state = shared();
        let first = ExternalEncoderHandle::acquire(&state);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _rollback = ExternalEncoderHandle::acquire(&state);
            assert_eq!(state.external.load(Ordering::Acquire), 2);
            panic!("later cloned field refused");
        }));
        assert!(result.is_err());
        assert_eq!(state.external.load(Ordering::Acquire), 1);
        assert_eq!(Arc::strong_count(&state), 2);
        drop(first);
        assert_eq!(state.external.load(Ordering::Acquire), 0);
    }

    #[test]
    fn field_custody_precedes_last_permit_release() {
        struct Capture {
            state: Arc<SharedEncoderState>,
            drops: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl Drop for Capture {
            fn drop(&mut self) {
                assert_eq!(
                    self.state.external.load(Ordering::Acquire),
                    1,
                    "readiness was published before preceding capture destruction"
                );
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }
        struct Fixture {
            _capture: Capture,
            _permit: ExternalEncoderHandle,
        }
        let state = shared();
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fixture = Fixture {
            _capture: Capture {
                state: state.clone(),
                drops: drops.clone(),
            },
            _permit: ExternalEncoderHandle::acquire(&state),
        };
        drop(fixture);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(state.external.load(Ordering::Acquire), 0);
    }
}
