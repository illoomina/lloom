use crate::kernels::metal::{
    arena,
    executor::{DispatchAction, DispatchRecord, ExecutorSlot, Grid},
    trace, Buffer, ComputePipeline, Fence,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLResourceUsage, MTLSize,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
};

/// Shared cross-encoder output map: maps buffer pointer -> fence of the last encoder that wrote it.
/// Used by subsequent encoders to call waitForFence before reading those buffers.
pub type PrevCeOutputs = Arc<Mutex<HashMap<usize, Arc<Fence>>>>;

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
        if let Ok(v) = std::env::var("CANDLE_METAL_HAZARD_KEY") {
            match v.as_str() {
                "range" => HAZARD_KEY.store(true, Ordering::Relaxed),
                "pointer" | "ptr" | "" => {}
                other => {
                    // Loudly, because the silent alternative is measuring the
                    // default while believing the variable was honoured.
                    panic!("CANDLE_METAL_HAZARD_KEY={other:?}; want range or pointer")
                }
            }
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

    fn absorb(&mut self, other: &mut BoundSet) {
        self.ptrs.extend(other.ptrs.drain());
        for (ptr, mut v) in other.ranges.drain() {
            self.ranges.entry(ptr).or_default().append(&mut v);
        }
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
    /// Name of the pipeline most recently set, so a dispatch can be attributed
    /// to a kernel. Only maintained when an executor is installed or profiling
    /// is on -- Metal has no way to read the bound pipeline back, and doing so
    /// per dispatch on the classical path would be cost for nobody.
    pub current_pipeline: Option<Arc<str>>,
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
            hazard_key,
            prev_outputs: BoundSet::default(),
            next_outputs: BoundSet::default(),
            prev_inputs: BoundSet::default(),
            next_inputs: BoundSet::default(),
            needs_barrier: false,
            pending_kinds: HazardKinds::NONE,
            all_outputs: HashSet::new(),
            waited_fences: HashSet::new(),
            current_pipeline: None,
        }
    }
}

#[derive(Clone)]
pub struct ComputeCommandEncoder {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Retained so we can register completion handlers on this CB.
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Per-encoder-session fence. Updated at end_encoding.
    pub(crate) fence: Arc<Fence>,
    /// Hazard tracking state. Arc shared between the canonical encoder in EntryState
    /// and the clone held by CommandsGuard. Uncontended in practice (CommandsGuard
    /// holds the outer Commands mutex for the entire kernel dispatch).
    pub(crate) state: Arc<Mutex<EncoderState>>,
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
    pub(crate) param_capture: Arc<Mutex<ParamCapture>>,
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

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> ComputeCommandEncoder {
        Self::with_executor(
            raw,
            command_buffer,
            fence,
            prev_ce_outputs,
            Arc::new(ExecutorSlot::Classical),
        )
    }

    /// As [`Self::new`], but submitting dispatches through `executor`.
    pub fn with_executor(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
        executor: Arc<ExecutorSlot>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            command_buffer,
            fence,
            state: Arc::new(Mutex::new(EncoderState::with_hazard_key(hazard_key()))),
            prev_ce_outputs,
            executor,
            capturing: Arc::new(AtomicBool::new(false)),
            param_capture: Arc::new(Mutex::new(ParamCapture::default())),
        }
    }

    /// Wait on the fence of `ptr`'s last writer, if any, and only once.
    ///
    /// This replaces waiting on every live fence at encoder creation. The
    /// encoder already records every buffer it binds, so the wait can be
    /// limited to buffers this encoder actually touches.
    fn wait_for_buffer(&self, ptr: usize) {
        let fence = {
            let map = self.prev_ce_outputs.lock().unwrap();
            map.get(&ptr).cloned()
        };
        let Some(fence) = fence else { return };

        let mut state = self.state.lock().unwrap();
        if state.waited_fences.insert(Arc::as_ptr(&fence) as usize) {
            drop(state);
            self.raw.waitForFence(fence.raw());
        }
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.auto_barrier();
        trace::record_dispatch(
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
        let kernel = self.state.lock().unwrap().current_pipeline.clone();
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
        }
    }

    /// Run `length` pre-encoded ICB commands from `location`, here.
    ///
    /// Called at the position where a run of replayed dispatches begins, so the
    /// commands land between the same two classically-encoded neighbours the
    /// originals had. `metal::icb::Run` carries why that placement is the whole
    /// of the correctness argument.
    fn execute_icb_range(
        &self,
        range: &crate::kernels::metal::executor::IcbRange,
        location: usize,
        length: usize,
    ) {
        // SAFETY: every resource an encoded command binds is in `resident` --
        // the plan collects them as it encodes, so the set cannot fall behind
        // the commands -- and residency has to be declared on *this* encoder,
        // which is why it is done at every execute rather than once at install.
        // Omitting it is silent corruption rather than an error (`DESIGN.md`
        // §3.7), and unified memory can mask the omission.
        unsafe {
            for buf in &range.resident {
                self.raw.useResource_usage(
                    ProtocolObject::from_ref(buf.as_ref()),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            self.raw.executeCommandsInBuffer_withRange(
                &range.icb,
                objc2_foundation::NSRange { location, length },
            );
        }
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
        let name = {
            let s = self.state.lock().unwrap();
            s.current_pipeline.clone()
        };
        if profiling {
            crate::kernels::metal::profile::record_dispatch(name.as_ref());
        }
        crate::kernels::metal::hazard_audit::record_dispatch(
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

        let mut s = self.state.lock().unwrap();
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
                crate::kernels::metal::hazard_audit::record_barrier_suppressed();
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
            trace::record_barrier_kinds(kinds);
            trace::record_barrier();
            crate::kernels::metal::hazard_audit::record_barrier(kinds);
            s.needs_barrier = false;
            // Replaced, not extended: everything bound before the barrier is
            // now ordered against everything after it.
            s.prev_outputs = s.next_outputs.take();
            s.prev_inputs = s.next_inputs.take();
        } else {
            let mut next_out = s.next_outputs.take();
            s.prev_outputs.absorb(&mut next_out);
            let mut next_in = s.next_inputs.take();
            s.prev_inputs.absorb(&mut next_in);
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

    pub fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        let index = self.capture_buffer_index(index);
        // An arena slot is a region of a shared allocation, so the byte the
        // kernel must start at is the slot's base plus the layout offset the
        // caller computed. `base_offset` is 0 for every ordinary allocation,
        // which is why this is a no-op everywhere except the arena -- see
        // `Buffer::base_offset` for why the addition belongs here and not at
        // the call sites.
        let offset = offset + buffer.map_or(0, Buffer::base_offset);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(index, ptr, offset, buf.length(), false);
            // The clock a liveness recording must use: a value is live until the
            // last dispatch that binds it (`DESIGN.md` §6.7 L4).
            arena::note_bind(ptr);
            // Read-after-write against an earlier encoder: order against that
            // buffer's last writer only.
            self.wait_for_buffer(ptr);
            let range = Self::bound_range(buf, offset);
            let mut s = self.state.lock().unwrap();
            let key = s.hazard_key;
            // Read-after-write within this encoder. See `read_hazards` for why a
            // read has exactly one possible direction (issue #185).
            let kinds = read_hazards(&s.prev_outputs, key, &range);
            if !kinds.is_empty() {
                s.needs_barrier = true;
                s.pending_kinds.merge(kinds);
            }
            s.next_inputs.insert(key, range);
            crate::kernels::metal::hazard_audit::record_binding(&range, false);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, false);
            }
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        let index = self.capture_buffer_index(index);
        // See `set_input_buffer`: the arena's `base + offset`, added once at the
        // choke point every binding passes through.
        let offset = offset + buffer.map_or(0, Buffer::base_offset);
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            trace::record_binding(index, ptr, offset, buf.length(), true);
            arena::note_bind(ptr);
            // Write-after-write or write-after-read against an earlier encoder.
            self.wait_for_buffer(ptr);
            let range = Self::bound_range(buf, offset);
            let mut s = self.state.lock().unwrap();
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
            crate::kernels::metal::hazard_audit::record_binding(&range, true);
            s.all_outputs.insert(ptr);
            drop(s);
            if !self.executor.is_classical() {
                self.executor.will_bind_buffer(index, buf, offset, true);
            }
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
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
        let mut cap = self.param_capture.lock().unwrap();
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
    pub(crate) fn capture_array(&self, len_bytes: usize, bytes: *const c_void) -> bool {
        if !self.capturing.load(Ordering::Relaxed) {
            return false;
        }
        let index = self.capture_buffer_index(usize::MAX);
        let device = crate::kernels::metal::Device::new(self.command_buffer.device());
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
        self.wait_for_buffer(ptr);
        {
            let range = Self::bound_range(&buffer, 0);
            let mut s = self.state.lock().unwrap();
            let key = s.hazard_key;
            s.next_inputs.insert(key, range);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.as_ref()), 0, index)
        }
        // The staging buffer must stay alive until the dispatch it feeds has
        // completed, not merely until it is encoded. It is parked here and
        // handed to the caller by `end_param_capture`, which holds it across
        // the dispatch -- releasing it at capture-close would drop it while the
        // GPU may still be reading, which is the in-flight-reuse failure
        // `DESIGN.md` §2.3.8b describes and no fence can see.
        self.param_capture.lock().unwrap().staged.push(buffer);
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
        let mut cap = self.param_capture.lock().unwrap();
        let slot = cap.next_buffer;
        cap.next_buffer += 1;
        slot
    }

    /// Begin capturing scalars into a packed-params block.
    ///
    /// Scoped rather than persistent: [`Self::end_param_capture`] returns the
    /// bytes and closes it, so a capture cannot leak into the next dispatch.
    pub fn begin_param_capture(&self) {
        let mut cap = self.param_capture.lock().unwrap();
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
        let mut cap = self.param_capture.lock().unwrap();
        let mut bytes = std::mem::take(&mut cap.bytes);
        let staged = std::mem::take(&mut cap.staged);
        while align != 0 && !bytes.len().is_multiple_of(align) {
            bytes.push(0);
        }
        (bytes, staged)
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        trace::record_pipeline(pipeline.name().unwrap_or("<unnamed>"));
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
        if !has_executor && !crate::kernels::metal::profile::enabled() {
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
        self.state.lock().unwrap().current_pipeline = name;
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

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.updateFence(self.fence.raw());
        self.raw.endEncoding();
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        trace::record_pipeline(pipeline.name().unwrap_or("<unnamed>"));
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

pub struct BlitCommandEncoder {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
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
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        fence: Arc<Fence>,
        prev_ce_outputs: PrevCeOutputs,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder {
            raw,
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

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;

        // Signal this blit encoder's fence after all blit commands complete
        self.update_fence(&self.fence);
        self.raw.endEncoding();

        // Register outputs so subsequent encoders can wait.
        {
            let mut map = self.prev_ce_outputs.lock().unwrap();
            for &out in &self.tracked_outputs {
                map.insert(out, Arc::clone(&self.fence));
            }
        }
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
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
        use objc2_metal::MTLBlitCommandEncoder as _;

        let fences: Vec<Arc<Fence>> = {
            let map = self.prev_ce_outputs.lock().unwrap();
            let mut out: Vec<Arc<Fence>> = Vec::new();
            for ptr in ptrs {
                if let Some(f) = map.get(ptr) {
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
    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        let src_ptr = src_buffer.raw_ptr() as usize;
        let dst_ptr = dst_buffer.raw_ptr() as usize;
        self.wait_for_last_writers(&[src_ptr, dst_ptr]);

        self.tracked_outputs.push(dst_ptr);

        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        let ptr = buffer.raw_ptr() as usize;
        self.wait_for_last_writers(&[ptr]);
        self.tracked_outputs.push(ptr);

        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
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
