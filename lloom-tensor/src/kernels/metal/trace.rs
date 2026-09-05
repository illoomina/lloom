//! Per-dispatch recording of the Metal command stream.
//!
//! Answers "is the decode dispatch sequence identical across tokens?" — the
//! question that gates whether an `MTLIndirectCommandBuffer` can be encoded once
//! and replayed, since an ICB is only replayable if the command sequence is
//! identical between invocations.
//!
//! Off unless `CANDLE_METAL_TRACE=1`, checked once into a `OnceLock`. When off
//! the cost is one relaxed atomic load per dispatch and nothing else: no
//! allocation, no lock, no formatting. That matters because the trace has to be
//! able to run in a release build against the real model — an instrumented
//! debug build would measure a different dispatch pattern than the one shipped.
//!
//! # Buffer identity
//!
//! Raw `MTLBuffer` pointers are recorded through a stable interning table rather
//! than verbatim. Addresses are meaningless across processes and would make
//! every run's trace differ for no reason; the interned id is assigned in order
//! of first use, so two runs that allocate the same logical buffers in the same
//! order produce identical traces. That is exactly the distinction the ICB
//! question needs — "a different buffer" has to be separable from "the same
//! buffer at a different address".
//!
//! The raw address is emitted alongside the id, since the allocator question in
//! `DESIGN.md` §9 wants to know whether a logical slot is backed by a stable
//! allocation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One buffer binding recorded against the dispatch that consumed it.
#[derive(Clone, PartialEq, Eq)]
pub struct Binding {
    /// Argument-table index the buffer was bound at.
    pub index: usize,
    /// Interned buffer identity, stable within a process.
    pub buffer_id: u64,
    /// Raw `MTLBuffer` address, for the allocator-stability question.
    pub buffer_addr: usize,
    /// Byte offset into the buffer.
    pub offset: usize,
    /// Bytes the handle addresses from `offset` -- an arena view's slot rather
    /// than the whole arena.
    ///
    /// **Added by issue #185, and its absence was a recorded wall.** §9.2j
    /// finding 3 hit it (*"`record_binding` carries `ptr`, `offset` and
    /// `is_output` but no length, and an interval test needs the extent on both
    /// sides"*) and #144's `edge-cover.py` documents the consequence it had to
    /// accept: keyed on `(buffer, offset)`, equality is *sufficient* for overlap
    /// and not *necessary*, so two bindings at different offsets that genuinely
    /// overlap are missed and the model **under-reports edges**. That is why
    /// that script reads its verdict only against mutation controls and never as
    /// a proof of safety.
    ///
    /// With the extent recorded the offline test can be the same interval test
    /// `BoundRange::overlaps` runs, so the audit stops resting on absence of
    /// evidence.
    ///
    /// Zero means an unknown extent, and consumers must treat it as covering the
    /// whole allocation -- failing toward ordering, exactly as
    /// `BoundRange::overlaps` does, since a spurious edge costs a false positive
    /// in a report and a missed one costs silent corruption (§3.5).
    pub len: usize,
    /// Whether the binding was made as an input or an output.
    pub is_output: bool,
}

/// A single `dispatchThreads`/`dispatchThreadgroups` call and its bound state.
#[derive(Clone)]
pub struct Dispatch {
    /// Monotonic index across the whole trace.
    pub seq: u64,
    /// Kernel name from the pipeline that was current at dispatch time.
    pub pipeline: String,
    /// `(width, height, depth)` of the grid, in the unit the dispatch used.
    pub grid: (usize, usize, usize),
    /// `(width, height, depth)` threads per threadgroup.
    pub threadgroup: (usize, usize, usize),
    /// True when dispatched as threadgroups rather than threads. The two are
    /// different units, so a trace that conflated them could report a false
    /// difference between two identical dispatches.
    pub by_threadgroups: bool,
    /// Buffers bound since the previous dispatch, in binding order.
    pub bindings: Vec<Binding>,
    /// Whether `auto_barrier` emitted a `memoryBarrierWithScope(Buffers)`
    /// immediately before this dispatch.
    ///
    /// Recorded rather than reconstructed. `DESIGN.md` §9.2e requires arena work
    /// to report barriers per token, and the count can only be *simulated* from
    /// a trace if the simulation also knows where each encoder began -- candle
    /// starts a fresh `EncoderState` every `CANDLE_METAL_COMPUTE_PER_BUFFER`
    /// dispatches (50 by default), and a simulation that misses those
    /// boundaries accumulates `prev_outputs` for the whole token and
    /// over-counts. Observing the barrier directly removes the modelling
    /// question entirely.
    pub barrier: bool,
    /// Which hazard directions that barrier was for (issue #185).
    ///
    /// Empty when `barrier` is false. **This is the field that makes a barrier
    /// count attributable**: §11.3p had to attribute the 505 by *position* --
    /// covered non-head, run head, gap -- because the kind was computed at the
    /// two emission sites and discarded into a `bool`. *"These N are WAR, and WAR
    /// is the one a different layout would remove"* is not a sentence any prior
    /// artifact could produce.
    ///
    /// A set rather than one value: a barrier is a latch and any number of
    /// bindings may set it before it fires. See `HazardKinds`.
    pub kinds: crate::kernels::metal::encoder::HazardKinds,
    /// Which encoder session this dispatch belongs to, counted from the start of
    /// the trace.
    ///
    /// Hazard state is per encoder, so this is what makes a barrier count
    /// interpretable: two dispatches in different sessions cannot be ordered by
    /// a barrier at all, only by a fence.
    pub encoder: u64,
    /// Caller-supplied marker this dispatch falls under, e.g. a decode step.
    pub region: Option<String>,
}

impl Dispatch {
    /// The part of a dispatch that must match for an ICB slot to be replayable:
    /// same kernel, same grid, same bindings. Excludes `seq` and `region`, which
    /// are position labels rather than command content.
    fn signature_parts(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(format!(
            "{} grid={:?} tg={:?} tgmode={}",
            self.pipeline, self.grid, self.threadgroup, self.by_threadgroups
        ))
        .chain(self.bindings.iter().map(|b| {
            format!(
                "  [{}] {} buf#{} off={}",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.buffer_id,
                b.offset
            )
        }))
    }

    /// Full command content, including buffer identity. Two dispatches with the
    /// same signature are replayable from one ICB slot.
    pub fn signature(&self) -> String {
        self.signature_parts().collect::<Vec<_>>().join("\n")
    }

    /// Command content with buffer identity erased, keeping offsets and the
    /// input/output role.
    ///
    /// Separating this from `signature` is what distinguishes the issue's second
    /// outcome ("differs only in buffer identity" — an allocator problem, fixable)
    /// from its third ("differs structurally" — the §14.4 trigger). Collapsing
    /// them into one comparison would report the harder verdict for the easier
    /// fault.
    pub fn shape_signature(&self) -> String {
        std::iter::once(format!(
            "{} grid={:?} tg={:?} tgmode={}",
            self.pipeline, self.grid, self.threadgroup, self.by_threadgroups
        ))
        .chain(self.bindings.iter().map(|b| {
            format!(
                "  [{}] {} off={}",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.offset
            )
        }))
        .collect::<Vec<_>>()
        .join("\n")
    }

    /// Kernel name and binding slots only — no grid, no offsets, no buffers.
    ///
    /// Separates "the op sequence changed" from "the same ops ran over more
    /// data". Grid size is a dispatch-tier parameter (`DESIGN.md` §7.1), so a
    /// sequence whose kernels are fixed and whose grids merely scale with
    /// `kv_len` is a materially different finding — and a much more recoverable
    /// one — than a sequence whose kernels differ.
    pub fn kernel_signature(&self) -> String {
        std::iter::once(self.pipeline.clone())
            .chain(self.bindings.iter().map(|b| {
                format!(
                    "  [{}] {}",
                    b.index,
                    if b.is_output { "out" } else { "in " }
                )
            }))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

struct TraceState {
    dispatches: Vec<Dispatch>,
    /// Bindings accumulated since the last dispatch.
    pending: Vec<Binding>,
    /// Pipeline name set by the most recent `set_compute_pipeline_state`.
    current_pipeline: Option<String>,
    /// Buffer address -> interned id.
    buffer_ids: HashMap<usize, u64>,
    next_buffer_id: u64,
    region: Option<String>,
    next_seq: u64,
    /// Set by `record_barrier` when `auto_barrier` emits one, consumed by the
    /// dispatch that immediately follows it.
    pending_barrier: bool,
    /// The directions behind `pending_barrier`, consumed with it (issue #185).
    pending_kinds: crate::kernels::metal::encoder::HazardKinds,
    /// Encoder sessions begun so far, so a dispatch can name the one it is in.
    encoder: u64,
}

impl TraceState {
    fn new() -> Self {
        TraceState {
            dispatches: Vec::new(),
            pending: Vec::new(),
            current_pipeline: None,
            buffer_ids: HashMap::new(),
            next_buffer_id: 0,
            region: None,
            next_seq: 0,
            pending_barrier: false,
            pending_kinds: crate::kernels::metal::encoder::HazardKinds::NONE,
            encoder: 0,
        }
    }

    fn intern(&mut self, addr: usize) -> u64 {
        if let Some(id) = self.buffer_ids.get(&addr) {
            return *id;
        }
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.buffer_ids.insert(addr, id);
        id
    }
}

fn state() -> &'static Mutex<TraceState> {
    static STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(TraceState::new()))
}

/// Whether tracing was requested via `CANDLE_METAL_TRACE`.
///
/// Read once. A per-dispatch `std::env::var` would be a lock and an allocation
/// on the hot path, which is precisely the kind of measurement-tool cost
/// `DESIGN.md` §2.4 warns about.
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CANDLE_METAL_TRACE")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false)
    })
}

/// Set independently of `enabled()` so recording can be gated to a region of
/// interest — tracing every prefill dispatch of a 700-token prompt to answer a
/// question about decode would bury the answer.
static RECORDING: AtomicBool = AtomicBool::new(false);

/// Dispatches observed while recording was off, so the trace can report what it
/// did not see rather than implying it saw everything.
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// True when a dispatch should be recorded.
#[inline]
pub fn is_recording() -> bool {
    enabled() && RECORDING.load(Ordering::Relaxed)
}

/// Turn recording on or off. No-op unless `CANDLE_METAL_TRACE` is set.
pub fn set_recording(on: bool) {
    if enabled() {
        RECORDING.store(on, Ordering::Relaxed);
    }
}

/// Label subsequent dispatches, e.g. with a decode step index.
pub fn set_region(region: Option<String>) {
    if !enabled() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.region = region;
    }
}

/// Record the pipeline that later dispatches will run.
#[inline]
pub fn record_pipeline(name: &str) {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.current_pipeline = Some(name.to_string());
    }
}

/// Record that `auto_barrier` emitted a barrier before the next dispatch.
///
/// Called from the barrier site itself, so this is an observation of what
/// candle did rather than a reconstruction of what it would do. `DESIGN.md`
/// §9.2e asks for barriers per token as a standing check on arena work, and a
/// count derived from bindings alone would have to model encoder boundaries
/// (§9.2e's simulation does; see `tools/barrier-count/`).
#[inline]
pub fn record_barrier() {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.pending_barrier = true;
    }
}

/// Record which hazard directions the barrier about to be emitted is for
/// (issue #185).
///
/// Called from `auto_barrier` immediately before [`record_barrier`], because
/// this is the only point where the answer exists: `prev_outputs` and
/// `prev_inputs` are replaced two lines later, so the evidence a downstream
/// reconstruction would need is gone. §9.2f's rule that the barrier count cannot
/// be simulated applies with more force to its attribution.
#[inline]
pub fn record_barrier_kinds(kinds: crate::kernels::metal::encoder::HazardKinds) {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.pending_kinds = kinds;
    }
}

/// Record that `auto_barrier` had a barrier pending and **did not emit it**,
/// because the position is being replayed from the ICB (issue #144,
/// `ReplayBarriers::SkipReplayed`).
///
/// Counted at the suppression site rather than by differencing two runs, for
/// §2.4's reason: *an instrument that cannot be shown to have engaged has not
/// measured anything*, and #69's determinism gate was vacuous because both arms
/// silently ran the default. A nonzero count here is the quantity that shows
/// the axis took effect.
///
/// **It is engagement proof and never the correctness argument** (§2.4). A
/// wrongly-suppressed edge leaves this count higher and the barrier count lower,
/// both entirely plausible; the correctness evidence is §15.1 #7's digest gate
/// and §15.1 #8's `lfm2-smoke`.
///
/// Note the pending barrier is **deferred rather than discharged** -- see
/// `auto_barrier` -- so one suppressed here may still be emitted at the next
/// classically-encoded position. This therefore counts *suppression events*,
/// not edges removed, and the two are different quantities.
/// # Why this counter is NOT gated on `is_recording`
///
/// Everything else in this module is, because a trace is opt-in and expensive.
/// This one must not be, and the reason is the whole point of having it: the
/// harness that **gates** this axis is `lfm2-determinism`, which never calls
/// `set_recording` at all. A counter behind `is_recording()` would read 0 there
/// on both arms, and a reader would conclude the axis had not engaged -- or
/// worse, would not notice.
///
/// That is precisely #69's vacuous determinism run (§9.2f): the harness consumed
/// the `OnceLock` guarding the mode switch, both arms ran the default, and the
/// "changed" arm reported a passing digest for the unchanged path. It was caught
/// by checking that the **barrier count moved between arms**, not by trusting
/// the flag. This counter is that check for this axis, so it has to be readable
/// in the run that does the gating.
///
/// It is a single relaxed increment on a path that already takes a mutex, so
/// the cost is nil beside `auto_barrier`'s own lock.
#[inline]
pub fn record_barrier_suppressed() {
    BARRIERS_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
}

/// Suppression events since process start (issue #144).
///
/// Process-cumulative rather than per step, because the quantity it exists to
/// answer is *"did the switch engage at all"* and that is answered by any
/// nonzero value. Per-step figures come from the dispatch trace, which records
/// barriers against the position that follows them.
pub fn barriers_suppressed() -> usize {
    BARRIERS_SUPPRESSED.load(Ordering::Relaxed)
}

/// Suppression events, independent of whether a trace is being recorded.
static BARRIERS_SUPPRESSED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Record one `executeCommandsInBuffer` and the `useResource` declarations made
/// for it, independent of whether a trace is being recorded.
///
/// The residency term of a replayed step is `runs x |resident(run)|`
/// (`DESIGN.md` §11.3w). Counted at the call site rather than derived from the
/// plan, so a harness reports what the encoder declared, not what the plan
/// intended to.
#[inline]
pub fn record_execute_range(resources: usize) {
    EXECUTE_RANGES.fetch_add(1, Ordering::Relaxed);
    USE_RESOURCE.fetch_add(resources as u64, Ordering::Relaxed);
}

/// `executeCommandsInBuffer` calls made, process-wide.
pub fn execute_range_count() -> u64 {
    EXECUTE_RANGES.load(Ordering::Relaxed)
}

/// `useResource` calls made for ICB ranges, process-wide.
pub fn use_resource_count() -> u64 {
    USE_RESOURCE.load(Ordering::Relaxed)
}

static EXECUTE_RANGES: AtomicU64 = AtomicU64::new(0);
static USE_RESOURCE: AtomicU64 = AtomicU64::new(0);

/// Record that a new encoder session began, resetting hazard state.
///
/// Barriers order dispatches only *within* an encoder, so a barrier count is
/// uninterpretable without knowing where the sessions divide. Candle opens one
/// every `CANDLE_METAL_COMPUTE_PER_BUFFER` dispatches.
#[inline]
pub fn record_encoder_begin() {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.encoder += 1;
    }
}

/// Record a buffer binding against the dispatch that will consume it.
#[inline]
pub fn record_binding(index: usize, addr: usize, offset: usize, len: usize, is_output: bool) {
    if !is_recording() {
        return;
    }
    if let Ok(mut s) = state().lock() {
        let buffer_id = s.intern(addr);
        s.pending.push(Binding {
            index,
            buffer_id,
            buffer_addr: addr,
            offset,
            len,
            is_output,
        });
    }
}

/// Record a dispatch, consuming the bindings accumulated since the last one.
///
/// Bindings are cleared even when not recording, so that turning recording on
/// mid-stream does not attribute a previous dispatch's bindings to the first
/// recorded one.
#[inline]
pub fn record_dispatch(
    grid: (usize, usize, usize),
    threadgroup: (usize, usize, usize),
    by_threadgroups: bool,
) {
    if !enabled() {
        return;
    }
    if !RECORDING.load(Ordering::Relaxed) {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut s) = state().lock() {
            s.pending.clear();
            // Cleared with the bindings, and for the same reason: an unrecorded
            // dispatch's barrier must not be attributed to the first recorded
            // one.
            s.pending_barrier = false;
            s.pending_kinds = crate::kernels::metal::encoder::HazardKinds::NONE;
        }
        return;
    }
    if let Ok(mut s) = state().lock() {
        let seq = s.next_seq;
        s.next_seq += 1;
        let bindings = std::mem::take(&mut s.pending);
        let pipeline = s
            .current_pipeline
            .clone()
            // A dispatch with no pipeline recorded would be a gap in the trace,
            // so it is named rather than dropped.
            .unwrap_or_else(|| "<unknown>".to_string());
        let region = s.region.clone();
        let barrier = std::mem::take(&mut s.pending_barrier);
        let kinds = s.pending_kinds.take();
        let encoder = s.encoder;
        s.dispatches.push(Dispatch {
            seq,
            pipeline,
            grid,
            threadgroup,
            by_threadgroups,
            bindings,
            barrier,
            kinds,
            encoder,
            region,
        });
    }
}

/// Every dispatch recorded so far.
pub fn take_dispatches() -> Vec<Dispatch> {
    if !enabled() {
        return Vec::new();
    }
    match state().lock() {
        Ok(mut s) => std::mem::take(&mut s.dispatches),
        Err(_) => Vec::new(),
    }
}

/// Dispatches seen while recording was off.
pub fn skipped_count() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// Whether `CANDLE_METAL_TRACE` requested tracing, for a harness to report when
/// it produced nothing because the variable was unset.
pub fn trace_requested() -> bool {
    enabled()
}
