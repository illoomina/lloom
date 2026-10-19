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
//! The recorder is present in debug builds or with `--features profile`; ordinary
//! optimized shipping keeps the engagement counters and refuses explicit windows.
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

#[cfg(any(debug_assertions, feature = "profile"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

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

/// Why [`DeviceTrace::open`] refused a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceRefused {
    /// The optional recorder is absent from this build.
    NotCompiled,
    /// The trace switch is off: nothing is recorded, so a window would read
    /// empty and look like a trace of nothing.
    TraceOff,
    /// A window is already open on this device; a second would clear its
    /// dispatches.
    WindowOpen,
}

pub static BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENCODER_SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COMMAND_BUFFERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(barriers, encoder sessions, command buffers)` since process start.
pub fn shape_counts() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        BARRIERS.load(Relaxed),
        ENCODER_SESSIONS.load(Relaxed),
        COMMAND_BUFFERS.load(Relaxed),
    )
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

/// One encoder session opened ([`ENCODER_SESSIONS`]).
///
/// Barriers order dispatches only *within* an encoder, so a barrier count is
/// uninterpretable without knowing where the sessions divide. Candle opens one
/// every `CANDLE_METAL_COMPUTE_PER_BUFFER` dispatches.
#[inline]
pub fn note_encoder_session() {
    ENCODER_SESSIONS.fetch_add(1, Ordering::Relaxed);
}

/// One command buffer committed ([`COMMAND_BUFFERS`]).
#[inline]
pub fn note_command_buffer() {
    COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(any(debug_assertions, feature = "profile"))]
mod recording;
#[cfg(any(debug_assertions, feature = "profile"))]
pub use recording::{
    is_recording, record_binding, record_dispatch, record_encoder_begin, record_pipeline,
    set_recording, set_region, skipped_count, take_dispatches, trace_requested, DeviceTrace,
    TraceWindow,
};

#[inline]
pub fn record_barrier(
    device: Option<&DeviceTrace>,
    kinds: crate::kernels::metal::encoder::HazardKinds,
) {
    BARRIERS.fetch_add(1, Ordering::Relaxed);
    #[cfg(any(debug_assertions, feature = "profile"))]
    recording::record_barrier(device, kinds);
    #[cfg(not(any(debug_assertions, feature = "profile")))]
    let _ = (device, kinds);
}

#[cfg(not(any(debug_assertions, feature = "profile")))]
mod unavailable {
    use super::*;
    /// No recorder allocation or locking exists in this build.
    #[derive(Default)]
    pub struct DeviceTrace {
        _private: (),
    }
    impl DeviceTrace {
        pub fn open(self: &Arc<Self>) -> Result<TraceWindow, TraceRefused> {
            Err(TraceRefused::NotCompiled)
        }
    }
    pub struct TraceWindow {
        _private: (),
    }
    impl TraceWindow {
        pub fn close(self) -> Vec<Dispatch> {
            Vec::new()
        }
    }
    /// Effective trace-active status, not the raw requested tuning value.
    /// A legacy unavailable request is reported once and cached inactive.
    /// Use `tuning::try_tuning` for fallible configured setup.
    pub fn trace_requested() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            crate::tuning::effective_instrument("trace", crate::tuning::tuning().instrument.trace)
        })
    }
    pub fn is_recording() -> bool {
        let _ = trace_requested();
        false
    }
    /// An enable request returns an actionable unsupported error; disabling works.
    pub fn set_recording(on: bool) -> crate::Result<()> {
        crate::tuning::check_profile_request("trace", on).map_err(crate::Error::wrap)
    }
    pub fn set_region(_region: Option<String>) {}
    pub fn record_pipeline(_device: Option<&DeviceTrace>, _name: &str) {}
    pub fn record_encoder_begin(_device: Option<&DeviceTrace>) {}
    pub fn record_binding(
        _device: Option<&DeviceTrace>,
        _index: usize,
        _addr: usize,
        _offset: usize,
        _len: usize,
        _is_output: bool,
    ) {
    }
    pub fn record_dispatch(
        _device: Option<&DeviceTrace>,
        _grid: (usize, usize, usize),
        _threadgroup: (usize, usize, usize),
        _by_threadgroups: bool,
    ) {
    }
    pub fn take_dispatches() -> Vec<Dispatch> {
        Vec::new()
    }
    pub fn skipped_count() -> u64 {
        0
    }
}
#[cfg(not(any(debug_assertions, feature = "profile")))]
pub use unavailable::*;
