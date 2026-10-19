use super::*;
use std::collections::HashMap;
use std::sync::Mutex;

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

    fn note_pipeline(&mut self, name: &str) {
        self.current_pipeline = Some(name.to_string());
    }

    fn note_barrier(&mut self, kinds: crate::kernels::metal::encoder::HazardKinds) {
        self.pending_kinds = kinds;
        self.pending_barrier = true;
    }

    fn note_binding(
        &mut self,
        index: usize,
        addr: usize,
        offset: usize,
        len: usize,
        is_output: bool,
    ) {
        let buffer_id = self.intern(addr);
        self.pending.push(Binding {
            index,
            buffer_id,
            buffer_addr: addr,
            offset,
            len,
            is_output,
        });
    }

    fn note_dispatch(
        &mut self,
        grid: (usize, usize, usize),
        threadgroup: (usize, usize, usize),
        by_threadgroups: bool,
    ) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let bindings = std::mem::take(&mut self.pending);
        let pipeline = self
            .current_pipeline
            .clone()
            // A dispatch with no pipeline recorded would be a gap in the trace,
            // so it is named rather than dropped.
            .unwrap_or_else(|| "<unknown>".to_string());
        let region = self.region.clone();
        let barrier = std::mem::take(&mut self.pending_barrier);
        let kinds = self.pending_kinds.take();
        let encoder = self.encoder;
        self.dispatches.push(Dispatch {
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

/// One device's dispatch trace: every dispatch encoded on that device's queue
/// while a [`TraceWindow`] is open, with its pipeline, bindings, barrier and
/// encoder session.
///
/// The process-wide trace is one recorder behind one recording switch, so
/// two devices encoding at once -- two tests on parallel threads -- interleave
/// their pipelines and bindings in it, and one caller's
/// [`take_dispatches`] or `set_recording(false)` empties or stops the other's.
/// This one is owned by the device's `Commands` and fed by the encoders it
/// opens, whose dispatches the `Commands` lock already serializes, so a
/// window reads its own device's command stream and nothing else, whichever
/// thread encodes. Recorded beside the process-wide trace, never instead of
/// it, independent of [`set_recording`], and only under the trace switch
/// (`CANDLE_METAL_TRACE`): with it off the encoder holds no device trace. The
/// window records no region; buffer ids are interned from zero per window.
/// (`invariants.md` §15.1n)
#[derive(Default)]
pub struct DeviceTrace {
    open: AtomicBool,
    state: parking_lot::Mutex<Option<TraceState>>,
}

impl DeviceTrace {
    /// Open a window: from here until [`TraceWindow::close`], every dispatch
    /// encoded on this device is recorded.
    pub fn open(self: &Arc<Self>) -> Result<TraceWindow, TraceRefused> {
        if !enabled() {
            return Err(TraceRefused::TraceOff);
        }
        let mut state = self.state.lock();
        if self
            .open
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(TraceRefused::WindowOpen);
        }
        *state = Some(TraceState::new());
        Ok(TraceWindow {
            trace: Arc::clone(self),
        })
    }

    #[inline]
    fn with(&self, f: impl FnOnce(&mut TraceState)) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        if let Some(s) = self.state.lock().as_mut() {
            f(s);
        }
    }
}

/// An open [`DeviceTrace`] window. Dropping it without [`Self::close`] closes
/// it and discards the dispatches.
pub struct TraceWindow {
    trace: Arc<DeviceTrace>,
}

impl TraceWindow {
    /// Close the window and return its dispatches in encode order.
    pub fn close(self) -> Vec<Dispatch> {
        let state = self.trace.state.lock().take();
        drop(self);
        state.map(|s| s.dispatches).unwrap_or_default()
    }
}

impl Drop for TraceWindow {
    fn drop(&mut self) {
        self.trace.open.store(false, Ordering::Release);
        self.trace.state.lock().take();
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
    *ENABLED.get_or_init(|| crate::tuning::tuning().instrument.trace)
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

/// Turn recording on or off. An unavailable enable request returns an error;
/// in a diagnostic build this is a no-op unless `CANDLE_METAL_TRACE` is set.
pub fn set_recording(on: bool) -> crate::Result<()> {
    crate::tuning::check_profile_request("trace", on).map_err(crate::Error::wrap)?;
    if enabled() {
        RECORDING.store(on, Ordering::Relaxed);
    }
    Ok(())
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
pub fn record_pipeline(device: Option<&DeviceTrace>, name: &str) {
    if !enabled() {
        return;
    }
    if let Some(d) = device {
        d.with(|s| s.note_pipeline(name));
    }
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.note_pipeline(name);
    }
}

/// Record that `auto_barrier` emitted a barrier before the next dispatch, and
/// which hazard directions it is for.
///
/// Called from the barrier site itself, so this is an observation of what
/// candle did rather than a reconstruction of what it would do. `DESIGN.md`
/// §9.2e asks for barriers per token as a standing check on arena work, and a
/// count derived from bindings alone would have to model encoder boundaries
/// (§9.2e's simulation does; see `tools/barrier-count/`). The kinds exist only
/// here: `prev_outputs` and `prev_inputs` are replaced right after, so a
/// downstream reconstruction has no evidence left to attribute from (§9.2f).
/// Barriers emitted, encoder sessions opened and command buffers committed
/// since process start, for a per-token census of what a pass plan changes
/// about the shape of a step (#814 item B). Three relaxed increments on paths
/// that already take a lock; unlike everything else in this module they are
/// NOT gated on a trace being recorded, for `record_barrier_suppressed`'s
/// reason: the runs that need them do not call `set_recording`.
#[inline]
pub fn record_barrier(
    device: Option<&DeviceTrace>,
    kinds: crate::kernels::metal::encoder::HazardKinds,
) {
    if !enabled() {
        return;
    }
    if let Some(d) = device {
        d.with(|s| s.note_barrier(kinds));
    }
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.note_barrier(kinds);
    }
}

pub fn record_encoder_begin(device: Option<&DeviceTrace>) {
    if !enabled() {
        return;
    }
    if let Some(d) = device {
        d.with(|s| s.encoder += 1);
    }
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.encoder += 1;
    }
}

/// Record a buffer binding against the dispatch that will consume it.
#[inline]
pub fn record_binding(
    device: Option<&DeviceTrace>,
    index: usize,
    addr: usize,
    offset: usize,
    len: usize,
    is_output: bool,
) {
    if !enabled() {
        return;
    }
    if let Some(d) = device {
        d.with(|s| s.note_binding(index, addr, offset, len, is_output));
    }
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(mut s) = state().lock() {
        s.note_binding(index, addr, offset, len, is_output);
    }
}

/// Record a dispatch, consuming the bindings accumulated since the last one.
///
/// Bindings are cleared even when not recording, so that turning recording on
/// mid-stream does not attribute a previous dispatch's bindings to the first
/// recorded one.
#[inline]
pub fn record_dispatch(
    device: Option<&DeviceTrace>,
    grid: (usize, usize, usize),
    threadgroup: (usize, usize, usize),
    by_threadgroups: bool,
) {
    if !enabled() {
        return;
    }
    if let Some(d) = device {
        d.with(|s| s.note_dispatch(grid, threadgroup, by_threadgroups));
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
        s.note_dispatch(grid, threadgroup, by_threadgroups);
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

/// Effective trace-active status. In a diagnostic build this equals the
/// requested tuning; a shipping build reports an unavailable legacy request
/// once and caches inactive. Configured setup uses `tuning::try_tuning`.
pub fn trace_requested() -> bool {
    enabled()
}
