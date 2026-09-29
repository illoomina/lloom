//! Opt-in profiling of the Metal submission path: GPU busy time and dispatch count.
//!
//! The question this exists to answer is where a decode token's wall time goes.
//! Metal reports `GPUStartTime`/`GPUEndTime` per command buffer, so summing
//! `end - start` over every command buffer in a window gives GPU busy time, and
//! the window's wall time minus that is time the GPU was not executing --
//! which for a serial decode loop is CPU-side encode, submission and readback.
//!
//! Two properties of the measurement bound what it can conclude:
//!
//! * **Busy time is per command buffer, not per dispatch.** Dispatches inside one
//!   encoder overlap (`MTLDispatchType::Concurrent`), so a per-dispatch sum would
//!   double-count. The command buffer is the smallest unit Metal reports that
//!   does not.
//! * **Command buffers can overlap each other.** Candle commits every
//!   `CANDLE_METAL_COMPUTE_PER_BUFFER` encoders without waiting, so two buffers
//!   can be in flight at once and their intervals can intersect. Summing them
//!   would then exceed true busy time. The union of the intervals is reported
//!   alongside the sum; where they differ the union is the honest figure, and the
//!   gap between them measures how much the submission pipeline actually overlaps.
//!
//! Enable with `CANDLE_METAL_PROFILE=1`. Disabled, every hook is a cached bool
//! check and nothing is recorded -- it must be able to run in a release build
//! against the real model, since an instrumented debug build would measure a
//! different dispatch pattern than the one that ships.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Whether profiling was requested, read once from the environment.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::tuning::tuning().instrument.profile)
}

/// One command buffer's GPU execution interval, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuInterval {
    pub start: f64,
    pub end: f64,
}

/// One completed command buffer: its GPU interval, the host's commit stamp in
/// the same clock (`CLOCK_UPTIME_RAW`, domain A of `DESIGN.md` §3.4b, which
/// `GPUStartTime`/`GPUEndTime` share), and the decode step it was committed
/// in (`set_step`; 0 before the loop's first step).
///
/// The commit stamp is what turns a gap between two buffers into a verdict
/// (issue #702): if the next buffer was committed **after** the previous one
/// ended, the GPU had nothing to run and the host caused the idle
/// (*starved*); if it was already committed, the idle is the driver's
/// retire-and-start between two queued buffers (*seam*), which no host thread
/// can shorten.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandBufferRecord {
    pub interval: GpuInterval,
    /// When the host committed this buffer, seconds in the GPU clock. Stamped
    /// at completion-handler registration, which immediately precedes the
    /// `commit` call, so it reads a few microseconds **early** -- the bias is
    /// toward calling a gap a seam, never toward calling it starved.
    pub committed: f64,
    /// The decode step current at commit (`set_step`).
    pub step: u64,
    /// The pipeline label of the LAST dispatch encoded before this buffer was
    /// committed. Exact when a buffer holds one dispatch
    /// (`CANDLE_METAL_COMPUTE_PER_BUFFER=1`, the per-kernel timing arm, lloom
    /// #715); otherwise it names the buffer's last kernel only.
    pub label: Option<Arc<str>>,
}

/// The label of the most recent dispatch, for [`record_command_buffer`] to
/// stamp onto the buffer it is committed in. One slot, written at every
/// dispatch under the recorder's lock it already takes.
#[derive(Default)]
struct LastLabel(Option<Arc<str>>);

#[derive(Default)]
struct Recorder {
    /// See [`LastLabel`].
    last_label: LastLabel,
    /// GPU intervals of every command buffer that has completed, with the
    /// commit stamp and step that let a gap be attributed.
    records: Vec<CommandBufferRecord>,
    /// Dispatches per pipeline label, for the per-token dispatch inventory.
    ///
    /// **Keyed on `Arc<str>` rather than `String`, so counting a dispatch
    /// allocates nothing** (issue #579). The caller already holds the label as
    /// an `Arc<str>` -- `ComputePipeline` stores it that way and
    /// `EncoderState::current_pipeline` keeps it that way -- so a lookup
    /// borrows and only a *first* sighting of a label clones the handle into
    /// the key.
    ///
    /// The `String` key this replaces cost a heap allocation and a copy at
    /// **every** dispatch: 675 per decode token, measured (§11.2), on the
    /// instrument every measurement in this project runs under. §2.4's rule is
    /// that an instrument perturbing what it measures has not measured the
    /// thing its name promises, and §6.4a had already priced this exact shape
    /// at ~3.6 % of forward-pass CPU without connecting it to the census.
    dispatches: HashMap<Arc<str>, u64>,
}

fn recorder() -> &'static Mutex<Recorder> {
    static RECORDER: OnceLock<Mutex<Recorder>> = OnceLock::new();
    RECORDER.get_or_init(|| Mutex::new(Recorder::default()))
}

/// Dispatches encoded since the last reset. Separate from the per-label map so
/// the count is available even where no label is supplied.
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// Command buffers committed since the last reset.
static COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
/// Compute encoders opened since the last reset. Encoder breaks are what
/// serialize the GPU (`DESIGN.md` §3.5), so the count belongs next to the
/// dispatch count rather than being inferred from it.
static ENCODERS: AtomicU64 = AtomicU64::new(0);
/// Command-buffer lifecycle time spent on the encoding thread since the last
/// reset: `commit_swap_locked` whole (end the encoder, publish residency,
/// signal, handlers, commit, create the next buffer, swap), the
/// `commandBuffer()` creation inside it, and `flush_and_prune` whole. The
/// host-side price of a command-buffer seam, which the GPU-interval union
/// cannot see (issue #702).
static COMMITS: AtomicU64 = AtomicU64::new(0);
static COMMIT_NS: AtomicU64 = AtomicU64::new(0);
static CREATE_CB_NS: AtomicU64 = AtomicU64::new(0);
static PRUNES: AtomicU64 = AtomicU64::new(0);
static PRUNE_NS: AtomicU64 = AtomicU64::new(0);
/// Cross-encoder fence-map reads, hits, and `waitForFence` calls, summed once
/// per encoder close from the session's own counters (`EncoderState`) -- so a
/// decode step's per-bind cost has a count beside the dispatch count (lloom
/// #727 §27 counted 1101 reads per step from a trace; #732 made the count a
/// profiler figure and changed it).
static FENCE_PROBES: AtomicU64 = AtomicU64::new(0);
static FENCE_HITS: AtomicU64 = AtomicU64::new(0);
static FENCE_WAITS: AtomicU64 = AtomicU64::new(0);

/// The decode step the loop is in, for attributing a command buffer's gap to
/// a step (`CommandBufferRecord::step`). Written once per step by the loop,
/// only under `enabled()`; read at every completion-handler registration.
static STEP: AtomicU64 = AtomicU64::new(0);

/// Tell the recorder which decode step is current. One relaxed store; the
/// loop calls it only when `enabled()`.
#[inline]
pub fn set_step(step: u64) {
    STEP.store(step, Ordering::Relaxed);
}

/// The step current at this instant, as `set_step` last left it.
#[inline]
pub fn current_step() -> u64 {
    STEP.load(Ordering::Relaxed)
}

/// Now, in seconds of the clock `GPUStartTime`/`GPUEndTime` are in.
///
/// `CLOCK_UPTIME_RAW` -- domain A of `DESIGN.md` §3.4b, one 24 MHz timebase
/// shared by CPU and GPU on Apple silicon (`sampleTimestamps:gpuTimestamp:`
/// is the identity there). Not `Instant`, which is the same clock but does not
/// expose its absolute value. One `clock_gettime` per command buffer, under
/// `enabled()` only.
#[cfg(target_os = "macos")]
pub fn now_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`; `CLOCK_UPTIME_RAW` is a
    // clock id Darwin accepts. The call cannot fail for it.
    unsafe {
        libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut ts);
    }
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(not(target_os = "macos"))]
pub fn now_s() -> f64 {
    0.0
}

/// Record one `commit_swap_locked`, `elapsed` on the calling thread.
#[inline]
pub fn record_commit(elapsed: std::time::Duration) {
    COMMITS.fetch_add(1, Ordering::Relaxed);
    COMMIT_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// Record one `commandBuffer()` creation, `elapsed` on the calling thread.
#[inline]
pub fn record_create_cb(elapsed: std::time::Duration) {
    CREATE_CB_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

static END_ENC_NS: AtomicU64 = AtomicU64::new(0);
static RESIDENCY_NS: AtomicU64 = AtomicU64::new(0);
static CB_COMMIT_NS: AtomicU64 = AtomicU64::new(0);

/// The three halves of a `commit_swap_locked` beside the creation: ending
/// the encoder (the fence map), the residency set's commit, and the
/// command buffer's own `commit` with its handlers — so a seam's cost is
/// attributed, not summed (issue #749 P26: 0.45 ms a seam at prefill
/// against 16 µs at decode).
#[inline]
pub fn record_commit_parts(
    end_enc: std::time::Duration,
    residency: std::time::Duration,
    cb_commit: std::time::Duration,
) {
    END_ENC_NS.fetch_add(end_enc.as_nanos() as u64, Ordering::Relaxed);
    RESIDENCY_NS.fetch_add(residency.as_nanos() as u64, Ordering::Relaxed);
    CB_COMMIT_NS.fetch_add(cb_commit.as_nanos() as u64, Ordering::Relaxed);
}

/// Record one `flush_and_prune`, `elapsed` on the calling thread.
#[inline]
pub fn record_prune(elapsed: std::time::Duration) {
    PRUNES.fetch_add(1, Ordering::Relaxed);
    PRUNE_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// The key a dispatch with no recorded pipeline name is counted under.
///
/// Built once. The `&'static str` this replaces could not be a key at all once
/// the map became `Arc<str>`-keyed, and materialising one per unnamed dispatch
/// would reintroduce exactly the allocation this module was changed to remove.
fn unnamed() -> &'static Arc<str> {
    static UNNAMED: OnceLock<Arc<str>> = OnceLock::new();
    UNNAMED.get_or_init(|| Arc::from("<unnamed>"))
}

/// Record that a dispatch was encoded, attributed to `label`.
///
/// **Takes the caller's `Arc<str>` handle, and allocates nothing on the path
/// that runs** (issue #579). The label reaches here from
/// `EncoderState::current_pipeline`, which holds the same `Arc<str>`
/// `ComputePipeline` was built with, so:
///
/// * a label already in the map costs a hash, a compare and an increment;
/// * a label seen for the first time clones the handle -- a refcount bump, not
///   a copy of the bytes -- into the key.
///
/// Measured on `lfm2-decode-profile --n 200`: **17 distinct kernels over 675
/// dispatches per token**, so the clone happens 17 times in a run and the
/// other 675-per-token are pure lookups. Before this it was `label.to_string()`
/// at every dispatch -- a heap allocation and a copy, 675 per token, on the
/// instrument **every measurement in this project is taken under**. §2.4 is
/// the rule it was breaking (*check whether the measurement tool is the
/// cost*), and §6.4a had already priced the shape at ~3.6 % of forward-pass
/// CPU.
///
/// `None` means no pipeline name was recorded; it is counted under a single
/// shared `<unnamed>` key rather than dropped, so the per-label totals still
/// sum to `DISPATCHES`.
///
/// `input_bytes` is the largest input buffer the dispatch bound (see
/// `EncoderState::dispatch_input_max`); it is summed onto the open
/// counter-sampled encoder, and is 0 when the counters are off.
#[inline]
pub fn record_dispatch(label: Option<&Arc<str>>, input_bytes: usize) {
    if !enabled() {
        return;
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    {
        let mut rec = recorder().lock().unwrap();
        bump_label(&mut rec.dispatches, label.unwrap_or_else(|| unnamed()));
        rec.last_label.0 = label.cloned();
    }
    if counters_enabled() {
        counters().lock().note_dispatch(label, input_bytes);
    }
}

/// Increment `label`'s count, cloning the handle only to create a new entry.
///
/// Split out from [`record_dispatch`] so the property that makes this change
/// worth anything -- *a repeat label allocates nothing* -- is testable without
/// the process-global recorder and without a GPU. A test on the public
/// function could observe only the counts, which are identical either way.
#[inline]
fn bump_label(map: &mut HashMap<Arc<str>, u64>, label: &Arc<str>) {
    // `get_mut` first so the common case -- a label already counted -- neither
    // clones the handle nor constructs an entry. `entry()` would take the key
    // by value, which is the clone this exists to avoid.
    if let Some(count) = map.get_mut(label) {
        *count += 1;
    } else {
        map.insert(Arc::clone(label), 1);
    }
}

/// One device's dispatch census: per-label counts of the dispatches encoded
/// on that device's queue while a [`CensusWindow`] is open.
///
/// The process-wide census ([`snapshot`]) counts every device in the
/// process, so two windows open at once on different devices -- two tests on
/// parallel threads -- each count the other's dispatches, and a [`reset`] in
/// one clears the other's. This one is owned by the device's `Commands` and
/// fed by the encoders it opens, so a window sees its own device and nothing
/// else, whichever thread encodes. Recorded beside the process-wide census,
/// never instead of it, and only under [`enabled`]: a closed window costs a
/// relaxed load per dispatch, and with profiling off the encoder does not
/// hold one.
#[derive(Default)]
pub struct DeviceCensus {
    open: AtomicBool,
    dispatches: parking_lot::Mutex<HashMap<Arc<str>, u64>>,
}

/// Why [`DeviceCensus::open`] refused a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CensusRefused {
    /// `CANDLE_METAL_PROFILE` is off: nothing is recorded, so a window would
    /// read empty and look like a census of nothing.
    ProfileOff,
    /// A window is already open on this device; a second would clear its
    /// counts.
    WindowOpen,
}

impl DeviceCensus {
    /// Open a window: from here until [`CensusWindow::close`], every dispatch
    /// encoded on this device is counted under its pipeline label.
    pub fn open(self: &Arc<Self>) -> Result<CensusWindow, CensusRefused> {
        if !enabled() {
            return Err(CensusRefused::ProfileOff);
        }
        let mut map = self.dispatches.lock();
        if self
            .open
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(CensusRefused::WindowOpen);
        }
        map.clear();
        Ok(CensusWindow {
            census: Arc::clone(self),
        })
    }

    /// Count one dispatch, if a window is open. Called by the encoder,
    /// already behind [`enabled`].
    #[inline]
    pub(crate) fn record(&self, label: Option<&Arc<str>>) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        bump_label(
            &mut self.dispatches.lock(),
            label.unwrap_or_else(|| unnamed()),
        );
    }
}

/// An open [`DeviceCensus`] window. Dropping it without [`Self::close`]
/// closes it and discards the counts.
pub struct CensusWindow {
    census: Arc<DeviceCensus>,
}

impl CensusWindow {
    /// Close the window and return its counts by label, in
    /// [`Snapshot::by_label`]'s order (descending count, then label).
    pub fn close(self) -> Vec<(String, u64)> {
        let map = std::mem::take(&mut *self.census.dispatches.lock());
        drop(self);
        sorted_by_count(&map)
    }
}

impl Drop for CensusWindow {
    fn drop(&mut self) {
        self.census.open.store(false, Ordering::Release);
    }
}

/// Per-label counts, descending by count, ties by label.
fn sorted_by_count(map: &HashMap<Arc<str>, u64>) -> Vec<(String, u64)> {
    let mut by_label: Vec<(String, u64)> = map.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    by_label.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    by_label
}

/// Add one encoder session's fence-probe counts. Called at encoder close,
/// already behind `enabled()` at the call site.
#[inline]
pub fn record_fence_stats(probes: u32, hits: u32, waits: u32) {
    FENCE_PROBES.fetch_add(u64::from(probes), Ordering::Relaxed);
    FENCE_HITS.fetch_add(u64::from(hits), Ordering::Relaxed);
    FENCE_WAITS.fetch_add(u64::from(waits), Ordering::Relaxed);
}

/// Record that a compute encoder was opened.
#[inline]
pub fn record_encoder() {
    if !enabled() {
        return;
    }
    ENCODERS.fetch_add(1, Ordering::Relaxed);
}

/// Record a completed command buffer's GPU interval, with the host's commit
/// stamp (`now_s` at registration) and the step it was committed in.
#[inline]
pub fn record_command_buffer(start: f64, end: f64, committed: f64, step: u64) {
    if !enabled() {
        return;
    }
    COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
    // A command buffer that encoded no GPU work reports zeros; recording it
    // would drag the union's lower bound back to the epoch.
    if start <= 0.0 || end <= start {
        return;
    }
    let mut rec = recorder().lock().unwrap();
    let label = rec.last_label.0.clone();
    rec.records.push(CommandBufferRecord {
        interval: GpuInterval { start, end },
        committed,
        step,
        label,
    });
}

/// Whether compute encoders are timed by stage-boundary counter samples
/// (`CANDLE_METAL_PROFILE_COUNTERS=1`, only under `CANDLE_METAL_PROFILE`).
///
/// A command buffer's `GPUStartTime`/`GPUEndTime` bound its residency, so a
/// kernel inside it cannot be priced from them (`DESIGN.md` §13.5r). The
/// start and end of every compute encoder's stage are the finest point this
/// GPU samples (§13.1d: `atDispatchBoundary` is unsupported), so the encoder is
/// the unit, and [`encoder_per_dispatch`] is the arm that makes an encoder one
/// kernel.
pub fn counters_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled() && crate::tuning::tuning().instrument.profile_counters)
}

/// Whether every dispatch gets its own compute encoder
/// (`CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH=1`, only under
/// `CANDLE_METAL_PROFILE`). It changes the submission — a seam per dispatch
/// instead of a barrier — so a step timed under it is a different step, and
/// the price of that is reported beside it rather than assumed away.
pub fn encoder_per_dispatch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled()
            && crate::tuning::tuning()
                .instrument
                .profile_encoder_per_dispatch
    })
}

/// One compute encoder's GPU interval from its stage-boundary counter
/// samples, with what it held and its command buffer's interval (the
/// containment check: an encoder cannot run outside its buffer).
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderRecord {
    /// Seconds in domain A (§3.4b), the clock of `GpuInterval`.
    pub start: f64,
    pub end: f64,
    /// The decode step current when the encoder was opened.
    pub step: u64,
    /// The first dispatch's pipeline label.
    pub label: Option<Arc<str>>,
    pub dispatches: u32,
    /// Sum over its dispatches of the largest input buffer each bound.
    pub input_bytes: u64,
    /// Its command buffer's `GPUStartTime`/`GPUEndTime`.
    pub cb_start: f64,
    pub cb_end: f64,
}

type SampleBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>>;
type CounterSet =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCounterSet>>;

/// Samples in one counter sample buffer: the largest length this device
/// creates (32 768 B of 8-byte timestamps, `DESIGN.md` §13.1n).
const SAMPLES_PER_BUFFER: usize = 4096;

/// A sample buffer shared by the encoders of any number of command buffers.
///
/// The device holds at most 32 sample buffers at once, whatever their size
/// (§13.1n), and more command buffers than that can be in flight, so a buffer
/// per command buffer is refused under load. Slots are handed out in order and
/// the buffer restarts from slot 0 once every encoder written into it has been
/// resolved: buffers are sized to the samples in flight, not to the command
/// buffers.
struct Slab {
    buffer: SampleBuffer,
    /// The next unused sample index.
    next: usize,
    /// Encoders given slots here and not yet resolved.
    pending: usize,
}

/// An encoder opened on a command buffer that has not completed.
struct PendingEncoder {
    step: u64,
    label: Option<Arc<str>>,
    dispatches: u32,
    input_bytes: u64,
    /// Its slab, and the first of its two samples there.
    slab: usize,
    index: usize,
}

#[derive(Default)]
struct Counters {
    set: Option<CounterSet>,
    /// Why this device cannot sample at all, once asked.
    unavailable: Option<String>,
    /// Every sample buffer created; never more than the device's live limit.
    slabs: Vec<Slab>,
    /// The slab new encoders take slots from.
    current: Option<usize>,
    /// Set by a refused allocation, cleared when a command buffer completes:
    /// until then an encoder with no free slot is dropped without asking again.
    at_limit: bool,
    /// Each incomplete command buffer's sampled encoders, in order, keyed on
    /// the `MTLCommandBuffer` object's address.
    by_cb: HashMap<usize, Vec<PendingEncoder>>,
    /// Most command buffers holding sampled encoders at once.
    in_flight_peak: usize,
    /// The command buffer whose last encoder is the one dispatches go to.
    /// Process-wide, so attribution assumes one encoding thread at a time: two
    /// threads encoding concurrently would interleave their dispatch counts
    /// and bytes (the intervals themselves stay each encoder's own).
    open: Option<usize>,
    records: Vec<EncoderRecord>,
    /// Encoders left unsampled: no slot free and no buffer to be had.
    dropped: u64,
    /// Sample buffer allocations the device refused.
    refused: u64,
    /// Metal's reason for the last refusal, with the buffers live at the time.
    refusal: Option<String>,
    /// The longest sample buffer allocation, created or refused, in seconds.
    alloc_max_s: f64,
    /// Samples that resolved to the error marker, or a range that would not resolve.
    bad: u64,
    /// Encoders that ran no dispatch, or whose samples resolved to zero: no
    /// work reached the GPU (§13.1d).
    empty: u64,
    /// Seconds per sample unit, fixed by the first resolved buffer (`sample_unit_s`).
    unit_s: Option<f64>,
}

// SAFETY: the Metal objects held here (a counter set, counter sample buffers)
// are reference-counted Objective-C objects that Metal documents as usable
// from any thread; every access is serialized by the mutex that owns this.
unsafe impl Send for Counters {}

fn counters() -> &'static parking_lot::Mutex<Counters> {
    static COUNTERS: OnceLock<parking_lot::Mutex<Counters>> = OnceLock::new();
    COUNTERS.get_or_init(|| parking_lot::Mutex::new(Counters::default()))
}

/// Seconds per `mach_absolute_time` tick (125/3 ns on Apple silicon, §3.4b).
fn tick_s() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    static TICK: OnceLock<f64> = OnceLock::new();
    *TICK.get_or_init(|| {
        let mut tb = Timebase { numer: 0, denom: 0 };
        // SAFETY: `mach_timebase_info` writes one `mach_timebase_info_data_t`
        // (two u32s, the layout above) through a valid out-pointer.
        unsafe { mach_timebase_info(&mut tb) };
        tb.numer as f64 / tb.denom.max(1) as f64 * 1e-9
    })
}

/// Seconds per counter-sample unit, from one sample and the `GPUStartTime`
/// (seconds, domain A) of the command buffer it was taken in.
///
/// Metal calls the sample a GPU timestamp and does not fix its unit. It is
/// measured here rather than assumed: the two candidates are the nanosecond
/// and the mach tick, 41.67x apart, and a sample taken microseconds into a
/// buffer on a clock running for days distinguishes them at any precision
/// that matters. `None` when neither fits, so a device writing a third unit
/// fails loudly (every sample counted bad) rather than converting wrongly.
fn sample_unit_s(sample: u64, cb_start: f64) -> Option<f64> {
    let ratio = cb_start / sample as f64;
    [1e-9, tick_s()]
        .into_iter()
        .find(|unit| (ratio / unit - 1.0).abs() < 1e-3)
}

impl Counters {
    fn init(&mut self, device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>) {
        use objc2_metal::{
            MTLCommonCounterSetTimestamp, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice as _,
        };
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            self.unavailable = Some("the device does not sample at stage boundaries".into());
            return;
        }
        // SAFETY: a framework-provided static NSString.
        let ts = unsafe { MTLCommonCounterSetTimestamp };
        self.set = device
            .counterSets()
            .and_then(|sets| sets.iter().find(|s| s.name().isEqualToString(ts)));
        if self.set.is_none() {
            self.unavailable = Some("the device exposes no timestamp counter set".into());
        }
    }

    /// A new sample buffer, or `None` when the device refuses one. A refusal
    /// is counted and not retried until a command buffer completes.
    fn new_buffer(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    ) -> Option<SampleBuffer> {
        use objc2_metal::{MTLCounterSampleBufferDescriptor, MTLDevice as _, MTLStorageMode};
        if self.at_limit {
            return None;
        }
        let set = self.set.as_ref()?;
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(set));
        d.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a plain count within the device's range; the device validates
        // it and fails the creation below rather than overrunning anything.
        unsafe { d.setSampleCount(SAMPLES_PER_BUFFER) };
        let t = std::time::Instant::now();
        let made = device.newCounterSampleBufferWithDescriptor_error(&d);
        self.alloc_max_s = self.alloc_max_s.max(t.elapsed().as_secs_f64());
        match made {
            Ok(b) => Some(b),
            Err(e) => {
                self.refused += 1;
                self.at_limit = true;
                self.refusal = Some(format!(
                    "{} buffers live: {}",
                    self.slabs.len(),
                    e.localizedDescription()
                ));
                None
            }
        }
    }

    /// Two consecutive sample slots `(slab, index)`: in the current slab, else
    /// in one every encoder of which has resolved, else in a new buffer.
    /// `None` when every slab is in use and the device will not give another.
    fn slot(
        &mut self,
        cb: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> Option<(usize, usize)> {
        use objc2_metal::MTLCommandBuffer as _;
        let fits = |s: &Slab| s.next + 2 <= SAMPLES_PER_BUFFER;
        let id = match self.current.filter(|&i| fits(&self.slabs[i])) {
            Some(i) => i,
            None => match self.slabs.iter().position(|s| s.pending == 0) {
                Some(i) => {
                    self.slabs[i].next = 0;
                    i
                }
                None => {
                    let buffer = self.new_buffer(&cb.device())?;
                    self.slabs.push(Slab {
                        buffer,
                        next: 0,
                        pending: 0,
                    });
                    self.slabs.len() - 1
                }
            },
        };
        self.current = Some(id);
        let s = &mut self.slabs[id];
        let index = s.next;
        s.next += 2;
        s.pending += 1;
        Some((id, index))
    }

    fn note_dispatch(&mut self, label: Option<&Arc<str>>, input_bytes: usize) {
        let Some(key) = self.open else { return };
        if let Some(e) = self.by_cb.get_mut(&key).and_then(|s| s.last_mut()) {
            e.dispatches += 1;
            if e.label.is_none() {
                e.label = label.cloned();
            }
            e.input_bytes += input_bytes as u64;
        }
    }
}

/// A sample buffer and the index of the first of two samples for a compute
/// encoder about to be opened on `cb`, or `None` to open it unsampled.
///
/// Called once per compute encoder. With the counters off it is one cached
/// bool; with them on, an encoder that cannot be sampled is counted as dropped
/// and its dispatches go unattributed rather than onto the previous encoder.
pub fn sample_encoder(
    cb: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Option<(SampleBuffer, usize)> {
    use objc2_metal::MTLCommandBuffer as _;
    if !counters_enabled() {
        return None;
    }
    let key = cb as *const _ as *const () as usize;
    let step = current_step();
    let mut c = counters().lock();
    if c.set.is_none() && c.unavailable.is_none() {
        c.init(&cb.device());
    }
    c.open = None;
    let Some((slab, index)) = c.slot(cb) else {
        c.dropped += 1;
        return None;
    };
    let buffer = c.slabs[slab].buffer.clone();
    let encoders = c.by_cb.entry(key).or_default();
    encoders.push(PendingEncoder {
        step,
        label: None,
        dispatches: 0,
        input_bytes: 0,
        slab,
        index,
    });
    c.in_flight_peak = c.in_flight_peak.max(c.by_cb.len());
    c.open = Some(key);
    Some((buffer, index))
}

/// Resolve the samples of the command buffer at `key`, which has completed
/// with the interval `[cb_start, cb_end]`. Called from its completion handler;
/// the lock is held only to take the entry and to file the records, not across
/// the resolve.
pub fn resolve_encoder_samples(key: usize, cb_start: f64, cb_end: f64) {
    use objc2_metal::MTLCounterSampleBuffer as _;
    if !counters_enabled() {
        return;
    }
    // Consecutive slots of one slab resolve as one range: `(slab, first, count)`.
    let (encoders, runs, mut unit) = {
        let mut c = counters().lock();
        c.at_limit = false;
        let Some(encoders) = c.by_cb.remove(&key) else {
            return;
        };
        let mut runs: Vec<(SampleBuffer, usize, usize)> = Vec::new();
        let mut last = usize::MAX;
        for e in &encoders {
            match runs.last_mut() {
                Some((_, first, n)) if e.slab == last && *first + 2 * *n == e.index => *n += 1,
                _ => runs.push((c.slabs[e.slab].buffer.clone(), e.index, 1)),
            }
            last = e.slab;
        }
        (encoders, runs, c.unit_s)
    };
    let n = encoders.len();
    let mut samples: Vec<Option<(u64, u64)>> = Vec::with_capacity(n);
    for (buffer, first, count) in &runs {
        // SAFETY: the range lies inside the buffer (`Counters::slot` bounds
        // every index by `SAMPLES_PER_BUFFER`), and the command buffer that
        // wrote it has completed; no other command buffer is given these slots
        // until this one's encoders are resolved.
        let data = unsafe {
            buffer.resolveCounterRange(objc2_foundation::NSRange::new(*first, 2 * count))
        };
        match data.map(|d| d.to_vec()) {
            Some(bytes) if bytes.len() >= 16 * count => {
                let at = |i: usize| {
                    u64::from_ne_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap_or([0; 8]))
                };
                samples.extend((0..*count).map(|i| Some((at(2 * i), at(2 * i + 1)))));
            }
            _ => samples.extend((0..*count).map(|_| None)),
        }
    }
    let (mut out, mut bad, mut empty) = (Vec::with_capacity(n), 0u64, 0u64);
    let mut drained: Vec<usize> = Vec::with_capacity(n);
    for (e, sample) in encoders.into_iter().zip(samples) {
        drained.push(e.slab);
        let Some((s, t)) = sample else {
            bad += 1;
            continue;
        };
        if s == u64::MAX || t == u64::MAX {
            bad += 1;
            continue;
        }
        // A slot is reused, so an encoder that ran nothing may still hold an
        // earlier encoder's samples; it is judged by its dispatches first.
        if e.dispatches == 0 || (s == 0 && t == 0) {
            empty += 1;
            continue;
        }
        if unit.is_none() {
            unit = sample_unit_s(s, cb_start);
        }
        match unit {
            None => bad += 1,
            Some(u) => out.push(EncoderRecord {
                start: s as f64 * u,
                end: t as f64 * u,
                step: e.step,
                label: e.label,
                dispatches: e.dispatches,
                input_bytes: e.input_bytes,
                cb_start,
                cb_end,
            }),
        }
    }
    let mut c = counters().lock();
    if c.unit_s.is_none() {
        c.unit_s = unit;
    }
    if c.open == Some(key) {
        c.open = None;
    }
    for slab in drained {
        c.slabs[slab].pending -= 1;
    }
    c.records.extend(out);
    c.bad += bad;
    c.empty += empty;
}

/// Encoders whose interval falls outside their command buffer's by more than
/// `slack_s`, or runs backwards: the check that the samples are in the clock
/// and unit the conversion assumes. Zero is the only readable answer.
pub fn encoder_containment(records: &[EncoderRecord], slack_s: f64) -> usize {
    records
        .iter()
        .filter(|r| r.end < r.start || r.start < r.cb_start - slack_s || r.end > r.cb_end + slack_s)
        .count()
}

/// The union of the encoders' intervals, seconds: what they cover of the GPU
/// timeline, to set against the command buffers' union (`gpu_busy`).
pub fn encoder_union_s(records: &[EncoderRecord]) -> f64 {
    let mut iv: Vec<GpuInterval> = records
        .iter()
        .map(|r| GpuInterval {
            start: r.start,
            end: r.end,
        })
        .collect();
    union_and_span(&mut iv).0
}

/// Encoder time by the kernel it held: one row per label over the
/// single-dispatch encoders, and one row for every encoder holding more than
/// one dispatch (whose time is not any one kernel's).
pub fn encoder_time_by_label(records: &[EncoderRecord]) -> Vec<EncoderLabelTiming> {
    let mut map: HashMap<String, (Vec<f64>, u64, u64)> = HashMap::new();
    for r in records {
        let key = if r.dispatches == 1 {
            r.label.as_deref().unwrap_or("<unlabelled>").to_string()
        } else {
            "<encoders of several dispatches>".to_string()
        };
        let e = map.entry(key).or_default();
        e.0.push(r.end - r.start);
        e.1 += u64::from(r.dispatches);
        e.2 += r.input_bytes;
    }
    let mut v: Vec<EncoderLabelTiming> = map
        .into_iter()
        .map(|(label, (mut xs, dispatches, input_bytes))| {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |f: f64| xs[(((xs.len() - 1) as f64) * f) as usize];
            EncoderLabelTiming {
                label,
                encoders: xs.len() as u64,
                dispatches,
                total_s: xs.iter().sum(),
                p50_s: q(0.5),
                p90_s: q(0.9),
                max_s: xs[xs.len() - 1],
                input_bytes,
            }
        })
        .collect();
    v.sort_by(|a, b| {
        b.total_s
            .partial_cmp(&a.total_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

/// One kernel's counter-sampled encoder time (see [`encoder_time_by_label`]).
#[derive(Clone, Debug)]
pub struct EncoderLabelTiming {
    pub label: String,
    pub encoders: u64,
    pub dispatches: u64,
    pub total_s: f64,
    pub p50_s: f64,
    pub p90_s: f64,
    pub max_s: f64,
    /// Summed largest-input bytes: the weight bytes, for a GEMV.
    pub input_bytes: u64,
}

/// GPU time per pipeline label over the recorded command buffers: `(label,
/// buffers, total seconds)`, largest total first. Meaningful when every buffer
/// held one dispatch (`CANDLE_METAL_COMPUTE_PER_BUFFER=1`); otherwise a buffer
/// is charged to its last kernel and the table says so by its counts (a
/// 50-dispatch buffer per label is not a kernel's time). Buffers with no
/// label are reported under `<unlabelled>`.
pub fn gpu_time_by_label() -> Vec<(String, u64, f64)> {
    gpu_time_by_label_detailed()
        .into_iter()
        .map(|k| (k.label, k.buffers, k.total_s))
        .collect()
}

/// One label's command buffers, with the interval DISTRIBUTION beside the sum.
///
/// # Why the distribution and not just the total (lloom #727)
///
/// `GPUStartTime`/`GPUEndTime` bound a command buffer's **residency on the
/// GPU**, not its execution. A buffer whose last dispatch is a one-element
/// `copy2d_u32` measured **14.5 ms** here — four orders above the work —
/// because the host synchronises inside it and the interval covers the wait.
/// Summed, that appears as "13.3 ms/step of copy2d_u32" and the real
/// arithmetic (every GEMV, rmsnorm, conv and attention kernel: **0.55 ms/step
/// together**) disappears beneath an artifact.
///
/// The falsifier that identified it: the inflated label FOLLOWS THE READBACK
/// ARM. `copy2d_u32` carries it on `ring-gpubatch-nonblocking`, a GEMV that
/// costs 37 us elsewhere carries 9 907 us on `ring-nonblocking`, and the
/// `blocking` arm — which syncs outside the buffer — shows every buffer at
/// ~77 us with no outlier at all.
///
/// So the median is the honest per-buffer figure and the max says how much
/// wait one buffer absorbed. A reader comparing `p50` against `max` sees the
/// artifact directly instead of inferring it. The total is kept because it is
/// what a caller summing to a step time needs, but it is no longer offered
/// alone.
pub fn gpu_time_by_label_detailed() -> Vec<LabelTiming> {
    let rec = recorder().lock().unwrap();
    let mut map: HashMap<String, Vec<f64>> = HashMap::new();
    for r in &rec.records {
        let key = r
            .label
            .as_deref()
            .map_or_else(|| "<unlabelled>".to_string(), str::to_string);
        map.entry(key)
            .or_default()
            .push(r.interval.end - r.interval.start);
    }
    let mut v: Vec<LabelTiming> = map
        .into_iter()
        .map(|(label, mut xs)| {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |f: f64| xs[(((xs.len() - 1) as f64) * f) as usize];
            LabelTiming {
                label,
                buffers: xs.len() as u64,
                total_s: xs.iter().sum(),
                p50_s: q(0.5),
                p99_s: q(0.99),
                max_s: xs[xs.len() - 1],
            }
        })
        .collect();
    v.sort_by(|a, b| {
        b.total_s
            .partial_cmp(&a.total_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

/// One label's command-buffer intervals: the sum, and the distribution that
/// says whether the sum is execution or a host wait one buffer absorbed.
#[derive(Clone, Debug)]
pub struct LabelTiming {
    pub label: String,
    /// Command buffers whose last dispatch carried this label.
    pub buffers: u64,
    /// Sum of their intervals — what a step-time breakdown adds up.
    pub total_s: f64,
    /// The MEDIAN buffer's interval: the honest per-buffer figure.
    pub p50_s: f64,
    pub p99_s: f64,
    /// The widest single interval. `max >> p50` means one buffer held a wait,
    /// and this label's `total` is that wait, not this kernel's cost.
    pub max_s: f64,
}

/// Merge sorted intervals, returning (union duration, span from first start to last end).
///
/// Split out from [`snapshot`] so the overlap handling -- the part that is easy
/// to get wrong and the reason the union is reported at all -- is testable
/// without a GPU.
fn union_and_span(intervals: &mut [GpuInterval]) -> (f64, f64) {
    if intervals.is_empty() {
        return (0.0, 0.0);
    }
    intervals.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut union = 0.0f64;
    let mut cur = intervals[0];
    for iv in &intervals[1..] {
        if iv.start <= cur.end {
            if iv.end > cur.end {
                cur.end = iv.end;
            }
        } else {
            union += cur.end - cur.start;
            cur = *iv;
        }
    }
    union += cur.end - cur.start;

    let first_start = intervals[0].start;
    let last_end = intervals.iter().map(|i| i.end).fold(f64::MIN, f64::max);
    (union, last_end - first_start)
}

/// One stretch of GPU idle between two consecutive command buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdleGap {
    /// Seconds from the previous buffer's end (the running maximum, so an
    /// overlapped buffer cannot open a gap inside another's interval) to this
    /// buffer's start.
    pub idle: f64,
    /// The part of `idle` the host is answerable for: how long after the
    /// previous buffer ended this one was committed, clamped to `[0, idle]`.
    /// Zero when this buffer was already queued -- then the whole gap is the
    /// seam.
    pub starved: f64,
    /// The step this buffer (the one that ended the idle) was committed in.
    pub step: u64,
    /// The previous buffer's end and this buffer's start, GPU clock seconds.
    pub prev_end: f64,
    pub start: f64,
}

/// The idle gaps between consecutive command buffers, in start order.
///
/// Split out from [`snapshot`] so the attribution -- starved against seam, the
/// running-maximum end that keeps an overlapped buffer from opening a gap --
/// is testable without a GPU.
fn idle_gaps(records: &mut [CommandBufferRecord]) -> Vec<IdleGap> {
    records.sort_by(|a, b| {
        a.interval
            .start
            .partial_cmp(&b.interval.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut gaps = Vec::new();
    let mut prev_end = match records.first() {
        Some(r) => r.interval.end,
        None => return gaps,
    };
    for r in &records[1..] {
        if r.interval.start > prev_end {
            let idle = r.interval.start - prev_end;
            let starved = (r.committed - prev_end).clamp(0.0, idle);
            gaps.push(IdleGap {
                idle,
                starved,
                step: r.step,
                prev_end,
                start: r.interval.start,
            });
        }
        if r.interval.end > prev_end {
            prev_end = r.interval.end;
        }
    }
    gaps
}

/// A profiling window's totals.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub dispatches: u64,
    pub encoders: u64,
    pub command_buffers: u64,
    /// Command buffers that reported a usable interval. Lower than
    /// `command_buffers` when some encoded no GPU work, or when the caller
    /// snapshots before every completion handler has run.
    pub timed_command_buffers: u64,
    /// Sum of per-command-buffer GPU intervals. Over-counts when buffers overlap.
    pub gpu_busy_sum_s: f64,
    /// Union of those intervals: time during which at least one command buffer
    /// was executing. This is the figure to trust.
    pub gpu_busy_union_s: f64,
    /// From the first command buffer's start to the last one's end.
    pub gpu_span_s: f64,
    /// Dispatch counts by pipeline label, descending.
    pub by_label: Vec<(String, u64)>,
    /// `commit_swap_locked` calls, and the encoding-thread time they took.
    pub commits: u64,
    pub commit_ns: u64,
    /// The `commandBuffer()` creations inside those commits, encoding-thread time.
    pub create_cb_ns: u64,
    /// Inside those commits: ending the encoder, the residency commit, the
    /// command buffer's `commit` (see `record_commit_parts`).
    pub end_enc_ns: u64,
    pub residency_ns: u64,
    pub cb_commit_ns: u64,
    /// `flush_and_prune` calls, and the encoding-thread time they took.
    pub prunes: u64,
    pub prune_ns: u64,
    /// Cross-encoder fence-map reads, hits, and waits (see `FENCE_PROBES`).
    pub fence_probes: u64,
    pub fence_hits: u64,
    pub fence_waits: u64,
    /// Every timed command buffer, in GPU start order: interval, commit stamp,
    /// step. The raw series a dump or a plot reads (issue #702).
    pub command_buffer_records: Vec<CommandBufferRecord>,
    /// The idle between consecutive command buffers, attributed: the
    /// distribution behind `1 − busy/wall`, which the union alone reports as
    /// one sum (issue #702).
    pub idle_gaps: Vec<IdleGap>,
    /// `min(GPUStartTime − committed)` over the timed buffers, seconds. A
    /// buffer cannot start before it is committed, so a negative value says
    /// the two stamps are not in one clock and every `starved` figure is void.
    pub min_start_after_commit_s: f64,
    /// Every counter-sampled compute encoder that completed, in completion
    /// order (empty unless `counters_enabled()`).
    pub encoder_records: Vec<EncoderRecord>,
    /// Encoders opened unsampled: no free slot and no sample buffer to be had.
    pub encoder_samples_dropped: u64,
    /// Sample buffer allocations the device refused (each drops its encoder).
    pub counter_buffers_refused: u64,
    /// Metal's reason for the last refusal, with the buffers live at the time.
    pub counter_refusal: Option<String>,
    /// Sample buffers created since the process started (the device holds 32).
    pub counter_buffers_live: u64,
    /// Most command buffers holding sampled encoders at once, since the process started.
    pub encoder_buffers_in_flight_peak: u64,
    /// The longest sample buffer allocation, created or refused, in seconds.
    pub counter_alloc_max_s: f64,
    /// Encoders whose samples resolved to the error marker.
    pub encoder_samples_bad: u64,
    /// Encoders that ran no dispatch or whose samples resolved to zero (no
    /// work reached the GPU).
    pub encoder_samples_empty: u64,
    /// Command buffers holding sampled encoders that have not been resolved:
    /// in flight at the snapshot, or never given a completion handler.
    pub encoder_buffers_unresolved: u64,
    /// Why this device cannot sample at all, when the counters were asked for.
    /// A refused allocation is not this: it drops encoders and is counted above.
    pub counters_unavailable: Option<String>,
    /// Seconds per counter-sample unit, once one buffer has resolved.
    pub counter_unit_s: Option<f64>,
}

/// Take the current totals without clearing them.
pub fn snapshot() -> Snapshot {
    let rec = recorder().lock().unwrap();

    let mut records = rec.records.clone();
    let mut intervals: Vec<GpuInterval> = records.iter().map(|r| r.interval).collect();
    let sum: f64 = intervals.iter().map(|i| i.end - i.start).sum();
    let (union, span) = union_and_span(&mut intervals);
    let idle_gaps = idle_gaps(&mut records);
    let min_start_after_commit_s = records
        .iter()
        .map(|r| r.interval.start - r.committed)
        .fold(f64::INFINITY, f64::min);

    // `Snapshot` keeps `String` keys: it is taken once per window (a run, or a
    // token boundary under `--per-token`), never per dispatch, so the
    // allocation here is off the hot path by construction and callers that
    // format or serialise the result want an owned `String` anyway.
    let by_label = sorted_by_count(&rec.dispatches);
    drop(rec);

    let c = counters().lock();
    let encoder_records = c.records.clone();
    let (dropped, bad, empty, unresolved, unavailable, unit) = (
        c.dropped,
        c.bad,
        c.empty,
        c.by_cb.len() as u64,
        c.unavailable.clone(),
        c.unit_s,
    );
    let (refused, refusal, live, in_flight_peak, alloc_max_s) = (
        c.refused,
        c.refusal.clone(),
        c.slabs.len() as u64,
        c.in_flight_peak as u64,
        c.alloc_max_s,
    );
    drop(c);

    Snapshot {
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        encoders: ENCODERS.load(Ordering::Relaxed),
        command_buffers: COMMAND_BUFFERS.load(Ordering::Relaxed),
        timed_command_buffers: intervals.len() as u64,
        gpu_busy_sum_s: sum,
        gpu_busy_union_s: union,
        gpu_span_s: span,
        by_label,
        commits: COMMITS.load(Ordering::Relaxed),
        commit_ns: COMMIT_NS.load(Ordering::Relaxed),
        create_cb_ns: CREATE_CB_NS.load(Ordering::Relaxed),
        end_enc_ns: END_ENC_NS.load(Ordering::Relaxed),
        residency_ns: RESIDENCY_NS.load(Ordering::Relaxed),
        cb_commit_ns: CB_COMMIT_NS.load(Ordering::Relaxed),
        prunes: PRUNES.load(Ordering::Relaxed),
        prune_ns: PRUNE_NS.load(Ordering::Relaxed),
        fence_probes: FENCE_PROBES.load(Ordering::Relaxed),
        fence_hits: FENCE_HITS.load(Ordering::Relaxed),
        fence_waits: FENCE_WAITS.load(Ordering::Relaxed),
        command_buffer_records: records,
        idle_gaps,
        min_start_after_commit_s,
        encoder_records,
        encoder_samples_dropped: dropped,
        counter_buffers_refused: refused,
        counter_refusal: refusal,
        counter_buffers_live: live,
        encoder_buffers_in_flight_peak: in_flight_peak,
        counter_alloc_max_s: alloc_max_s,
        encoder_samples_bad: bad,
        encoder_samples_empty: empty,
        encoder_buffers_unresolved: unresolved,
        counters_unavailable: unavailable,
        counter_unit_s: unit,
    }
}

/// Clear all counters, so the next window starts from zero.
///
/// Call after a device synchronization: an in-flight command buffer's completion
/// handler fires later and would otherwise land in the next window.
pub fn reset() {
    DISPATCHES.store(0, Ordering::Relaxed);
    ENCODERS.store(0, Ordering::Relaxed);
    COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    COMMITS.store(0, Ordering::Relaxed);
    COMMIT_NS.store(0, Ordering::Relaxed);
    CREATE_CB_NS.store(0, Ordering::Relaxed);
    PRUNES.store(0, Ordering::Relaxed);
    PRUNE_NS.store(0, Ordering::Relaxed);
    FENCE_PROBES.store(0, Ordering::Relaxed);
    FENCE_HITS.store(0, Ordering::Relaxed);
    FENCE_WAITS.store(0, Ordering::Relaxed);
    STEP.store(0, Ordering::Relaxed);
    let mut rec = recorder().lock().unwrap();
    rec.records.clear();
    rec.dispatches.clear();
    drop(rec);
    // Pending command buffers are kept: one still in flight resolves into the
    // next window, as its `GpuInterval` does (see above).
    let mut c = counters().lock();
    c.records.clear();
    c.dropped = 0;
    c.refused = 0;
    c.refusal = None;
    c.bad = 0;
    c.empty = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two command buffers in flight together must not be double-counted --
    /// the whole reason the union is reported next to the sum.
    #[test]
    fn union_merges_overlapping_intervals() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
        ];
        let (union, span) = union_and_span(&mut ivs);
        // Sum would be 2 + 2 + 1 = 5; the occupied time is 3 + 1 = 4.
        assert_eq!(union, 4.0);
        assert_eq!(span, 6.0);
    }

    /// Disjoint intervals: union and sum agree, which is the case where the
    /// simpler figure would have been fine.
    #[test]
    fn union_equals_sum_when_disjoint() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 1.0,
            },
            GpuInterval {
                start: 2.0,
                end: 3.0,
            },
        ];
        let (union, span) = union_and_span(&mut ivs);
        assert_eq!(union, 2.0);
        assert_eq!(span, 3.0);
    }

    /// Input order must not matter; the sweep sorts first.
    #[test]
    fn union_is_order_independent() {
        let mut a = vec![
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
        ];
        let mut b = vec![
            GpuInterval {
                start: 0.0,
                end: 2.0,
            },
            GpuInterval {
                start: 1.0,
                end: 3.0,
            },
            GpuInterval {
                start: 5.0,
                end: 6.0,
            },
        ];
        assert_eq!(union_and_span(&mut a), union_and_span(&mut b));
    }

    /// A fully contained interval must not extend the union past the outer end.
    #[test]
    fn union_handles_containment() {
        let mut ivs = vec![
            GpuInterval {
                start: 0.0,
                end: 10.0,
            },
            GpuInterval {
                start: 2.0,
                end: 3.0,
            },
        ];
        let (union, _) = union_and_span(&mut ivs);
        assert_eq!(union, 10.0);
    }

    /// The counter sample's unit is read from the data: a sample equal to the
    /// buffer's start in nanoseconds is nanoseconds, one equal to it in mach
    /// ticks is ticks, and a third unit is refused rather than converted.
    #[test]
    fn the_sample_unit_is_the_one_that_lands_on_the_buffers_start() {
        let cb_start = 409_705.901_880_666_7_f64;
        let ns = (cb_start * 1e9) as u64 + 3_000;
        assert_eq!(sample_unit_s(ns, cb_start), Some(1e-9));
        let ticks = (cb_start / tick_s()) as u64 + 72;
        assert_eq!(sample_unit_s(ticks, cb_start), Some(tick_s()));
        let micros = (cb_start * 1e6) as u64;
        assert_eq!(sample_unit_s(micros, cb_start), None);
    }

    /// Containment counts an interval outside its buffer, one running
    /// backwards, and nothing else.
    #[test]
    fn containment_counts_what_escapes_its_buffer() {
        let r = |start: f64, end: f64| EncoderRecord {
            start,
            end,
            step: 1,
            label: None,
            dispatches: 1,
            input_bytes: 0,
            cb_start: 10.0,
            cb_end: 20.0,
        };
        let rs = [
            r(11.0, 12.0),
            r(10.0, 20.0),
            r(9.0, 12.0),
            r(15.0, 21.0),
            r(14.0, 13.0),
        ];
        assert_eq!(encoder_containment(&rs, 1e-6), 3);
    }

    /// Per-label rows over single-dispatch encoders, and one row for the rest.
    #[test]
    fn encoder_time_is_keyed_on_the_kernel_only_when_the_encoder_is_one_kernel() {
        let r = |label: &str, dispatches: u32, dt: f64, bytes: u64| EncoderRecord {
            start: 0.0,
            end: dt,
            step: 1,
            label: Some(Arc::from(label)),
            dispatches,
            input_bytes: bytes,
            cb_start: 0.0,
            cb_end: 1.0,
        };
        let rows = encoder_time_by_label(&[
            r("gemv_a", 1, 2e-4, 100),
            r("gemv_a", 1, 4e-4, 100),
            r("rmsnorm", 1, 1e-5, 4),
            r("gemv_a", 3, 9e-4, 300),
        ]);
        let get = |l: &str| rows.iter().find(|k| k.label == l).unwrap();
        assert_eq!(get("gemv_a").encoders, 2);
        assert_eq!(get("gemv_a").input_bytes, 200);
        assert!((get("gemv_a").total_s - 6e-4).abs() < 1e-12);
        assert_eq!(get("<encoders of several dispatches>").dispatches, 3);
        assert_eq!(
            rows[0].label, "<encoders of several dispatches>",
            "largest total first"
        );
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(union_and_span(&mut []), (0.0, 0.0));
        assert!(idle_gaps(&mut []).is_empty());
    }

    fn rec(start: f64, end: f64, committed: f64, step: u64) -> CommandBufferRecord {
        CommandBufferRecord {
            interval: GpuInterval { start, end },
            committed,
            step,
            label: None,
        }
    }

    /// The attribution issue #702 rests on: a buffer committed before the
    /// previous one ended leaves a pure seam; one committed after it ended
    /// leaves a gap the host is answerable for up to the moment it committed.
    #[test]
    fn a_gap_is_starved_only_past_the_commit() {
        let mut rs = vec![
            rec(0.0, 1.0, -0.5, 1),
            // queued at 0.5, before the first ended: the 0.1 idle is all seam.
            rec(1.1, 2.0, 0.5, 2),
            // committed at 2.3, after the second ended at 2.0: of the 0.5
            // idle, 0.3 is starved and 0.2 is the seam after the commit.
            rec(2.5, 3.0, 2.3, 3),
        ];
        let gaps = idle_gaps(&mut rs);
        assert_eq!(gaps.len(), 2);
        assert!((gaps[0].idle - 0.1).abs() < 1e-12);
        assert_eq!(gaps[0].starved, 0.0);
        assert_eq!(gaps[0].step, 2);
        assert!((gaps[1].idle - 0.5).abs() < 1e-12);
        assert!((gaps[1].starved - 0.3).abs() < 1e-12);
        assert_eq!(gaps[1].step, 3);
    }

    /// A commit later than the start it precedes is a clock-domain error, not
    /// a starvation: `starved` is clamped to the idle, never past it.
    #[test]
    fn starved_never_exceeds_the_idle() {
        let mut rs = vec![rec(0.0, 1.0, 0.0, 1), rec(1.2, 2.0, 5.0, 2)];
        let gaps = idle_gaps(&mut rs);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].starved - gaps[0].idle).abs() < 1e-12);
    }

    /// An overlapped buffer -- one that ends inside a longer one -- must not
    /// open a gap: the running maximum end, the same rule the union uses.
    #[test]
    fn a_contained_buffer_opens_no_gap() {
        let mut rs = vec![
            rec(0.0, 10.0, 0.0, 1),
            rec(1.0, 2.0, 0.5, 1),
            rec(10.5, 11.0, 9.0, 2),
        ];
        let gaps = idle_gaps(&mut rs);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].prev_end - 10.0).abs() < 1e-12);
        assert!((gaps[0].idle - 0.5).abs() < 1e-12);
    }

    /// Back-to-back and touching buffers leave nothing to record.
    #[test]
    fn touching_buffers_leave_no_gap() {
        let mut rs = vec![rec(0.0, 1.0, 0.0, 1), rec(1.0, 2.0, 0.5, 1)];
        assert!(idle_gaps(&mut rs).is_empty());
    }

    /// The census must not allocate per dispatch (issue #579).
    ///
    /// **`Arc::strong_count` is an absolute reference, not a shape**
    /// (`CONTRIBUTING.md` §3.1b rule 1): it pins the exact number of live
    /// handles, so it fails with a sentence a reader can act on rather than
    /// with *"they were not all equal"*. Two handles -- the caller's and the
    /// map's key -- at every count from the first dispatch onward, so a
    /// `String`-keyed map (which copies the bytes and leaves this at 1) and an
    /// `Arc`-cloning-per-call map (which climbs past 2) are both killed, in
    /// opposite directions.
    ///
    /// This is what a runtime test can carry here. That the *production* call
    /// site passes the handle through rather than rebuilding one is a property
    /// of the write shape and is pinned separately, by
    /// `the_census_takes_the_callers_handle` below.
    #[test]
    fn a_repeat_label_clones_no_handle_and_copies_no_bytes() {
        let mut map: HashMap<Arc<str>, u64> = HashMap::new();
        let label: Arc<str> = Arc::from("rmsnorm_f16");
        assert_eq!(Arc::strong_count(&label), 1, "only the caller holds it yet");

        bump_label(&mut map, &label);
        assert_eq!(
            Arc::strong_count(&label),
            2,
            "the first sighting clones the handle into the key: the caller's \
             plus the map's. A String key would copy the bytes and leave this \
             at 1"
        );

        // 674 more, which is one decode token at the measured dispatch count.
        for _ in 0..674 {
            bump_label(&mut map, &label);
        }
        assert_eq!(
            Arc::strong_count(&label),
            2,
            "a repeat must reuse the key already there: still exactly the \
             caller's handle and the map's, after 675 dispatches"
        );
        assert_eq!(map.get(&label).copied(), Some(675));
    }

    /// Two labels are two entries, so the map is not silently collapsing them.
    ///
    /// The control for the test above: without it, a `bump_label` that ignored
    /// its argument and counted everything under one key would satisfy the
    /// strong-count assertion perfectly.
    #[test]
    fn distinct_labels_are_counted_separately() {
        let mut map: HashMap<Arc<str>, u64> = HashMap::new();
        let a: Arc<str> = Arc::from("rmsnorm_f16");
        let b: Arc<str> = Arc::from("sdpa_vector_2pass_1");
        bump_label(&mut map, &a);
        bump_label(&mut map, &b);
        bump_label(&mut map, &a);
        assert_eq!(map.get(&a).copied(), Some(2));
        assert_eq!(map.get(&b).copied(), Some(1));
        assert_eq!(map.len(), 2);
    }

    /// Two `Arc<str>` with the same text but different allocations must land in
    /// one entry.
    ///
    /// `Arc<str>`'s `Hash`/`Eq` delegate to `str`, so this holds -- but it is
    /// the property the census depends on and it is not obvious from the type.
    /// A pipeline rebuilt from the cache hands out a different allocation with
    /// the same name, and if that made a second entry the per-label totals
    /// would silently split.
    #[test]
    fn equal_text_in_two_allocations_is_one_entry() {
        let mut map: HashMap<Arc<str>, u64> = HashMap::new();
        let first: Arc<str> = Arc::from("gemv_f16");
        let second: Arc<str> = Arc::from("gemv_f16");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "the fixture must use two distinct allocations or it tests nothing"
        );
        bump_label(&mut map, &first);
        bump_label(&mut map, &second);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&first).copied(), Some(2));
    }

    /// The production call site hands the census the handle it already holds,
    /// rather than building a `String` or a fresh `Arc` from a `&str`.
    ///
    /// **A source-level census, for #507's reason**: this is a property of the
    /// *write shape*, and a runtime test observes only the counts, which are
    /// the same whether or not the handle was rebuilt.
    ///
    /// **Both of §9.2p's repairs are needed and neither alone suffices**, which
    /// that section measured and this test reproduced on its first run. The
    /// needles are **assembled by concatenation**, because a literal needle is
    /// a string the file it searches contains -- and the searched file has its
    /// **comment lines stripped**, because the comment explaining *why the old
    /// form was removed* names the old form. The first run failed on exactly
    /// that: `encoder.rs`'s own justification for this change, matched as
    /// though it were the change's absence. *"A better needle is still a
    /// needle"* -- assembly escapes the first hole, and only a comment filter
    /// escapes the second.
    #[test]
    fn the_census_takes_the_callers_handle() {
        // Comment lines removed, so a sentence *about* a pattern cannot be
        // read as an instance of it.
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
        let pipeline = code_only(include_str!("compute_pipeline.rs"));

        // The re-allocating form this replaced: `.name()` (a `&str`) fed to
        // `Arc::from`. Assembled so this assertion does not match itself.
        let rebuilt = ["pipeline.name()", ".map(", "Arc::from)"].concat();
        assert!(
            !encoder.contains(&rebuilt),
            "note_pipeline must clone the pipeline's existing Arc<str>, not \
             build a new one from &str: that allocated once per dispatch \
             (issue #579)"
        );

        let handle = ["fn name_", "handle(&self)"].concat();
        assert!(
            pipeline.contains(&handle),
            "ComputePipeline must expose its name as a shared handle; \
             note_pipeline has no other way to avoid the re-allocation"
        );

        // Non-vacuity: the needles must be capable of matching something, or
        // two absences read exactly like two passes (§15.1a's VACUOUS class).
        // This is the positive control for both assertions above -- the call
        // site the first requires absent in its old form, found in its new one,
        // in the same filtered text.
        let present = ["pipeline.name_", "handle()"].concat();
        assert!(
            encoder.contains(&present),
            "the search is over the wrong file or the call site was renamed: \
             this assertion is what stops the two above passing vacuously"
        );

        // And the filter must not have eaten the code. A `code_only` returning
        // nothing would satisfy every "must not contain" assertion above, so
        // the filter needs its own control: the searched text must still hold
        // the function whose body the census is about.
        assert!(
            encoder.contains(&["fn note_", "pipeline(&self"].concat()),
            "the comment filter removed the code it was meant to keep"
        );
    }
}
