use super::*;
use std::sync::Mutex;

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
pub(super) fn record_dispatch(label: Option<&Arc<str>>, input_bytes: usize) {
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
pub(super) fn bump_label(map: &mut HashMap<Arc<str>, u64>, label: &Arc<str>) {
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
pub(super) fn tick_s() -> f64 {
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
pub(super) fn sample_unit_s(sample: u64, cb_start: f64) -> Option<f64> {
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

/// The diagnostic record series, without snapshot aggregation.
pub(super) fn command_buffer_records() -> Vec<CommandBufferRecord> {
    recorder().lock().unwrap().records.clone()
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
        timed_command_buffers: intervals.len() as u64,
        gpu_busy_sum_s: sum,
        gpu_busy_union_s: union,
        gpu_span_s: span,
        by_label,
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
        ..super::scalar_snapshot()
    }
}

pub(super) fn reset() {
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
pub(super) fn record_command_buffer(start: f64, end: f64, committed: f64, step: u64) {
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
