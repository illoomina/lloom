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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Whether profiling was requested, read once from the environment.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("CANDLE_METAL_PROFILE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
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
#[inline]
pub fn record_dispatch(label: Option<&Arc<str>>) {
    if !enabled() {
        return;
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut rec = recorder().lock().unwrap();
    bump_label(&mut rec.dispatches, label.unwrap_or_else(|| unnamed()));
    rec.last_label.0 = label.cloned();
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
    let mut by_label: Vec<(String, u64)> = rec
        .dispatches
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    by_label.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

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
        prunes: PRUNES.load(Ordering::Relaxed),
        prune_ns: PRUNE_NS.load(Ordering::Relaxed),
        fence_probes: FENCE_PROBES.load(Ordering::Relaxed),
        fence_hits: FENCE_HITS.load(Ordering::Relaxed),
        fence_waits: FENCE_WAITS.load(Ordering::Relaxed),
        command_buffer_records: records,
        idle_gaps,
        min_start_after_commit_s,
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
