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

#[derive(Default)]
struct Recorder {
    /// GPU intervals of every command buffer that has completed.
    intervals: Vec<GpuInterval>,
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

/// Record that a compute encoder was opened.
#[inline]
pub fn record_encoder() {
    if !enabled() {
        return;
    }
    ENCODERS.fetch_add(1, Ordering::Relaxed);
}

/// Record a completed command buffer's GPU interval.
#[inline]
pub fn record_command_buffer(start: f64, end: f64) {
    if !enabled() {
        return;
    }
    COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
    // A command buffer that encoded no GPU work reports zeros; recording it
    // would drag the union's lower bound back to the epoch.
    if start <= 0.0 || end <= start {
        return;
    }
    recorder()
        .lock()
        .unwrap()
        .intervals
        .push(GpuInterval { start, end });
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
}

/// Take the current totals without clearing them.
pub fn snapshot() -> Snapshot {
    let rec = recorder().lock().unwrap();

    let mut intervals = rec.intervals.clone();
    let sum: f64 = intervals.iter().map(|i| i.end - i.start).sum();
    let (union, span) = union_and_span(&mut intervals);

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
    let mut rec = recorder().lock().unwrap();
    rec.intervals.clear();
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
