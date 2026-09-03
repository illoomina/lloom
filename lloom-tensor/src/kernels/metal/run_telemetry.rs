//! Memory-class and event telemetry, behind the `run-telemetry` feature.
//!
//! # What this answers
//!
//! `DESIGN.md` §9.1 names five memory classes — weights, conv state, KV,
//! activations and scratch — plus the residency set, which §6.3c establishes as
//! a **third lifetime participant** after it panicked the machine. Their sizes
//! are individually reachable (`pool_occupancy`, `arena`, `ScratchPlan`,
//! `residency_set_len`) and **nothing has ever put them on one timeline**.
//!
//! The diagnostic value is the *shape*: a curve that climbs without plateauing
//! as `n` or `kv_len` rises is a leak or an allocator defect. That shape is
//! already in this project's history twice, seen late both times:
//!
//! * §3.4a-i: peak `phys_footprint` 11.21 GB at 60 tokens against 14.90 GB at
//!   200 — ~3.7 GB per 140 tokens against a KV cache of 16 KB/token, so it is
//!   the pool. Recorded on #54 and not investigated.
//! * §3.4a-ii: our own process group climbing **0.53 → 14.64 GB without
//!   plateauing** across one run.
//! * §6.3c: at the kernel panic, wired was **56.14 GB of 64 GB with 0.45 GB
//!   free**. #163 had to reconstruct memory state from a panic log afterwards.
//!
//! A per-class series against token index would have shown all three while the
//! run was still going.
//!
//! # Why a feature and not an environment variable
//!
//! The existing `CANDLE_METAL_PROFILE` is env-gated, and §6.4a records what that
//! costs when a hook is not free: its per-dispatch kernel inventory allocates a
//! `String` per dispatch — 675 per token, ~3.6 % of forward-pass CPU, *the same
//! order as the thing being measured* — which is why it was split behind a
//! second variable. An env check is a cached bool, but the code around it is
//! still compiled, and any state it touches is still allocated.
//!
//! `run-telemetry` composes with a **release** build and compiles to nothing
//! when off: every entry point below is a `#[cfg]` pair whose off-arm is an
//! empty inline function, following `debug-labels`'s shape (`utils.rs`). The
//! off-cost claim is therefore structural rather than statistical — there is no
//! call to elide, no branch to predict, and no counter to contend on.
//!
//! # Why sampling is not on the per-token path
//!
//! #166 measured the obvious form of "notify on every buffer" at **+0.062
//! ms/token even batched**, against §11.2's whole 6.1 % non-GPU budget, because
//! `removeAllocation` puts a `commit()` on the per-token path (§6.7 L4, third
//! corollary). What made that guard affordable was separating the *record* from
//! the *call*.
//!
//! Same separation here, and it is why [`sample`] is called by the **harness**
//! at a chosen cadence rather than by the backend on every allocation: a
//! sample is a handful of counter reads under locks candle already takes, and it
//! happens tens of times per run rather than 675 times per token. The event
//! markers ([`mark`]) are the exception — they fire where the event does — and
//! they are a `Vec` push of two integers and a `&'static str`, with no
//! formatting and no allocation.

// Gated with the feature: off, this module is entry points with empty bodies,
// so importing synchronisation primitives would warn. That the off arm needs
// *no* imports at all is itself the clearest statement of what it compiles to.
#[cfg(feature = "run-telemetry")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "run-telemetry")]
use std::sync::{Mutex, OnceLock};

/// A point on the timeline, in `CLOCK_UPTIME_RAW` nanoseconds.
///
/// Domain A in `DESIGN.md` §3.4b — the same clock `lloom-sample` brackets its
/// IOReport reads in and the same one `GPUStartTime`/`GPUEndTime` live in.
/// Sharing it is what lets these series be overlaid on the sampler's without a
/// reconstruction. **Not `CLOCK_MONOTONIC_RAW`**, which on Darwin includes time
/// asleep where `CLOCK_UPTIME_RAW` does not — the opposite of Linux, and 39.38
/// hours of it on this machine.
#[cfg(feature = "run-telemetry")]
#[inline]
pub fn now_ns() -> u64 {
    // SAFETY: `clock_gettime_nsec_np` takes a clock id and returns a `u64`. It
    // reads no memory we own and cannot fail for a compile-time-known id.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

#[cfg(feature = "run-telemetry")]
const CLOCK_UPTIME_RAW: u32 = 8;

#[cfg(feature = "run-telemetry")]
extern "C" {
    fn clock_gettime_nsec_np(clock_id: u32) -> u64;
}

/// The memory classes of `DESIGN.md` §9.1, plus the residency set.
///
/// The residency set is here because §6.3c makes it a lifetime participant
/// rather than a curiosity: six insert sites against two remove sites, both
/// test-only, and a machine panic when teardown asked the kernel to remove
/// objects already gone. §6.7 L4's second corollary states the general rule —
/// *"which mechanisms hold a reference to this buffer" is a question a liveness
/// change must answer exhaustively* — and a class that is not plotted is one
/// nobody asks about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemClass {
    /// The buffer pool: live, free-listed, and awaiting GPU completion.
    Pool,
    /// The activation arena (§9.2), when one is installed.
    Arena,
    /// The kernel scratch class (§9.1a) — FlashDecoding partials today.
    Scratch,
    /// The KV cache (§10), whose size is the one that grows with `kv_len`.
    Kv,
    /// `MTLResidencySet` membership (§6.3c, §6.3d).
    Residency,
}

impl MemClass {
    pub fn as_str(self) -> &'static str {
        match self {
            MemClass::Pool => "pool",
            MemClass::Arena => "arena",
            MemClass::Scratch => "scratch",
            MemClass::Kv => "kv",
            MemClass::Residency => "residency",
        }
    }
}

/// Events worth marking on the timeline.
///
/// Chosen from the ones this project has had to reconstruct after the fact.
/// §10.2e's `{532, 598}` dispatch count is a **compaction** — the sliding conv
/// ring compacting when its window runs out — and the issue notes a plot would
/// have shown it immediately, where instead it appears as a bimodal dispatch
/// count that had to be explained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// A `MTLCommandBuffer` was created.
    CommandBufferCreate,
    /// A command buffer completed (its completion handler ran).
    CommandBufferComplete,
    /// A compute encoder session opened. §9.2f: candle opens one every
    /// `CANDLE_METAL_COMPUTE_PER_BUFFER` dispatches — 14 per decode token — and
    /// §9.2j found the boundaries fall *inside* layers rather than between them,
    /// which is arithmetic rather than structure and is invisible without a mark.
    EncoderSessionBegin,
    EncoderSessionEnd,
    /// The activation arena was installed (§9.2c).
    ArenaInstall,
    /// An ICB plan was built (§11.3l).
    IcbPlanBuild,
    /// A conv-state ring compaction (§10.2e). The event behind `{532, 598}`.
    Compaction,
    /// A decode token boundary, so every other series can be read against the
    /// token index the issue asks for.
    TokenBoundary,
}

impl Event {
    pub fn as_str(self) -> &'static str {
        match self {
            Event::CommandBufferCreate => "cb_create",
            Event::CommandBufferComplete => "cb_complete",
            Event::EncoderSessionBegin => "encoder_begin",
            Event::EncoderSessionEnd => "encoder_end",
            Event::ArenaInstall => "arena_install",
            Event::IcbPlanBuild => "icb_plan_build",
            Event::Compaction => "compaction",
            Event::TokenBoundary => "token_boundary",
        }
    }
}

/// One memory-class reading.
#[derive(Debug, Clone, Copy)]
pub struct MemSample {
    pub t_ns: u64,
    /// The decode token this reading belongs to, so the series can be plotted
    /// against token index rather than against wall time alone.
    pub token: u64,
    pub class: MemClass,
    pub live_bytes: u64,
    pub live_count: u64,
    /// Bytes released by the CPU and awaiting GPU completion. §6.3b's stranding
    /// — 11.6 buffers per token, 5231 → 13629 MB at 400 tokens — lives here and
    /// is invisible in a single "live bytes" figure.
    pub pending_bytes: u64,
}

/// One event marker.
#[derive(Debug, Clone, Copy)]
pub struct EventMark {
    pub t_ns: u64,
    pub token: u64,
    pub event: Event,
    /// An event-specific integer: a dispatch index, a plan size, a slot count.
    pub detail: u64,
}

#[cfg(feature = "run-telemetry")]
#[derive(Default)]
struct Recorder {
    mem: Vec<MemSample>,
    events: Vec<EventMark>,
}

#[cfg(feature = "run-telemetry")]
fn recorder() -> &'static Mutex<Recorder> {
    static R: OnceLock<Mutex<Recorder>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(Recorder {
            // Pre-reserved so a mid-run push does not reallocate. A decode run
            // of 1500 tokens marks a few thousand events at the cadences below.
            mem: Vec::with_capacity(4096),
            events: Vec::with_capacity(16384),
        })
    })
}

/// Runtime switch *within* a feature-enabled build.
///
/// The feature decides whether the code exists; this decides whether an existing
/// build records. Both, because §2.4 requires an instrument to be shown to have
/// engaged — #69's determinism gate consumed the `OnceLock` guarding its env
/// switch and ran both arms in the default configuration, reporting a passing
/// digest for the unchanged path. A build that *can* record and did not is a
/// different state from one that cannot, and [`engaged`] reports which.
#[cfg(feature = "run-telemetry")]
static ENABLED: AtomicBool = AtomicBool::new(false);

/// The current decode token, set by the harness at each boundary.
#[cfg(feature = "run-telemetry")]
static TOKEN: AtomicU64 = AtomicU64::new(0);

/// Counts every sample and mark accepted, so a run can prove the instrument ran.
#[cfg(feature = "run-telemetry")]
static ACCEPTED: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// The API. Each entry point is a `#[cfg]` pair; the off-arm is an empty inline
// function taking the same arguments, so a caller compiles identically with the
// feature off and the optimiser removes the call entirely.
// ---------------------------------------------------------------------------

/// Turns recording on or off in a feature-enabled build.
#[cfg(feature = "run-telemetry")]
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn set_enabled(_on: bool) {}

/// Whether this build can record at all.
pub const fn compiled() -> bool {
    cfg!(feature = "run-telemetry")
}

/// Whether recording is on *and* has accepted anything.
///
/// The second half is what makes it an engagement proof rather than a flag read
/// back to itself (§2.4, #69).
#[cfg(feature = "run-telemetry")]
pub fn engaged() -> bool {
    ENABLED.load(Ordering::Relaxed) && ACCEPTED.load(Ordering::Relaxed) > 0
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn engaged() -> bool {
    false
}

/// Advances the token counter. Called at each decode-step boundary.
#[cfg(feature = "run-telemetry")]
pub fn set_token(token: u64) {
    TOKEN.store(token, Ordering::Relaxed);
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn set_token(_token: u64) {}

/// Records a memory-class reading.
///
/// Called by the harness between tokens, not by the allocator on every
/// allocation — see the module docs for why.
#[cfg(feature = "run-telemetry")]
pub fn sample(class: MemClass, live_bytes: u64, live_count: u64, pending_bytes: u64) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let s = MemSample {
        t_ns: now_ns(),
        token: TOKEN.load(Ordering::Relaxed),
        class,
        live_bytes,
        live_count,
        pending_bytes,
    };
    if let Ok(mut r) = recorder().lock() {
        r.mem.push(s);
        ACCEPTED.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn sample(_class: MemClass, _live_bytes: u64, _live_count: u64, _pending_bytes: u64) {}

/// Marks an event.
///
/// A `Vec` push of two integers and a discriminant. No formatting, no
/// allocation, no syscall beyond the clock read — which is 41.67 ns at the
/// 24 MHz timebase floor (§3.4b).
#[cfg(feature = "run-telemetry")]
pub fn mark(event: Event, detail: u64) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let m = EventMark {
        t_ns: now_ns(),
        token: TOKEN.load(Ordering::Relaxed),
        event,
        detail,
    };
    if let Ok(mut r) = recorder().lock() {
        r.events.push(m);
        ACCEPTED.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn mark(_event: Event, _detail: u64) {}

/// Everything recorded so far, as `lloom-sample`-compatible JSONL.
///
/// Long format — `{run_id, t0_ns, t1_ns, signal, value, unit}` — deliberately
/// identical to `lloom-sample`'s rows, so the two files concatenate and the
/// existing plotter reads both. `lloom-sample`'s own decision #5 gives the
/// reason for the format: signals will be added, and a wide schema needs a
/// migration each time one is.
///
/// `t0 == t1` here because these readings are instantaneous, where an IOReport
/// counter delta spans a bracket. Keeping the same two fields rather than
/// collapsing to one is what makes the rows interleavable.
///
/// Called **at exit**, never during the measured window.
#[cfg(feature = "run-telemetry")]
pub fn to_jsonl(run_id: &str) -> String {
    let Ok(r) = recorder().lock() else {
        return String::new();
    };
    let mut out = String::with_capacity(r.mem.len() * 160 + r.events.len() * 120);
    for s in r.mem.iter() {
        for (suffix, value, unit) in [
            ("live_bytes", s.live_bytes, "bytes"),
            ("live_count", s.live_count, "count"),
            ("pending_bytes", s.pending_bytes, "bytes"),
        ] {
            out.push_str(&format!(
                "{{\"run_id\":\"{run_id}\",\"t0_ns\":{t},\"t1_ns\":{t},\
                 \"signal\":\"mem/{cls}/{suffix}\",\"value\":{value},\"unit\":\"{unit}\",\
                 \"token\":{tok}}}\n",
                t = s.t_ns,
                cls = s.class.as_str(),
                tok = s.token,
            ));
        }
    }
    for e in r.events.iter() {
        out.push_str(&format!(
            "{{\"run_id\":\"{run_id}\",\"t0_ns\":{t},\"t1_ns\":{t},\
             \"signal\":\"event/{ev}\",\"value\":{detail},\"unit\":\"marker\",\
             \"token\":{tok}}}\n",
            t = e.t_ns,
            ev = e.event.as_str(),
            detail = e.detail,
            tok = e.token,
        ));
    }
    out
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn to_jsonl(_run_id: &str) -> String {
    String::new()
}

/// How many samples and marks were accepted. Zero in a build with the feature
/// off, which is what makes the off-cost claim checkable from outside.
#[cfg(feature = "run-telemetry")]
pub fn accepted() -> u64 {
    ACCEPTED.load(Ordering::Relaxed)
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn accepted() -> u64 {
    0
}

/// Clears the recorder, so warmup does not pollute the steady-state series.
#[cfg(feature = "run-telemetry")]
pub fn reset() {
    if let Ok(mut r) = recorder().lock() {
        r.mem.clear();
        r.events.clear();
    }
    ACCEPTED.store(0, Ordering::Relaxed);
}

#[cfg(not(feature = "run-telemetry"))]
#[inline(always)]
pub fn reset() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The off-build must be inert, and this is the assertion that says so from
    /// outside rather than by reading the `#[cfg]`s.
    #[test]
    fn the_feature_off_build_records_nothing() {
        if compiled() {
            return; // The on-build is exercised by the tests below.
        }
        set_enabled(true);
        set_token(7);
        sample(MemClass::Pool, 1234, 5, 6);
        mark(Event::Compaction, 1);
        assert_eq!(accepted(), 0, "an off build must not record");
        assert!(!engaged());
        assert!(to_jsonl("r").is_empty());
    }

    #[cfg(feature = "run-telemetry")]
    #[test]
    fn samples_and_marks_land_on_one_clock_with_a_token_index() {
        reset();
        set_enabled(true);
        set_token(42);
        sample(MemClass::Pool, 1000, 3, 200);
        mark(Event::Compaction, 598);
        let out = to_jsonl("run-x");

        assert!(
            out.contains("\"signal\":\"mem/pool/live_bytes\",\"value\":1000"),
            "{out}"
        );
        assert!(
            out.contains("\"signal\":\"mem/pool/pending_bytes\",\"value\":200"),
            "{out}"
        );
        assert!(
            out.contains("\"signal\":\"event/compaction\",\"value\":598"),
            "{out}"
        );
        // Every row carries the token index, which is what the issue asks the
        // memory series to be plotted against.
        assert!(out.contains("\"token\":42"), "{out}");
        assert!(engaged(), "the instrument must be able to prove it ran");
        reset();
        set_enabled(false);
    }

    #[cfg(feature = "run-telemetry")]
    #[test]
    fn recording_is_off_until_enabled() {
        reset();
        set_enabled(false);
        sample(MemClass::Arena, 1, 1, 1);
        mark(Event::ArenaInstall, 0);
        assert_eq!(accepted(), 0);
        assert!(
            !engaged(),
            "a build that can record but did not is not engaged"
        );
    }

    #[cfg(feature = "run-telemetry")]
    #[test]
    fn every_class_and_event_has_a_distinct_name() {
        let classes = [
            MemClass::Pool,
            MemClass::Arena,
            MemClass::Scratch,
            MemClass::Kv,
            MemClass::Residency,
        ];
        let mut names: Vec<&str> = classes.iter().map(|c| c.as_str()).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "two memory classes share a name");

        let events = [
            Event::CommandBufferCreate,
            Event::CommandBufferComplete,
            Event::EncoderSessionBegin,
            Event::EncoderSessionEnd,
            Event::ArenaInstall,
            Event::IcbPlanBuild,
            Event::Compaction,
            Event::TokenBoundary,
        ];
        let mut ev: Vec<&str> = events.iter().map(|e| e.as_str()).collect();
        ev.sort_unstable();
        let m = ev.len();
        ev.dedup();
        assert_eq!(ev.len(), m, "two events share a name");
    }

    #[cfg(feature = "run-telemetry")]
    #[test]
    fn the_clock_is_uptime_not_wall_clock() {
        let a = now_ns();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = now_ns();
        assert!(b > a);
        let epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        // CLOCK_UPTIME_RAW is time since boot, nowhere near ns since 1970.
        assert!(
            b < epoch_ns / 2,
            "looks like a wall clock (DESIGN.md §3.4b)"
        );
    }
}
