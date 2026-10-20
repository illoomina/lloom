//! Where does an LFM2 decode token's time actually go? CPU or GPU, and why.
//!
//! Measurement harness for lloom issue #7 / `DESIGN.md` §3.4, §6.6, §11.2,
//! §16 P0 #1 and #4.
//!
//! Three numbers, measured in one run so they describe the same execution:
//!
//! * **wall time per decode token** — measured independently rather than
//!   inherited from the ~27 ms/token recorded in `DESIGN.md` §6.6.
//! * **GPU busy time per decode token** — the union of every command buffer's
//!   `GPUStartTime`..`GPUEndTime` in the window, via `CANDLE_METAL_PROFILE=1`.
//! * **dispatch count per decode token** — counted, not inherited from §11.2's
//!   ~240 estimate.
//!
//! `wall - gpu_busy` is the time the GPU was not executing. In a serial decode
//! loop with nothing to overlap against, that is CPU-side encode, submission and
//! readback: the answer to §16 P0 #1.
//!
//! # Why prefill is measured separately
//!
//! Prefill runs one forward pass over the whole prompt and decode runs one per
//! token. Averaging them together makes ms/token a function of prompt length,
//! which is how a decode figure ends up describing something else. The prompt's
//! forward pass is timed into its own bucket and excluded from the decode
//! average.
//!
//! # Why a warmup token is excluded
//!
//! The first decode token pays for pipeline compilation, buffer-pool growth and
//! page faults on freshly mmap'd weights. Those are real costs, but they are
//! one-time, and a steady-state figure is what the roofline in §16 P0 #4 is
//! being compared against. `--warmup` tokens are timed and reported, then left
//! out of the steady-state average.
//!
//! ```bash
//! CANDLE_METAL_PROFILE=1 cargo run --release --features metal \
//!   --example lfm2-decode-profile -- --n 200
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_examples::lfm2_setup::{default_model_dir, parse_config, tensor_names, weight_files};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use lloom::generation::poll::PollMechanism;
use lloom::generation::token_ring::TokenRing;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{
    AttnImpl, Config, ConvState, KvAppend, LayerType, Model, Shape as Lfm2Shape, State,
};
use lloom::nn::ops::FlashScratchSizing;
use lloom::nn::ops::KvQuant;
use lloom::nn::VarBuilder;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use tokenizers::Tokenizer;

/// The serialized run record (lloom issue #171).
///
/// Written **at exit**, beside the `RESULT` line. The only thing it does inside
/// a timed window is stamp each finished token with `CLOCK_UPTIME_RAW`, which is
/// what lets the memory timeline be plotted against token index instead of
/// against a reconstruction.
mod run_record;

/// The front end's per-token seam, reproduced (lloom issue #412).
///
/// `--seam` turns the decode loop into ambrogio's streaming shape — detokenize
/// the pending run, allocate a `String`, `blocking_send` it into a bounded
/// channel a real consumer drains — so the thing #412 asks the price of is a
/// slot in a rotation rather than a thing that has to be argued about.
mod seam;

/// One `sysctl -n` read, for the machine fields of the run record.
fn sysctl_str(key: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The 1-minute load average.
///
/// Recorded at the start **and** the end of a run, because one reading cannot
/// show a build starting midway through -- `lloom-probe`'s `Conditions` makes
/// the same split for the same reason. `.bench/README.md` §1.2: a number taken
/// on this machine without stating contention is not interpretable, and #40's
/// null control read **+65.8 %** with two other agents building.
fn load_average() -> Option<f64> {
    extern "C" {
        fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
    }
    let mut a = [0.0f64; 3];
    // SAFETY: `getloadavg` writes at most `nelem` doubles into the buffer, and
    // we pass the true length of a stack array we own.
    let n = unsafe { getloadavg(a.as_mut_ptr(), 3) };
    (n > 0).then_some(a[0])
}

/// The process's own resident and physical footprint, from `proc_pid_rusage`
/// (`DESIGN.md` §3.4a-i): `phys_footprint` is the quantity the OS bills the
/// process for, so it spans every allocation however it was made — the one
/// number that can show growth the five classes do not name (§9.5f candidate
/// 3). Shared with `lfm2-chat --mem-sample`.
use candle_examples::lfm2_setup::proc_footprint;

/// Write the by-class timeline recorded so far, truncating and rewriting.
///
/// # Why rewrite rather than append
///
/// `run_telemetry` accumulates into one `Vec` and exposes `to_jsonl` over the
/// whole of it; there is no "since last drain" cursor, and adding one would be a
/// change to a module #171 built and #205 is only wiring. Rewriting is O(samples
/// so far) per call, which at `--progress-every 25` over a 3 000-token run is 120
/// rewrites of a file that ends at ~200 KB -- **outside the per-token timing
/// window**, and bounded by the same flag that decides how often a sample is
/// taken at all.
///
/// The alternative -- writing only at exit -- is what the first version did, and
/// the `Cat` arm's OOM produced a **zero-byte** file because `?` returns before
/// the drain. A timeline that is absent from the run that fails is the
/// instrument missing the one event it was built for.
///
/// `rt_written` carries the last byte count purely so the final report can say
/// whether anything reached disk, without re-reading the file.
#[cfg(feature = "metal")]
fn flush_run_telemetry(args: &Args, rt_written: &mut usize) -> Result<()> {
    let Some(path) = args.run_telemetry_jsonl.as_ref() else {
        return Ok(());
    };
    use candle::kernels::metal::run_telemetry as rt;
    let run_id = args.run_id.clone().unwrap_or_else(|| "lfm2".to_string());
    let body = rt::to_jsonl(&run_id);
    *rt_written = body.len();
    std::fs::write(path, body)
        .with_context(|| format!("writing run telemetry to {}", path.display()))?;
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn flush_run_telemetry(args: &Args, _rt_written: &mut usize) -> Result<()> {
    validate_run_telemetry_request(args)
}

fn run_telemetry_compiled() -> bool {
    #[cfg(feature = "metal")]
    {
        candle::kernels::metal::run_telemetry::compiled()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Cold request validation shared by the actual entry and no-Metal writer.
/// It must precede shape_key, which reads checkpoint config.
fn validate_run_telemetry_request(args: &Args) -> Result<()> {
    if args.run_telemetry_jsonl.is_some() {
        if !run_telemetry_compiled() {
            anyhow::bail!(
                "--run-telemetry-jsonl needs the `metal-run-telemetry` feature, and this \
                 binary was built without it.\n  \
                 rebuild: cargo build --release --features metal-run-telemetry \
                 --example lfm2-decode-profile\n  \
                 (candle-examples gained that feature in lloom #205; before it, \
                 candle-core declared `metal-run-telemetry` and nothing above it did, \
                 so no harness could enable it at all.)"
            );
        }
        anyhow::ensure!(!args.cpu,
            "--run-telemetry-jsonl records Metal memory classes; select a Metal device (omit --cpu)");
    }
    Ok(())
}

/// KV bytes per token, all layers, from `DESIGN.md` §5.6's **geometry**.
///
/// `2 (K,V) x 8 kv_heads x 64 head_dim x 2 B` = 2048 B per attention layer, and
/// LFM2 has **8** attention layers of 30 (§5.3), so 16 384 B.
///
/// **Derived here rather than quoted from §5.6's table, because that table was
/// wrong.** #164 found it computed with **16 000** B/token where the line
/// directly above it derives 16 384 -- `0.524288 = 16000 x 32768 / 1e9`
/// exactly, which is what identified the arithmetic rather than leaving it as a
/// rounding question. The error is 2.4 %: 50 MB at B=1 x 128k, and 1.61 GB at
/// B=32 x 128k. Quoting the table here would have re-imported a corrected
/// defect into a new instrument.
const KV_BYTES_PER_TOKEN: u64 = 2 * 8 * 64 * 2 * 8;

/// One reading of every participant in a buffer's lifetime, at one token.
///
/// # Why all of these and not just the pool
///
/// `DESIGN.md` §6.3c establishes that a buffer's existence has **three**
/// participants — the CPU handle, the GPU completion epoch, and the residency
/// set — and §6.7 L4's third corollary is that a liveness question must be
/// asked of *every* participant rather than the ones in view. #204 asked the
/// pool and got a flat answer. These are the others:
///
/// | field | answers |
/// |---|---|
/// | `pool_*` | what the allocator holds — #204's flat quantity, reproduced so the arms are comparable |
/// | `residency` | §6.3c's third participant, which has membership tracking since #167 and has never been read at long context |
/// | `device_allocated` | `MTLDevice.currentAllocatedSize` — **the driver's own count**, which is not derived from any of our bookkeeping |
/// | `phys_footprint` | what the OS bills the process, which spans everything above and anything outside it |
/// | `allocations` / `evicted` | the create/destroy rate issue #206 predicts degenerates under `Cat` |
///
/// **`device_allocated` beside `phys_footprint` is the discriminator issue #206
/// asks for in its third acceptance item.** `kIOGPUCommandBufferCallbackErrorOutOfMemory`
/// is a *command buffer* error, so the failure need not be bytes at all. If the
/// driver's own allocated size tracks the pool while the footprint climbs, the
/// growth is outside Metal's allocations; if both climb together, it is bytes
/// and the question is whose. Neither reading is available from a pool counter.
/// A `[u64; N]` as a JSON array, for the band rows.
///
/// Hand-rolled because this file writes its JSON by `format!` rather than
/// through a serializer, and adding one for three fields would change how every
/// other field is emitted -- which would invalidate the committed series' shape
/// for a cosmetic gain.
fn json_u64_array(xs: &[u64]) -> String {
    let mut s = String::with_capacity(xs.len() * 8 + 2);
    s.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

/// Size bands for the overhang split (`DESIGN.md` §9.2u, #47).
///
/// Declared here rather than imported because `MemProbe` is compiled without
/// the `metal` feature while `candle_metal_kernels` is not, and a `cfg` on a
/// struct field would make the two builds different shapes.
/// `band_names_match_the_pools` pins it to the pool's own constant on the
/// `metal` build, so the two cannot drift.
const SIZE_BAND_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
struct MemProbe {
    token: u64,
    kv_len: u64,
    wall_ms: f64,
    pool_live: u64,
    pool_free: u64,
    pool_pending: u64,
    pool_live_buffers: u64,
    pool_free_buffers: u64,
    pool_free_buckets: u64,
    pool_pending_buffers: u64,
    /// Cumulative since process start, so a rate is a difference of two samples.
    allocations: u64,
    allocated_bytes: u64,
    evicted: u64,
    hits: u64,
    lookups: u64,
    buckets_probed: u64,
    /// Bytes callers asked for, and bytes they were given (`DESIGN.md` §9.2s,
    /// #47). Cumulative, so the overhang over a window is the difference of
    /// two samples, exactly as `allocations` is used.
    ///
    /// **Not derivable from `allocated_bytes`**, which is why these are
    /// separate fields rather than a computation over the ones above.
    /// `allocated_bytes` counts what went to `newBufferWithLength` -- fresh
    /// allocations only -- so it is blind to the pool HIT that is served from
    /// a strictly larger bucket, and in steady-state decode almost every
    /// acquisition is a hit. `served_bytes` counts every acquisition.
    requested_bytes: u64,
    served_bytes: u64,
    /// The two fields above, split by the size band of the REQUEST
    /// (`DESIGN.md` §9.2u, #47). Index-aligned with `SIZE_BAND_NAMES`,
    /// cumulative like everything else here, and summing to `requested_bytes` /
    /// `served_bytes` exactly -- which is what lets a band be read *against* the
    /// headline rather than merely beside it.
    ///
    /// **This is what separates a growing request from a widening gap.** The
    /// whole-pool ratio moves when either term moves, so on its own it cannot
    /// say whether the allocator got worse or the workload asked for more.
    /// Per band, the KV population's own growth is visible in its own row.
    band_acquisitions: [u64; SIZE_BAND_COUNT],
    band_requested_bytes: [u64; SIZE_BAND_COUNT],
    band_served_bytes: [u64; SIZE_BAND_COUNT],
    /// Range-query candidates rejected by the reuse-oversize bound
    /// (`DESIGN.md` §9.2v, #47).
    ///
    /// **The arm's engagement proof.** A bound that is installed but never
    /// binds and a bound that is not installed at all produce identical
    /// `requested`/`served` sums, so this is the only field that separates
    /// them -- #500's lesson that a mechanism with no production callers yields
    /// a clean null indistinguishable from a real one, and that a counter is
    /// the only thing that tells them apart.
    oversize_declined: u64,
    residency: u64,
    /// Residency-set activity, cumulative (`DESIGN.md` §6.3e, issue #210).
    ///
    /// **`residency_commits` is the quantity §6.3d's cost argument is about.**
    /// That section attributes eager unregistration's +0.062 ms/token to putting
    /// a `commit()` on the per-token path; these report how many there are, so
    /// the argument travels with a rate instead of being restated. Differenced
    /// between two samples they give commits per token, which is the figure the
    /// arms differ in.
    ///
    /// `residency_retired` nonzero with `residency_removed` at zero is the
    /// pre-#210 arm -- the 48 GB retention -- so the two together say which arm
    /// a run belongs to without trusting the environment.
    residency_commits: u64,
    residency_added: u64,
    residency_removed: u64,
    residency_retired: u64,
    device_allocated: u64,
    phys_footprint: u64,
    resident: u64,
}

impl MemProbe {
    #[cfg(feature = "metal")]
    fn read(dev: &candle::MetalDevice, token: u64, kv_len: u64, wall: f64) -> Self {
        let (shared, private) = dev.pool_occupancy();
        let (cs, cp) = dev.pool_counters();
        let rc = dev.residency_counters();
        let (phys, resident) = proc_footprint().unwrap_or((0, 0));
        Self {
            token,
            kv_len,
            wall_ms: wall * 1e3,
            pool_live: (shared.live_bytes + private.live_bytes) as u64,
            pool_free: (shared.free_bytes + private.free_bytes) as u64,
            pool_pending: (shared.pending_bytes + private.pending_bytes) as u64,
            pool_live_buffers: (shared.live_buffers + private.live_buffers) as u64,
            pool_free_buffers: (shared.free_buffers + private.free_buffers) as u64,
            pool_free_buckets: (shared.free_buckets + private.free_buckets) as u64,
            pool_pending_buffers: (shared.pending_buffers + private.pending_buffers) as u64,
            allocations: cs.allocations + cp.allocations,
            allocated_bytes: cs.allocated_bytes + cp.allocated_bytes,
            evicted: cs.evicted + cp.evicted,
            hits: cs.hits + cp.hits,
            lookups: cs.lookups + cp.lookups,
            buckets_probed: cs.buckets_probed + cp.buckets_probed,
            requested_bytes: cs.requested_bytes + cp.requested_bytes,
            served_bytes: cs.served_bytes + cp.served_bytes,
            // Summed over both pools, exactly as every counter above is, so a
            // band row and the headline are over one population (§9.2u).
            band_acquisitions: std::array::from_fn(|i| {
                cs.bands[i].acquisitions + cp.bands[i].acquisitions
            }),
            band_requested_bytes: std::array::from_fn(|i| {
                cs.bands[i].requested_bytes + cp.bands[i].requested_bytes
            }),
            band_served_bytes: std::array::from_fn(|i| {
                cs.bands[i].served_bytes + cp.bands[i].served_bytes
            }),
            oversize_declined: cs.oversize_declined + cp.oversize_declined,
            residency: dev.residency_set_len() as u64,
            residency_commits: rc.commits,
            residency_added: rc.added,
            residency_removed: rc.removed,
            residency_retired: rc.retired,
            device_allocated: dev.metal_device().current_allocated_size() as u64,
            phys_footprint: phys,
            resident,
        }
    }

    /// Push this reading into `run_telemetry`'s by-class timeline (#171, #205).
    ///
    /// # Why this exists, and why the Cargo line alone was not enough
    ///
    /// #171 built `metal::run_telemetry` and #205 made it *reachable* from an
    /// example. Neither made it **emit**: `grep -rn 'run_telemetry'` over the
    /// tree returns the module's own file and its `pub mod` line and **nothing
    /// else**, so before this function the feature compiled a recorder that
    /// nobody ever called. §3.4a-v predicted exactly that -- *"Not built here:
    /// the emitter ... the harness does not yet populate these fields, so they
    /// read as absent on every real run today"* -- and it is the reason
    /// `--run-telemetry` on its own would have produced an empty file rather
    /// than a flat one. **An empty series and a flat series are different
    /// findings**, and only the second is worth anything.
    ///
    /// This is the same shape as the defect #205 is about, one layer further
    /// in: a capability that is present, correct, and connected to nothing.
    /// §15.2 #11 is the rule -- *consume it or delete it*.
    ///
    /// # What each class reads, and which are honestly absent
    ///
    /// `sample` is called by the harness rather than the allocator, per the
    /// module's own docs: a sample is a handful of counter reads under locks
    /// candle already takes, taken tens of times per run instead of 675 times
    /// per token, which is what keeps it off the path §6.4a prices.
    ///
    /// | class | source | note |
    /// |---|---|---|
    /// | `Pool` | `pool_occupancy()` | `live` and `pending` kept separate, per §3.4a-iv |
    /// | `Residency` | `residency_set_len()` | a **count**, not bytes -- see below |
    /// | `Kv` | computed from §5.6's geometry | see below |
    /// | `Arena` | 0 unless installed | absent is reported as 0 and said so in the PR |
    /// | `Scratch` | 0 | **nothing on the LFM2 path dispatches a scratch kernel** (§9.1a), so this is structurally 0 rather than unmeasured |
    ///
    /// **`Residency` is a count in a bytes field and that is deliberate.**
    /// §6.3e is explicit that `residency_set_len()` is *"the wrong instrument"*
    /// for the retention -- it reports our membership record, which
    /// `retire_batch` shrinks, and it grew by ~1 700 while 265 694 allocations
    /// happened. It is plotted because §6.3c makes the set a lifetime
    /// participant and a class that is not plotted is one nobody asks about;
    /// it is **not** plotted as the retention, which `device_allocated` is.
    ///
    /// **`Kv` is computed, not read, and that is a real limitation.** Under
    /// `KvAppend=InPlace` the cache is a `Reserve` allocation whose *live*
    /// extent is `kv_len` while its *allocated* extent is `kv_capacity` from
    /// the first token; under `Cat` it is reallocated every token. Neither
    /// exposes a byte count, so this is `kv_len x 16 384 B` -- §5.6's geometry,
    /// **not** §5.6's table, which #164 found low by 2.4 % because it was
    /// computed with 16 000. So the `kv` curve is *what the cache logically
    /// holds*, and the difference from what it *occupies* is a `Reserve`
    /// figure the PR states rather than hides.
    #[cfg(feature = "metal")]
    fn emit_telemetry(&self, kv_bytes_per_token: u64) {
        use candle::kernels::metal::run_telemetry as rt;

        rt::set_token(self.token);
        rt::sample(
            rt::MemClass::Pool,
            self.pool_live,
            self.pool_live_buffers,
            self.pool_pending,
        );
        // Count in the `live_count` column and zero bytes, so a reader cannot
        // mistake a membership count for a byte figure (§6.3e's caution).
        rt::sample(rt::MemClass::Residency, 0, self.residency, 0);
        rt::sample(
            rt::MemClass::Kv,
            self.kv_len.saturating_mul(kv_bytes_per_token),
            self.kv_len,
            0,
        );
        // Reported as zero rather than omitted: an absent class and a
        // zero-valued one are different claims, and §9.1a establishes that
        // nothing on this path allocates scratch, so zero is the measurement.
        rt::sample(rt::MemClass::Arena, 0, 0, 0);
        rt::sample(rt::MemClass::Scratch, 0, 0, 0);
        rt::mark(rt::Event::TokenBoundary, self.kv_len);
    }

    #[cfg(not(feature = "metal"))]
    fn emit_telemetry(&self, _kv_bytes_per_token: u64) {}

    /// One line for a human watching a long run, on stderr.
    fn human(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        format!(
            "progress: token {:>7}  kv_len {:>7}  wall {:7.3} ms  \
             pool live {:6.3} free {:6.3} pending {:6.3} GB  \
             dev {:6.3} GB  phys {:6.3} GB  resid {:>6}  alloc {:>9}  evict {:>9}  \
             rm {:>9}  ret {:>9}  commits {:>9}",
            self.token,
            self.kv_len,
            self.wall_ms,
            gb(self.pool_live),
            gb(self.pool_free),
            gb(self.pool_pending),
            gb(self.device_allocated),
            gb(self.phys_footprint),
            self.residency,
            self.allocations,
            self.evicted,
            self.residency_removed,
            self.residency_retired,
            self.residency_commits,
        )
    }

    /// One JSON object per sample, for the committed series.
    ///
    /// Bytes stay **bytes** here and are divided only for the human line: a
    /// series that has already been rounded to GB cannot be differenced to
    /// recover a per-token rate, which is the quantity issue #206's second
    /// acceptance item asks for.
    fn jsonl(&self) -> String {
        format!(
            "{{\"token\":{},\"kv_len\":{},\"wall_ms\":{:.4},\
             \"pool_live\":{},\"pool_free\":{},\"pool_pending\":{},\
             \"pool_live_buffers\":{},\"pool_free_buffers\":{},\
             \"pool_free_buckets\":{},\"pool_pending_buffers\":{},\
             \"allocations\":{},\"allocated_bytes\":{},\"evicted\":{},\
             \"hits\":{},\"lookups\":{},\"buckets_probed\":{},\
             \"requested_bytes\":{},\"served_bytes\":{},\
             \"band_acquisitions\":{},\"band_requested_bytes\":{},\
             \"band_served_bytes\":{},\"oversize_declined\":{},\
             \"residency\":{},\"residency_commits\":{},\
             \"residency_added\":{},\"residency_removed\":{},\
             \"residency_retired\":{},\"device_allocated\":{},\
             \"phys_footprint\":{},\"resident\":{}}}",
            self.token,
            self.kv_len,
            self.wall_ms,
            self.pool_live,
            self.pool_free,
            self.pool_pending,
            self.pool_live_buffers,
            self.pool_free_buffers,
            self.pool_free_buckets,
            self.pool_pending_buffers,
            self.allocations,
            self.allocated_bytes,
            self.evicted,
            self.hits,
            self.lookups,
            self.buckets_probed,
            self.requested_bytes,
            self.served_bytes,
            json_u64_array(&self.band_acquisitions),
            json_u64_array(&self.band_requested_bytes),
            json_u64_array(&self.band_served_bytes),
            self.oversize_declined,
            self.residency,
            self.residency_commits,
            self.residency_added,
            self.residency_removed,
            self.residency_retired,
            self.device_allocated,
            self.phys_footprint,
            self.resident,
        )
    }
}

/// Access to the Metal profiling counters, with a no-op stand-in off Metal.
///
/// The shim keeps the call sites unconditional: without it every snapshot and
/// reset would need its own `#[cfg]`, and a `cfg`-heavy measurement loop is
/// exactly where a "which build did this number come from?" mistake hides.
#[cfg(feature = "metal")]
mod profile {
    pub use candle::metal_backend::profile::{enabled, reset, snapshot};
}

#[cfg(not(feature = "metal"))]
mod profile {
    #[derive(Clone, Debug, Default)]
    pub struct Snapshot {
        pub dispatches: u64,
        pub encoders: u64,
        pub fence_probes: u64,
        pub fence_hits: u64,
        pub fence_waits: u64,
        pub command_buffers: u64,
        pub timed_command_buffers: u64,
        pub gpu_busy_sum_s: f64,
        pub gpu_busy_union_s: f64,
        pub gpu_span_s: f64,
        pub by_label: Vec<(String, u64)>,
    }
    pub fn enabled() -> bool {
        false
    }
    pub fn reset() {}
    pub fn snapshot() -> Snapshot {
        Snapshot::default()
    }
}

/// The resolved variant configuration for one run (`DESIGN.md` §7.1, #105).
///
/// Parsed and validated once, before the clock starts, so an unsatisfiable
/// combination fails immediately rather than after a five-minute measurement.
/// `announce` prints it, because a timing figure whose configuration is not
/// stated is the defect #105 exists to remove: today's `.bench/` rows are
/// all-defaults and nothing says so.
struct Axes {
    /// Whether the arena is installed at all.
    arena: bool,
    /// Decode steps recorded on the pool before the arena is installed.
    record_steps: usize,
    #[cfg(feature = "metal")]
    layout: candle::metal_backend::ArenaLayout,
    /// #70's GPU bump allocator, rather than #69's CPU plan.
    gpu_offsets: bool,
    /// #47's session region, for the population the arena declines.
    session_region: bool,
    /// The Metal device, so an axis arm can be read back from the mechanism
    /// rather than echoed from the flag that asked for it (§7.1a-i).
    #[cfg(feature = "metal")]
    device: Option<candle::MetalDevice>,
    /// The attention arm, echoed for the run line.
    ///
    /// **The resolved enum, not the flag string** (issue #241). It was a
    /// `String` re-tested by hand, and the test was `attn == "sdpa"` with an
    /// `else` -- a two-armed conditional over what became a three-armed axis
    /// when #116 added `FlashDecoding`, so `--attn flash` reported
    /// `AttnImpl=Generic` on every `RESULT` line. Holding the enum makes the
    /// rendering an exhaustive `match`, so a fourth arm is a compile error
    /// where it was previously a silent mislabel.
    attn: AttnImpl,
    /// The KV-append arm, echoed for the run line (issue #142).
    ///
    /// The enum for the same reason as `attn` above: this axis has two arms
    /// today and a hand-written binary test is correct only by coincidence of
    /// that count.
    kv_append: KvAppend,
    kv_quant: KvQuant,
    /// The conv-state arm (#141), echoed for the run line.
    conv_state: ConvState,
    /// Whether the decode conv is the fused step, from `Config`: it decides
    /// which kernel the conv-state arm above runs on.
    conv_fused: bool,
    /// The Q/K/V layout and the conv carry, from `Config`. No flag here
    /// selects them; a preset does, so the line states them.
    qkv_layout: lloom::models::lfm2::QkvLayout,
    conv_prefill_ctx: bool,
    /// KV tokens per page, and pages per chunk (#116).
    ///
    /// Only reachable under `AttnImpl=FlashDecoding` and inert otherwise. Held
    /// so `axis_pairs` can state them: an axis that is inert is still an axis
    /// the run took a value for, and recording it as `UNRECORDED` would make
    /// this run unpoolable with any run that did record it (#241).
    flash_page_size: usize,
    flash_k: usize,
    /// The scratch sizing arm (#234), echoed for the run line.
    ///
    /// The enum and not the flag string, for #241's reason: the arm is decided
    /// on `Config`, so that is where it is read from, and it renders through an
    /// exhaustive `match` so a fourth arm is a compile error rather than a
    /// silently wrong value.
    flash_scratch_sizing: FlashScratchSizing,
    /// The admission fraction, or `None` when no budget is installed (#249).
    ///
    /// Read back from the `Config` the `Cache` was built from rather than from
    /// the flag, which is #241's discipline: the arm is decided on `Config`, so
    /// that is where the arm actually is. Before this, `axis_pairs` rendered
    /// `MemoryBudget=None` **unconditionally** with a comment saying no harness
    /// installs one — true when written, and this is the harness that does, so
    /// leaving it would have been §7.1a-i's *present-and-wrong* failure rather
    /// than the safer absent one.
    memory_budget: Option<f64>,
    /// Concurrent sequences (#249). Not a variant axis in §7.1a's registry —
    /// it is a workload parameter like `--n` — but it is stated beside them
    /// because a run's `B` decides whether its figures may be pooled with
    /// another's, which is exactly what the axis vector is for.
    batch: usize,
    /// Where the command buffer is cut, and the period the counter arm uses
    /// (`DESIGN.md` §13.5m, issue #420 stage 3).
    ///
    /// **Captured from the device at construction, never re-read from the
    /// environment at render time.** `CANDLE_METAL_NODES_PER_CB` selects the
    /// policy and `Commands::new` resolves it once; reading the variable again
    /// here would be reading the flag rather than the mechanism, which is the
    /// route §7.1a-i's `AttnImpl` defect took. `None` on a non-Metal device,
    /// where there are no command buffers to cut — a third fact, and distinct
    /// from either arm.
    #[cfg(feature = "metal")]
    cut_policy: Option<(candle::metal_backend::CutPolicy, usize)>,
    /// The two GEMV epilogue fusions (§12.2i, #509's wiring of #484's and
    /// #509's kernels).
    ///
    /// Read back from `Config` rather than from the flag, per #241: these are
    /// construction-tier axes whose arm is decided on the config the model was
    /// built from, so that is where the arm actually is.
    gemv_silu: bool,
    gemv_residual: bool,
    gemv_gate: lloom::models::lfm2::GateFusion,
    /// The non-blocking token path (§11.5g, #513's wiring), and its two
    /// sub-arms (#535, #559). Lloom #567 brings all three onto this harness.
    ///
    /// `token_ring` and `nonblocking_eos` are **harness-level** arms: nothing
    /// in `Config` or on the device records them, because what they change is
    /// which code path this loop takes. They are read from the resolved
    /// booleans the loop branches on rather than re-derived from the flag
    /// strings, which is #241's discipline applied to the one place it can be:
    /// there is a single parse and the same value decides the branch and the
    /// line.
    token_ring: bool,
    nonblocking_eos: bool,
    /// The poll mechanism, held as the **resolved enum** rather than the flag
    /// string, so `poll.name()` renders what `wait()` will actually do.
    ///
    /// Inert without `--token-ring` — nothing polls — and stated anyway, for
    /// `FlashPageSize`'s reason: *inert* and *unrecorded* are different facts.
    poll: PollMechanism,
    /// The completion arm (§11.5o), **read back from the device** rather than
    /// from the flag (§7.1a-i).
    ///
    /// `Commands::new` resolves `CANDLE_METAL_COMPLETION_WAIT` once through a
    /// `OnceLock`, so the device is where the arm actually is; reading the flag
    /// here would report an arm the run may not have executed, which is the
    /// wrong-arm shape that compares equal to every real run of that arm.
    /// `None` on a non-Metal device, where there is no command buffer to wait
    /// for — a third fact, distinct from either arm.
    ///
    /// **Carries a parameterised arm's PERIOD**, which the device's own
    /// `completion_wait_name()` does not — see [`Axes::completion_wait_arm`].
    completion_wait: Option<&'static str>,
    /// A parameterised arm's period, or `None` on the arms that have none.
    ///
    /// **Two arms carry one** (#583): `shared-event-sleep`'s is a sleep in
    /// **microseconds** and `shared-event-wait`'s is a timeout in
    /// **milliseconds**. The unit is implied by the arm name they are rendered
    /// beside, so one field serves both rather than one field per unit — a
    /// second copy of the list is what §11.3h warns decays.
    ///
    /// Held beside the name rather than folded into it because the name is a
    /// `&'static str` from the device and the period is a run-time value; the
    /// two are joined only at render time, in one place.
    completion_wait_period: Option<u64>,
}

/// The `AttnImpl` arm, as `.bench/configurations.md` §1 spells it.
///
/// # Why this is a `match` on the enum and not a string test
///
/// It was `if attn == "sdpa" { "Sdpa" } else { "Generic" }` — a **two-armed
/// conditional over a three-armed axis** — so `--attn flash` reported
/// `AttnImpl=Generic` on every `RESULT` line it ever produced (issue #241).
///
/// **That is worse than an omission.** An absent field reads as `UNRECORDED`
/// and a reader knows to go looking; a wrong field is a false statement, and a
/// flash run was actively mislabelled as the arm it is not. It also means a
/// completeness check — "does the line name every axis?" — passes on it, which
/// is why #215's audit of the same function did not catch this: **presence is
/// not correctness.**
///
/// Exhaustive by construction: a fourth arm added to [`AttnImpl`] fails to
/// compile here rather than silently rendering as one of these three.
fn attn_impl_arm(attn: AttnImpl) -> &'static str {
    match attn {
        AttnImpl::Generic => "Generic",
        AttnImpl::Sdpa => "Sdpa",
        AttnImpl::FlashDecoding => "FlashDecoding",
    }
}

/// The `KvAppend` arm (#142). Exhaustive for [`attn_impl_arm`]'s reason.
///
/// This axis has two arms today, so the `if kv_append == "in-place"` it
/// replaces was *correct* — and correct by coincidence of the arm count, which
/// is the property #241 is about. A binary test over a two-armed axis and a
/// binary test over a three-armed axis are the same code.
fn kv_append_arm(kv: KvAppend) -> &'static str {
    match kv {
        KvAppend::Cat => "Cat",
        KvAppend::InPlace => "InPlace",
        KvAppend::Paged => "Paged",
    }
}

/// The `KvQuant` arm (#218, `DESIGN.md` §10.7). Exhaustive, for the reason
/// above: three arms today, and a binary test over a three-armed axis is #241's
/// exact defect.
///
/// **Added to `config_line()` in the same change that adds the axis.** §7.1a
/// records this surface falling behind the registry five times — an axis added
/// to the registry and not here renders `UNRECORDED`, which is the *safe*
/// direction but still means no comparison can hold the axis fixed. There is no
/// reason to incur that debt on a new axis when the fix is one line.
fn kv_quant_arm(kv: KvQuant) -> &'static str {
    match kv {
        KvQuant::Off => "Off",
        KvQuant::Int8 => "Int8",
        KvQuant::Int4 => "Int4",
    }
}

/// The `ConvState` arm (#141, §10.2g), with its parameters.
///
/// Already exhaustive before #241 — recorded here because it is the shape the
/// other two now have, and because it is the reason the audit found only one
/// defect: a `match` on an enum was the pattern this file already used in two
/// of three places.
fn conv_state_arm(conv: ConvState) -> String {
    match conv {
        ConvState::Shuffle => "Shuffle".to_string(),
        ConvState::SlidingRing { k, slack } => format!("SlidingRing(k={k},slack={slack})"),
        ConvState::RotatingRing { k } => format!("RotatingRing(k={k})"),
    }
}

/// The `Poll` arm, **with its period** (lloom #567, `DESIGN.md` §11.5o).
///
/// [`PollMechanism::name`] renders every `Sleep(micros)` as bare `sleep`, which
/// is right for a human-readable engagement line and wrong for an axis arm:
/// §11.5o measures `sleep:50` and `sleep:200` at **15.0 against 3.9 polls/ms**
/// and **+0.039 against +0.121 ms** of wake latency, so they are two arms. An
/// arm rendered as bare `sleep` would make them compare **equal** in #171's
/// store — `CutPolicy`'s exact case (`Counter(50)` against `Counter(55)`,
/// §13.5m), and §7.1a-i's present-and-wrong direction rather than the safe
/// absent one.
///
/// The two unparameterised arms defer to `name()` so a fourth mechanism added
/// to [`PollMechanism`] is still a compile error there (#241).
/// Rendered in **microseconds**, which is the unit the flag is spelled in
/// (`--poll sleep:200`), so the arm on the line round-trips to the invocation
/// that produced it rather than to a `Duration`'s `Debug`.
fn poll_arm(poll: PollMechanism) -> String {
    match poll {
        PollMechanism::Sleep(d) => format!("sleep:{}", d.as_micros()),
        other => other.name().to_string(),
    }
}

impl Axes {
    /// Resolves the axes for one run.
    ///
    /// `config` is the model configuration `main` has *already* built, and the
    /// attention, KV-append and conv-state arms are read back from it rather
    /// than re-derived from the flag strings (issue #241). Before, `main`
    /// parsed `--attn` into `config.attn_impl` with a three-armed `match` and
    /// this function separately kept the raw string, which `config_line` then
    /// tested with a two-armed `if`. **Two parses of one flag is what let them
    /// disagree**, and the disagreement was silent: `--attn flash` selected
    /// `FlashDecoding` and reported `Generic`. One parse, read back from the
    /// mechanism that decides it, is the same discipline `math_mode_name` and
    /// `hazard_key_name` already follow.
    fn resolve(args: &Args, _device: &Device, config: &Config) -> Result<Self> {
        #[cfg(feature = "metal")]
        let layout = match args.arena_layout.as_str() {
            "packed" => candle::metal_backend::ArenaLayout::Packed,
            "non-aliasing" | "nonaliasing" | "reference" => {
                candle::metal_backend::ArenaLayout::NonAliasing
            }
            other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
        };

        let gpu_offsets = match args.arena_offsets.as_str() {
            "cpu" => false,
            "gpu" => true,
            other => anyhow::bail!("unknown --arena-offsets {other:?}; want cpu or gpu"),
        };
        if gpu_offsets {
            if !args.arena {
                anyhow::bail!(
                    "--arena-offsets gpu needs --arena: there is nothing to allocate over"
                );
            }
            // Refused rather than downgraded (#70, §9.2g). A packed plan reuses
            // slots, so its offsets are not monotone and a forward-only cursor
            // cannot reproduce them; falling back to the CPU here would report
            // #69's numbers under this flag's name.
            #[cfg(feature = "metal")]
            if layout != candle::metal_backend::ArenaLayout::NonAliasing {
                anyhow::bail!(
                    "--arena-offsets gpu needs --arena-layout non-aliasing: a packed plan reuses \
                     slots, so its offsets are not monotone and no bump allocator can reproduce \
                     them"
                );
            }
        }

        // An explicit flag wins; otherwise the selector is left alone so
        // `CANDLE_METAL_HAZARD_KEY` is honoured. Calling the setter
        // unconditionally consumes the `OnceLock` guarding the env switch and
        // silently pins the default -- which it did in #69, and which made a
        // determinism run that believed it was testing route (a) actually test
        // the default (§2.4, §9.2f).
        #[cfg(feature = "metal")]
        if let Some(requested) = args.hazard_key.as_deref() {
            let key = match requested {
                "pointer" | "ptr" => candle::metal_backend::HazardKey::Pointer,
                "range" => candle::metal_backend::HazardKey::Range,
                other => anyhow::bail!("unknown --hazard-key {other:?}; want pointer or range"),
            };
            candle::metal_backend::set_hazard_key(key);
        }
        #[cfg(not(feature = "metal"))]
        if args.hazard_key.is_some() {
            anyhow::bail!("--hazard-key needs a Metal build");
        }

        if args.arena && !matches!(_device, Device::Metal(_)) {
            anyhow::bail!("--arena needs a Metal device");
        }

        // #47's session region. Refused rather than silently ignored, on the
        // reason every other axis here is refused: a flag that does nothing
        // makes a run compare EQUAL to one that took the arm, which is the
        // failure §7.1a-i names and the one a measurement cannot recover from.
        if args.session_region && !args.arena {
            anyhow::bail!(
                "--session-region needs --arena: the reservations are derived from the \
                 arena recording's two steps, which is what measures the per-token stride"
            );
        }
        if args.session_region && !matches!(_device, Device::Metal(_)) {
            anyhow::bail!("--session-region needs a Metal device");
        }

        // ---- #567's three async arms, refused rather than defaulted --------
        //
        // Parsed before the clock for the reason every other axis here is: an
        // unsatisfiable combination must fail immediately rather than after a
        // five-minute measurement, and a silently-ignored flag makes a run
        // report an arm it did not execute (§2.4, after #69's vacuous
        // determinism run).
        // Unset takes the type's default rather than a string spelled in the
        // arg attribute (#578).
        let poll = match &args.poll {
            Some(s) => PollMechanism::from_str(s).map_err(anyhow::Error::msg)?,
            None => PollMechanism::default(),
        };

        // **The ring is `--temperature 0` only.** §11.5f measures `sample_f`'s
        // temperature path taking the same `to_cpu` over the whole vocabulary
        // -- 128 000 f32 against argmax's 4 -- and §11.2c puts 74 % of that arm
        // in a host `sort_by`. `sample_argmax_on_device` is the only sampler
        // that leaves the token on the device, so a `--token-ring
        // --temperature 0.7` run would take the host path while reporting the
        // ring arm: §7.1a-i's wrong-arm shape.
        anyhow::ensure!(
            !args.token_ring || args.temperature <= 0.,
            "--token-ring requires --temperature 0 (DESIGN.md §11.5g): the device-resident \
             sampler is argmax-only, and a temperature run would take the host readback \
             while reporting the ring arm"
        );
        // The non-blocking EOS read is a property OF the ring arm, so asking
        // for it without the ring is a contradiction rather than a no-op.
        anyhow::ensure!(
            !args.nonblocking_eos || args.token_ring,
            "--nonblocking-eos requires --token-ring (DESIGN.md §11.5n, #535): the flag it \
             polls is published by the ring, so without it there is nothing to poll"
        );
        // The ring publishes ONE slot and the model reads it as `[1, 1]`.
        // §6.6c records the readback as `B` round trips per step at B>1, which
        // makes the slot `B` slots and is #263's change -- so a batched ring
        // run would publish row 0's token as every row's input. Refused rather
        // than measured wrong.
        anyhow::ensure!(
            !args.token_ring || args.batch == 1,
            "--token-ring is a B=1 mechanism (DESIGN.md §11.5g, §6.6c): the ring holds one \
             persistent [1, 1] input slot, so a batched run would feed row 0's token to \
             every row"
        );
        // The two sampling arms are mutually exclusive **mechanisms**, not
        // compatible flags: #319's arm enqueues an `argmax` whose result is
        // read back to the host, and the ring's whole point is that nothing is
        // read back. Silently letting one win would report both.
        anyhow::ensure!(
            !(args.token_ring && args.sample_in_buffer),
            "--token-ring and --sample-in-buffer are two different removals of the same \
             round trip (DESIGN.md §11.5b, §11.5g) and cannot both be selected: the first \
             leaves the token on the device, the second reads it back from the forward \
             pass's command buffer"
        );
        // `--speculate` has its own loop with its own shape (a window consumes
        // K and emits between 1 and K), and the ring publishes exactly one
        // token per step. Refused rather than silently ignored on that path.
        anyhow::ensure!(
            !(args.token_ring && args.speculate > 0),
            "--token-ring and --speculate cannot be combined: the verify loop emits \
             between 1 and K tokens per window and the ring publishes one per step"
        );
        if args.token_ring && !matches!(_device, Device::Metal(_)) {
            anyhow::bail!("--token-ring needs a Metal device");
        }

        Ok(Self {
            arena: args.arena,
            record_steps: args.arena_record_steps,
            #[cfg(feature = "metal")]
            layout,
            gpu_offsets,
            session_region: args.session_region,
            #[cfg(feature = "metal")]
            device: match _device {
                Device::Metal(d) => Some(d.clone()),
                _ => None,
            },
            // Read back from the configuration the model was built from, not
            // re-parsed from the flag (#241). These three are the axes whose
            // arm is decided on `Config`, so this is where the arm actually is.
            attn: config.attn_impl,
            kv_append: config.kv_append,
            kv_quant: config.kv_quant,
            conv_state: config.conv_state,
            conv_fused: config.conv_fused,
            qkv_layout: config.qkv_layout,
            conv_prefill_ctx: config.conv_prefill_ctx,
            flash_page_size: config.flash_page_size,
            flash_k: config.flash_pages_per_chunk,
            flash_scratch_sizing: config.flash_scratch_sizing,
            memory_budget: config.memory_budget.map(|b| b.fraction),
            batch: args.batch,
            // Resolved by `Commands::new` from `CANDLE_METAL_NODES_PER_CB` once
            // per device; read here rather than re-derived from the variable,
            // per #241. The period travels with the policy because a counter
            // cut at 50 and one at 55 are different arms -- #420 stage 3's
            // control is a counter arm at a period chosen to match the node
            // arm's session count, and a line saying only `Counter` could not
            // separate it from the shipping default.
            #[cfg(feature = "metal")]
            cut_policy: match _device {
                Device::Metal(m) => Some((m.cut_policy(), m.compute_per_buffer())),
                _ => None,
            },
            // From the config, not from `args`, for the reason the three above
            // give: the arm is decided on `Config` and read back from it.
            gemv_silu: config.gemv_silu,
            gemv_residual: config.gemv_residual,
            gemv_gate: config.gemv_gate,
            // #567's three async arms. The two booleans are the same values the
            // decode loop branches on; `poll` is the resolved enum, so the line
            // renders what `wait()` does rather than what was typed.
            token_ring: args.token_ring,
            nonblocking_eos: args.nonblocking_eos,
            poll,
            // **Read back from the device, not from the flag** (§7.1a-i). The
            // arm was installed into the environment before `Device::new_metal`
            // and resolved once inside `Commands::new`; `main` has already
            // asserted the two agree, so this reads the mechanism.
            #[cfg(feature = "metal")]
            completion_wait: match _device {
                Device::Metal(m) => Some(m.completion_wait_name()),
                _ => None,
            },
            #[cfg(not(feature = "metal"))]
            completion_wait: None,
            // The period comes from the PARSED arm rather than the device,
            // because `completion_wait_name()` returns a `&'static str` that
            // has already discarded it. That is safe here and only here: `main`
            // has asserted the parsed arm and the device's agree by name
            // immediately after constructing the device, so this reads the
            // period *of the arm the device installed*. Without that assertion
            // it would be reading the flag (§7.1a-i).
            //
            // **The period comes from the ARM rather than from a match here**
            // (#583). This was a local `match` naming each parameterised arm,
            // which is a list that goes stale silently: an arm added without a
            // branch falls into `_ => None` and renders **equal** to the same
            // arm at every other period. `CompletionWait::period()` is the one
            // place that knows, and it is an exhaustive match, so a new arm is
            // a compile error there rather than a wrong axis vector here.
            #[cfg(feature = "metal")]
            completion_wait_period: candle::metal_backend::CompletionWait::from_str_arm(
                &args.completion_wait,
            )
            .ok()
            .and_then(|a| a.period()),
            #[cfg(not(feature = "metal"))]
            completion_wait_period: None,
        })
    }

    /// The `CompletionWait` arm, **with its period** (lloom #567).
    ///
    /// `CompletionWait::name()` renders every `SharedEventSleep { micros }` as
    /// bare `shared-event-sleep`, which is right for the human-readable
    /// engagement line `lfm2-release` prints and wrong for an axis arm — the
    /// same split `poll_arm` makes for `PollMechanism`, and the same defect
    /// `CutPolicy` carries its period to avoid (§13.5m).
    ///
    /// **Measured, the two sleeping arms are not one arm**: §11.5o reads
    /// `sleep:50` and `sleep:200` at **15.0 against 3.9 polls/ms** and
    /// **+0.039 against +0.121 ms** of wake latency. Rendered bare they would
    /// compare **equal** in #171's store, which is §7.1a-i's present-and-wrong
    /// direction — a wrong arm compares equal to every run of that arm, where
    /// an absent one compares equal to nothing.
    ///
    /// The two unparameterised arms defer to the device's own name, so they
    /// stay read back from the mechanism rather than from the flag.
    ///
    /// **`shared-event-wait` joined the parameterised set** (#583) and its
    /// period is a **timeout in milliseconds**, not a sleep in microseconds.
    /// The rendering is `arm:period` either way because the arm name already
    /// says which unit it is — `shared-event-sleep:50` is 50 µs and
    /// `shared-event-wait:2000` is 2000 ms — and a second field for the unit
    /// would be a second copy of a list (§11.3h).
    fn completion_wait_arm(&self) -> String {
        match (self.completion_wait, self.completion_wait_period) {
            (Some(name), Some(period)) => format!("{name}:{period}"),
            (Some(name), None) => name.to_string(),
            (None, _) => "n/a".to_string(),
        }
    }

    /// The configuration as one line, in the form the registry keys on.
    ///
    /// # One source, rendered twice
    ///
    /// This is [`Self::axis_pairs`] joined, and it is derived rather than
    /// written out for the reason issue #241 exists: it *was* a separate
    /// `format!` with its own copy of each axis's rendering, and the two copies
    /// disagreed. Both carried `if attn == "sdpa" { "Sdpa" } else { "Generic" }`
    /// and both had to be fixed; a third consumer would have needed a third
    /// fix. §11.3h's recurring lesson is that a second copy of a list is a copy
    /// that goes stale, and this had two.
    ///
    /// The shape is unchanged for the axes that were already here — same
    /// `Axis=Arm` tokens, same separator — so the committed corpus still
    /// ingests. `ingest.rs` splits this field on whitespace and then on `=`,
    /// keyed rather than positional, so **added axes append and change nothing
    /// about how an older line parses.** What changes is that the six axes
    /// §7.1a records as absent are now stated: `BarrierScope`, `MathMode`,
    /// `MemoryBudget`, `ScratchLayout`, `FlashPageSize` and `FlashK`.
    fn config_line(&self) -> String {
        self.axis_pairs()
            .into_iter()
            .map(|(axis, arm)| format!("{axis}={arm}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Every axis `.bench/configurations.md` §1 declares, as `(axis, arm)`.
    ///
    /// **This is now the single source, and `config_line()` is derived from
    /// it** (issue #241). It used to be the second of two lists, each with its
    /// own copy of every axis's rendering — and both copies carried the same
    /// two-armed `AttnImpl` conditional, so both had to be fixed. A third
    /// consumer would have needed a third fix.
    ///
    /// # What it reports, and from where
    ///
    /// All **sixteen** axes the registry declares, each read from the mechanism
    /// that actually decides it rather than from a flag — `MathMode` from the
    /// same environment variable `kernel.rs:203` reads, `Executor` from the
    /// device, and the three `Config` axes from the `Config` the model was
    /// built from. A run recorded through this cannot be silently pooled with
    /// one taken under a different math mode, which is the merge the store must
    /// refuse.
    ///
    /// Historical note, because the reasoning is worth keeping: this function
    /// was written when `config_line()` emitted **eight** of the then-eleven
    /// axes, and existed to report the other three. `.bench/configurations.md`
    /// §1 states the consequence exactly — the `config=[…]` line "catches an
    /// axis added to the harness and not to this file. It does **not** catch an
    /// axis added to neither", which is the failure #122 recorded for
    /// `MathMode` (`DESIGN.md` §2.3.9).
    ///
    /// **The stated reason for not fixing `config_line()` was not true.**
    /// It read: *"changing its shape would break the ingest of the corpus that
    /// already exists."* `lloom-runs`' `ingest.rs` splits `config=[…]` on
    /// whitespace and then on `=` — **keyed, not positional** — so added axes
    /// append and change nothing about how an older line parses. Checked
    /// against the parser rather than assumed, which is what #241 did.
    fn axis_pairs(&self) -> Vec<(String, String)> {
        let p = |k: &str, v: &str| (k.to_string(), v.to_string());
        vec![
            // Not selectable here -- nothing on the LFM2 path dispatches the
            // packed variants (§11.3k) -- but stated, because an axis that is
            // pinned at a default is still an axis the run was taken under.
            p("ParamStyle", "Split"),
            p("Executor", Self::executor_name()),
            p(
                "ArenaLayout",
                &if self.arena {
                    #[cfg(feature = "metal")]
                    {
                        match self.layout {
                            candle::metal_backend::ArenaLayout::Packed => "Packed".to_string(),
                            candle::metal_backend::ArenaLayout::NonAliasing => {
                                "NonAliasing".to_string()
                            }
                        }
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        "none".to_string()
                    }
                } else {
                    "none(pool)".to_string()
                },
            ),
            p("ArenaOffsets", if self.gpu_offsets { "Gpu" } else { "Cpu" }),
            p("HazardKey", Self::hazard_key_name()),
            // These three render from the resolved enum through an exhaustive
            // `match` (#241). The `AttnImpl` line was `if attn == "sdpa"` with
            // an `else`, which reported `Generic` for `FlashDecoding` -- a
            // wrong value rather than a missing one, and therefore invisible to
            // any check that asks whether the axis is present.
            p("AttnImpl", attn_impl_arm(self.attn)),
            p("KvAppend", kv_append_arm(self.kv_append)),
            p("KvQuant", kv_quant_arm(self.kv_quant)),
            p("ConvState", &conv_state_arm(self.conv_state)),
            // Read back from `Config`, for the `AttnImpl` reason: the fused
            // conv decides whether `ConvState` runs on `shortconv_step` or on
            // the unfused `broadcast_mul + sum`, and a line without it pools
            // the two.
            p("ConvFused", if self.conv_fused { "on" } else { "off" }),
            p("QkvLayout", self.qkv_layout.as_str()),
            p(
                "ConvPrefillCtx",
                if self.conv_prefill_ctx { "on" } else { "off" },
            ),
            // Only selects anything under `Executor=Icb`, which this harness
            // does not take. Stated at its default rather than omitted: an
            // unstated axis and an axis at its default are different facts, and
            // conflating them is what this whole function exists to stop.
            p("BarrierScope", "RunStart"),
            // **An arm rather than `none`, as of #234.** It read `none` because
            // nothing on the decode path allocated from the scratch class, and
            // that was true until #116 gave it a consumer — after which the
            // line said *"the mechanism is not installed"* about a mechanism
            // that was installed and running at `Grow`. §7.1a-i's
            // present-and-wrong case: a false off-state compares **equal** to
            // real runs of other configurations, where an absent axis would
            // only have been `UNRECORDED`.
            //
            // Rendered from the resolved enum through an exhaustive `match`
            // (#241, #245), so the arm on the line is the arm the model was
            // built with and a fourth arm is a compile error. It is stated on
            // every run including the `Generic` and `Sdpa` ones, where it is
            // inert: *inert* and *unrecorded* are different facts.
            p("ScratchSizing", self.flash_scratch_sizing.name()),
            p("MathMode", Self::math_mode_name()),
            // The five below were absent entirely, which is the *other* half of
            // #241's audit and a different defect from the wrong-value one
            // above. `ReplayBarriers` is on the `RESULT` line and was not here;
            // the last four postdate this function (#186, #195, #116). An axis
            // the run took and did not record is `UNRECORDED` in the store --
            // honest, and it means the run cannot be pooled with one that did
            // record it, so the corpus fragments silently.
            p("ReplayBarriers", "Always"),
            // **This harness now installs one** (#249, `--memory-budget`), so
            // this renders the resolved value rather than the constant `None`
            // it carried while the comment here said *"no harness does"*
            // (§9.5l). That sentence was true when written and became false
            // without this line being edited — §11.3h's recurring shape, and
            // §7.1a-i's *present-and-wrong* case, which is the unsafe direction:
            // a false `None` compares **equal** to real unbudgeted runs where an
            // absent axis would only have been `UNRECORDED`.
            p(
                "MemoryBudget",
                &match self.memory_budget {
                    None => "None".to_string(),
                    Some(f) => format!("{f}"),
                },
            ),
            // Nothing on the decode path allocates from the scratch class, so
            // neither arm is reachable here; `Planes` is what §9.1a's figures
            // were computed under.
            p("ScratchLayout", "Planes"),
            // Only reachable under `AttnImpl=FlashDecoding` and inert
            // otherwise -- but *inert* and *unrecorded* are different facts,
            // and the run did take a value for them.
            p("FlashPageSize", &self.flash_page_size.to_string()),
            p("FlashK", &self.flash_k.to_string()),
            // **Not one of §7.1a's sixteen axes**, and stated anyway. `B` is a
            // workload parameter rather than a variant arm — it selects no
            // kernel and flips no default — but it decides whether two runs'
            // figures are comparable, which is what an axis vector is for. A
            // B=8 row pooled with a B=1 row would be comparing an aggregate
            // against a per-sequence rate. #171's store keys on this field, so
            // a batched run and an unbatched one cannot silently merge.
            p("Batch", &self.batch.to_string()),
            // **Where the command buffer is cut**, and the axis this run exists
            // to move (`DESIGN.md` §13.5m, issue #420 stage 3). Rendered from
            // the device's own resolved policy through an exhaustive `match`:
            // a run that meant to take the node arm and silently got the
            // counter would report a clean null for the wrong reason, which is
            // #69's vacuous determinism run in this axis (§2.4).
            p("CutPolicy", &self.cut_policy_arm()),
            // The two GEMV epilogue fusions (§12.2i, #509's wiring). Stated
            // whether or not they are set, for `FlashPageSize`'s reason: inert
            // and unrecorded are different facts, and a run taken before these
            // existed is `UNRECORDED` rather than `off` — which matters more
            // here than usual, because these two move `logits_sha256` while
            // leaving `tokens_sha256` alone, so a run pooled wrongly would
            // compare two different computations that agree on the field a
            // reader checks first.
            p("GemvSilu", if self.gemv_silu { "on" } else { "off" }),
            p(
                "GemvResidual",
                if self.gemv_residual { "on" } else { "off" },
            ),
            p("GemvGate", self.gemv_gate.name()),
            // The three async arms (§11.5g/§11.5n/§11.5o), brought onto this
            // harness by #567 so one binary can select them **beside** the two
            // epilogues above and report `step_ms_per_token` over the pair.
            //
            // Stated whether or not they are set, for `FlashPageSize`'s reason
            // — *inert* and *unrecorded* are different facts — and the reason
            // is sharper here than usual: these arms are designed **not** to
            // move either digest (§11.5o's gate is that they change how the
            // host learns bytes are ready, not what they are), so two runs
            // differing only in one of them render identical digests and
            // identical text. **The axis field is the only thing that
            // separates them**, and #171's store pools runs whose `config=[…]`
            // agrees. An unrecorded async arm is therefore the exact merge
            // §7.1a-i calls the unsafe direction.
            p("TokenRing", if self.token_ring { "on" } else { "off" }),
            // Rendered from the resolved enum, and **the sleep arm carries its
            // period** — `CutPolicy`'s rule exactly (§13.5m): `PollMechanism::
            // name()` renders every `Sleep(micros)` as bare `sleep`, so
            // `sleep:50` and `sleep:200` would be one arm on the line while
            // §11.5o measures them at 15.0 against 3.9 polls/ms and +0.039
            // against +0.121 ms of wake latency. Two runs whose poll arm has
            // the same *name* and a different *period* are not the same
            // configuration, and #171's store keys on this field.
            //
            // `name()` is used for the two spin arms so a fourth mechanism is
            // still a compile error there (#241); only the parameterised arm is
            // widened, which is where the information is.
            p("Poll", &poll_arm(self.poll)),
            p(
                "NonblockingEos",
                if self.nonblocking_eos { "on" } else { "off" },
            ),
            // From the device, not the flag, and **with the sleeping arm's
            // period** — see `completion_wait_arm`. `n/a` on a non-Metal build,
            // which is distinct from `synchronize`: one is "no command buffer
            // to wait for" and the other is "the blocking wait was selected".
            p("CompletionWait", &self.completion_wait_arm()),
            // The two COMPILE-TIER axes (§7.1's first tier), which every axis
            // above this line is not: these decide what the METALLIB contains,
            // where the others select among kernels it already carries.
            //
            // **`MetalCflags` was the live instance of §2.3.9's failure**, and
            // as of #516 it is a latent one rather than a live one. The
            // passthrough has existed since #434 and became load-bearing with
            // #509 — `GemvResidual=on` REQUIRED `-DLLOOM_509_RESIDUAL_EPILOGUE`
            // or the name was absent from the metallib — while sitting in
            // neither the registry nor this line, so two runs that compiled
            // different programs rendered identical `config=[…]` fields. #516
            // removed that `-D` (it coupled a run-time field to a build-time
            // flag), so **no shipping arm passes one today and this field reads
            // `none` on every default run**. It stays stated: `-flto` is
            // selectable through the same variable and is off by default in the
            // Metal compiler, and a future `-D` would otherwise re-open the hole
            // silently. Read from the crate's own `OnceLock` rather than from a
            // flag, because §7.1a-i's present-and-wrong case is the unsafe
            // direction.
            //
            // `MergedUnit` is #477's arm and is new, so every earlier run is
            // honestly `UNRECORDED` on it rather than silently `off`.
            p("MetalCflags", &Self::metal_cflags_arm()),
            p("MergedUnit", Self::merged_unit_arm()),
            // The buffer-allocation granularity (§9.2, #586). Read back from
            // `MetalDevice::buffer_alignment()` — the `OnceLock` `buf_size`
            // itself reads — rather than from the environment variable, for
            // `CompletionWait`'s reason: a flag echoed to itself is the
            // two-parses-of-one-flag shape that made `--attn flash` report
            // `AttnImpl=Generic` for the life of #116 (§7.1a-i).
            //
            // **Stated on every run, including the default one.** This axis had
            // no selector at all until #586, so unlike every axis above it
            // there is no interval of runs that took an arm without recording
            // it — but the *reason* to state it at the default is
            // `FlashPageSize`'s: a run at 128 and a run that predates the axis
            // are different facts, and only the first can be pooled with a
            // future 128 run.
            //
            // The arm is the byte count verbatim, `CutPolicy`'s rule: 64 and
            // 256 are two granularities and a bare `aligned` would pool them.
            p("BufferAlignment", &Self::buffer_alignment_arm()),
            // The reuse-oversize bound (§9.2v, #47). Read back from the
            // `OnceLock` `acquire` itself consults, on `BufferAlignment`'s
            // reasoning exactly: a flag echoed to itself is §7.1a-i's
            // present-and-wrong case, and this axis has the same failure
            // available to it -- a run that rendered a bound it did not take
            // would compare EQUAL to every real run of that arm.
            //
            // Stated on every run including the default `off`, for the same
            // reason `BufferAlignment` is: this axis did not exist before #47,
            // so a run at `off` and a run predating the axis are different
            // facts, and only the first can be pooled with a future `off` run.
            p("ReuseOversizeBound", &Self::reuse_oversize_bound_arm()),
            // Same reasoning, and the same failure available to it: an arm
            // echoed from the flag rather than read from the pool would render
            // a policy the eviction path did not take (§7.1a-i). Stated on
            // every run including the default `age`, since a run predating the
            // axis and a run at `age` are different facts (§9.2w).
            p("EvictionOrder", &Self::eviction_order_arm()),
            // #599's SiLU variant axis (`DESIGN.md` §12.2l) — which of the
            // seven implementations the 30 `usilu` dispatches per token
            // resolve to.
            //
            // **Read back from `candle::kernels::silu_impl()`**, the same
            // process state `MetalStorage::unary_impl` consults when it picks a
            // `[[host_name]]`, on `ReuseOversizeBound`'s reasoning exactly: a
            // flag echoed to itself is §7.1a-i's present-and-wrong case, and a
            // run that rendered an arm it did not take would compare EQUAL to
            // every real run of that arm.
            //
            // Stated on every run including the default `exact`, for
            // `BufferAlignment`'s reason: this axis did not exist before #599,
            // so a run at `exact` and a run predating the axis are different
            // facts, and only the first can be pooled with a future `exact`
            // run.
            p("SiluImpl", Self::silu_impl_arm()),
            // #47's session region (`DESIGN.md` §9.2x).
            //
            // **Read back from the device**, never echoed from the flag, on
            // `ReuseOversizeBound`'s reasoning exactly: a run that rendered
            // `on` while the install had silently produced no tenants would
            // compare EQUAL to a real `on` run, which is §7.1a-i's
            // present-and-wrong case and the failure #500 measured.
            //
            // The arm carries the **tenant count**, not a bare `on`, because
            // "installed" and "installed and covering something" are different
            // states and only the second is the arm anyone wants to pool on.
            p("SessionRegion", &self.session_region_arm()),
        ]
    }

    /// The `SessionRegion` arm: what the device actually has installed.
    ///
    /// `off` when none is installed — the default and what ships — and
    /// `tenants:<N>` naming the tenant count when one is, so an
    /// installed-but-empty region renders `tenants:0` rather than passing for
    /// a covering one.
    ///
    /// The count sits behind a `:` on `Poll`'s `sleep:<micros>` pattern rather
    /// than glued on as `t56`, because `.bench/configurations.md` declares the
    /// arm once as `tenants:<N>` and `lloom-runs`'s coverage check matches a
    /// rendered arm to a declared one on the stem before the separator. A
    /// glued spelling has no stem, so every run of it would read as an
    /// undeclared arm and the declared one as never rendered — the first form
    /// of this renderer did exactly that, and the check refused it.
    ///
    /// Reads the device rather than a field the install wrote, because a field
    /// IS the flag echoed to itself one indirection later: it would render `t8`
    /// for a run whose install silently produced nothing. This reaches the
    /// `Option<SessionRegion>` the allocation path itself consults, which is the
    /// only state that can answer honestly (§7.1a-i).
    fn session_region_arm(&self) -> String {
        #[cfg(feature = "metal")]
        {
            match self
                .device
                .as_ref()
                .and_then(|d| d.session_region_tenants())
            {
                Some(n) => format!("tenants:{n}"),
                None => "off".to_string(),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a".to_string()
        }
    }

    /// The `EvictionOrder` arm: the order `evict_over_budget` actually applies.
    ///
    /// `age` for the oldest-first behaviour that ships, spelled rather than
    /// left empty, and `n/a` off Metal where there is no pool to evict from --
    /// both following `ReuseOversizeBound` below.
    ///
    /// **These arms are strings, which removes §9.2v's rendering defect by
    /// construction**: there is no `Display` rounding to disagree with the
    /// registry about.
    fn eviction_order_arm() -> String {
        #[cfg(feature = "metal")]
        {
            candle::kernels::metal::buffer_pool::eviction_order()
                .as_str()
                .to_string()
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a".to_string()
        }
    }

    /// The `SiluImpl` arm: which SiLU the `usilu` dispatches resolve to.
    ///
    /// The CLI spelling, which is also what `.bench/configurations.md` §1
    /// declares — `exact`, `recip`, `literal`, `coarse4`, `coarse1`,
    /// `hardswish`, `relu`. Taken from `SiluImpl::name()` rather than written
    /// as a literal here, so the harness and the registry cannot disagree about
    /// an arm's name the way `ReuseOversizeBound`'s float rendering did (#47).
    #[cfg(feature = "metal")]
    fn silu_impl_arm() -> &'static str {
        candle::kernels::silu_impl().name()
    }

    /// Without the Metal backend there is no SiLU kernel to select, so the axis
    /// is honestly absent rather than defaulted to an arm nothing took.
    #[cfg(not(feature = "metal"))]
    fn silu_impl_arm() -> &'static str {
        "UNRECORDED"
    }

    /// The `ReuseOversizeBound` arm: the ceiling `acquire` actually applies.
    ///
    /// `off` for the unbounded behaviour that ships, spelled rather than left
    /// empty -- `MemoryBudget`'s rule, that an arm rendering as the empty
    /// string is indistinguishable from a field the harness failed to fill.
    /// `n/a` off Metal, following `BufferAlignment`: there is no pool for a
    /// bound to apply to, which is a different fact from having taken the
    /// default arm.
    ///
    /// # `{:.1}` rather than `{}`, and it is not cosmetic
    ///
    /// Rust's `Display` for `f64` drops a trailing zero, so `LLOOM_47_…=4.0`
    /// renders `4` while `.bench/configurations.md` declares the arm `4.0` --
    /// and the two then **fail to compare equal across a sweep**, which is
    /// exactly the pooling failure the byte-count-verbatim rule exists to
    /// prevent for `BufferAlignment`. One decimal place is canonical for every
    /// bound a sweep would take; a bound needing more precision than that is
    /// not a ratio anyone is choosing deliberately.
    ///
    /// **Caught by `every_registry_arm_is_renderable_by_the_harness`**, which
    /// reads real `RESULT` lines rather than a rendering the test invented --
    /// the check §7.1a exists for, firing in the direction that a declared arm
    /// and a rendered arm had drifted apart.
    fn reuse_oversize_bound_arm() -> String {
        #[cfg(feature = "metal")]
        {
            match candle::kernels::metal::buffer_pool::reuse_oversize_bound() {
                Some(v) => format!("{v:.1}"),
                None => "off".to_string(),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a".to_string()
        }
    }

    /// The `BufferAlignment` arm: the granularity `buf_size` actually rounds to.
    ///
    /// `n/a` on a non-Metal build, following `CompletionWait`: there is no
    /// Metal allocator for an alignment to apply to, which is a different fact
    /// from having taken the default arm.
    fn buffer_alignment_arm() -> String {
        #[cfg(feature = "metal")]
        {
            candle::metal_backend::MetalDevice::buffer_alignment().to_string()
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a".to_string()
        }
    }

    /// The `MetalCflags` arm: the compile flags as the compiler resolved them.
    ///
    /// `none` rather than an empty string, for `MemoryBudget`'s reason: an arm
    /// that renders as the empty string is indistinguishable from a field the
    /// harness failed to fill in, and §7.1a-i is about exactly that confusion.
    /// The flags are rendered **verbatim** — two different `-D`s are two
    /// different metallibs and must not compare equal.
    fn metal_cflags_arm() -> String {
        #[cfg(feature = "metal")]
        {
            candle::kernels::metal_compile_flags()
                .map(|f| f.to_string())
                .unwrap_or_else(|| "none".to_string())
        }
        #[cfg(not(feature = "metal"))]
        {
            // No Metal compiler in this build, so there is no metallib for a
            // flag to reach. Distinct from `none`, which means "a Metal build
            // that set no flags".
            "n/a".to_string()
        }
    }

    /// The `MergedUnit` arm (#477): whether the sixteen non-MLX sources were
    /// compiled as one translation unit.
    fn merged_unit_arm() -> &'static str {
        #[cfg(feature = "metal")]
        {
            if candle::kernels::merged_translation_unit() {
                "on"
            } else {
                "off"
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a"
        }
    }

    /// The `CutPolicy` arm, with its period (`DESIGN.md` §13.5m).
    ///
    /// Exhaustive for [`attn_impl_arm`]'s reason: a third policy added to
    /// [`candle::metal_backend::CutPolicy`] fails to compile here rather than
    /// rendering silently as one of these two.
    ///
    /// The period is part of the arm name on **both** sides. `Counter(50)` and
    /// `Counter(55)` differ in session count — 11 against 10 on the shipping
    /// decode step — and #420 stage 3's control is a counter arm chosen to
    /// match the node arm's count, so a line saying only `Counter` would not be
    /// separable from the shipping default.
    fn cut_policy_arm(&self) -> String {
        #[cfg(feature = "metal")]
        {
            match self.cut_policy {
                Some((candle::metal_backend::CutPolicy::DispatchCounter, period)) => {
                    format!("Counter({period})")
                }
                Some((candle::metal_backend::CutPolicy::NodeIndex { nodes_per_cb }, _)) => {
                    format!("Node({nodes_per_cb})")
                }
                Some((candle::metal_backend::CutPolicy::NodeAligned, period)) => {
                    format!("NodeAligned({period})")
                }
                // No Metal device, so there are no command buffers to cut.
                // Distinct from either arm rather than folded into the default.
                None => "n/a".to_string(),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a".to_string()
        }
    }

    /// The math mode this build will compile its kernels under: the
    /// switch the kernel compiler reads (`backend.metal.compiler.fast_math`),
    /// so the recorded value is the one that decides codegen. Note the off
    /// arm is `Relaxed`, **not** `Safe` — `MTLMathMode` has three values and
    /// neither arm available today is the strict one (§2.3.5, §2.3.9).
    fn math_mode_name() -> &'static str {
        if candle::tuning::tuning().metal.fast_math {
            "Fast"
        } else {
            "Relaxed+Precise"
        }
    }

    fn executor_name() -> &'static str {
        // This harness never selects the ICB executor -- #115 ships `--icb` on
        // the trace harness, and §17 Phase 2 item 10 forbids timing a
        // 78 %-coverage executor against a full baseline anyway.
        "Classical"
    }

    fn hazard_key_name() -> &'static str {
        #[cfg(feature = "metal")]
        {
            match candle::metal_backend::hazard_key() {
                candle::metal_backend::HazardKey::Pointer => "Pointer",
                candle::metal_backend::HazardKey::Range => "Range",
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            "n/a"
        }
    }

    fn announce(&self) {
        eprintln!("configuration: {}", self.config_line());
    }
}

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode CPU/GPU split and dispatch-count profile")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Decode tokens to generate and time.
    #[arg(long, short = 'n', default_value_t = 200)]
    n: usize,

    /// Leading decode tokens excluded from the steady-state average.
    ///
    /// They still run and are still reported; they are just not what the
    /// roofline comparison is about.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Concurrent sequences per decode step — `DESIGN.md` §13.4a's `B`.
    ///
    /// **N sequences advancing in LOCKSTEP FROM THE SAME PROMPT.** Every row
    /// runs the same prompt and, under argmax, produces the same tokens, so
    /// they step together for the whole run and no scheduler is needed.
    ///
    /// # What this measures, and what it deliberately does not
    ///
    /// §13.4a's central claim is that **weight traffic is per step, not per
    /// token**: one 5.394 GB sweep serves all `B` rows, so aggregate throughput
    /// should rise with `B` while step time barely moves. That claim is
    /// arithmetic (64.8 tok/s at B=1 against a projected 518.6 at B=8) and had
    /// never been run, because no harness could set `B` at all. This flag is
    /// the instrument; measuring with it is #251's.
    ///
    /// **Ragged batches are out of scope**, and that is a scope decision rather
    /// than a limitation discovered late. Different-length sequences need
    /// either padding or a scheduler, and both are engine questions (§10.3's
    /// paging, #148). Uniform lockstep is enough to measure the
    /// weight-amortisation claim, which is the claim that matters — and it is
    /// the only shape whose correctness gate is exact: at `B > 1` with
    /// identical prompts, **every row must produce the token stream B=1
    /// produces**, which `--batch-check` asserts.
    ///
    /// `1` is the default, so an unflagged run is byte-for-byte the B=1 path
    /// every recorded figure belongs to.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Assert that every batch row's token stream is identical.
    ///
    /// The correctness gate for `--batch`, and it is **stronger than a digest**
    /// (issue #249): a divergence between rows means the batch dimension is
    /// leaking — one sequence's output depending on its neighbours, which
    /// §2.3.3 #7 forbids and which is exactly the defect a batched harness
    /// would otherwise hide. Checked every step rather than at the end, so the
    /// failure names the step and the row.
    ///
    /// On by default: the check is a slice comparison over `B` `u32`s per
    /// token, and a gate that has to be remembered is a gate that is not run
    /// (§11.3j).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    batch_check: bool,

    /// Speculative window width `K` — `DESIGN.md` §10.2a's `advance`/`resolve`.
    ///
    /// **This is #284: the port of `lfm2-determinism`'s verify loop (#89) onto
    /// the harness that has GPU-busy timing, `--batch` and the run store.** The
    /// direction is §14.5 step 1's, and the reason is that
    /// **GPU-busy per accepted token has never been produced for any
    /// speculative run**: #89's cost curve (K=2 at 1.245x, K=8 at 0.587x) is
    /// wall-clock `elapsed_ms` from `lfm2-determinism`, a harness whose own
    /// documentation says that field *"is not comparable between arms"*. So the
    /// most-cited speculative figures in this project rest on a quantity their
    /// own harness disclaims, and this flag is what makes the honest one
    /// takeable.
    ///
    /// # The denominator, which is why this is not just a flag
    ///
    /// §10.2i states the obstacle exactly: the profiler's loop *"emits one token
    /// per step and times it, where a verify pass consumes K and emits between 1
    /// and K, so `wall_ms_per_token` needs a denominator that does not exist
    /// there."* It exists now — `spec_accepted` — and the per-token fields are
    /// computed over **accepted tokens** rather than over steps whenever a
    /// window ran. `wall_ms_per_token` keeps its name and its meaning at `K = 0`
    /// (where a step *is* a token), so every recorded row stays comparable.
    ///
    /// `0` is the default and disables the mechanism entirely — no `advance`,
    /// no `forward_all`, the ordinary decode loop — so an unflagged run is the
    /// path every recorded figure belongs to.
    ///
    /// Requires `--conv-state rotating:<K'>` with `K' >= K` and
    /// `--kv-append in-place`; `Cache::advance` refuses everything else before
    /// any state is written (§10.2h).
    #[arg(long, default_value_t = 0)]
    speculate: usize,

    /// Which proposer drives `--speculate`: `oracle` or `wrong:<N>`.
    ///
    /// Both are sequences rather than models, and that is the point (#89):
    /// **greedy speculation is output-identical by construction**, so the
    /// mechanism can be measured without a draft model at all. The proposer
    /// decides only *how many* proposals are accepted, never *what* is emitted.
    ///
    /// * `oracle` proposes the token the target would have emitted, so every
    ///   position is accepted and the window is the best case.
    /// * `wrong:<N>` corrupts every `N`-th proposal, so a rejection happens on a
    ///   schedule and the rollback is exercised on most windows. This is the
    ///   **engagement proof** §2.4 requires: an arm that cannot be shown to have
    ///   rejected anything has not exercised `resolve`, and #89 held the output
    ///   invariant under 40 and 80 rejections with the stream unmoved.
    #[arg(long, default_value = "oracle")]
    spec_proposer: String,

    /// Assert every row accepts the same prefix length at `--batch N`.
    ///
    /// **The per-row accept test, and the gate that catches a leak** (#284,
    /// #252). See the acceptance loop for why uniformity is a *theorem* here
    /// rather than an assumption, and therefore why a violation is a defect
    /// rather than a tuning artifact.
    ///
    /// On by default, for `--batch-check`'s reason: a gate that has to be
    /// remembered is a gate that is not run (§11.3j).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    spec_accept_check: bool,

    /// Install `DESIGN.md` §9.5's admission check at the given fraction of
    /// `recommendedMaxWorkingSetSize`.
    ///
    /// **Off by default, and this is the first harness to carry it** — §9.5l
    /// records that *no* harness installs a budget, so `admission::Budget` has
    /// never refused anything on a real run. Issue #249 makes that reachable
    /// for the reason §9.5b gives: **KV × B is the only class that can consume
    /// the machine on its own**, and `B` is the axis this flag's sibling adds.
    ///
    /// The predicted peak is `weights + B × capacity × 16 KiB + conv + act +
    /// scratch`, so raising `--batch` walks it toward the refusal §9.5b
    /// computes at `B=16 ctx 128k` — 41.4 GB, which the real denominator
    /// (55.663 GB × 0.65 = 36.181 GB) **refuses**. A batched harness that OOMs
    /// where admission should have refused is a defect in admission, and
    /// finding it is a result.
    ///
    /// `0.65` is §9.5k's table value and **not a measurement** (§9.5l), which
    /// is why it is a flag rather than a constant here too.
    #[arg(long)]
    memory_budget: Option<f64>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// 0 selects argmax. Sampling choice does not affect the timing, but fixing
    /// it keeps the generated text identical across runs so a timing change
    /// cannot be a different-text change in disguise.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    dtype: Option<String>,

    /// Print every token's wall and GPU time, not just the summary.
    #[arg(long)]
    per_token: bool,

    /// Split the sample into its GPU-work and host-stall halves (lloom #172).
    ///
    /// Ported from `bench/issue-172-sample-cost`, which `DESIGN.md` §11.5a
    /// records as the mechanism that produced the 441 µs figure and that could
    /// not be re-run from the tip. Synchronizes between the `argmax` dispatch
    /// and the 4-byte `to_scalar` so the two are attributed separately.
    ///
    /// **Diagnostic, not a total.** The extra synchronization inflates
    /// `sample / token`, so only the split is readable -- §11.5a measures the
    /// inflation at 17 % and records that pooling this arm with the clean one
    /// is what a gate fired on.
    #[arg(long)]
    sample_split: bool,

    /// Sample inside the forward pass's command buffer (lloom #319).
    ///
    /// The default path samples the PREVIOUS iteration's logits, so `argmax`
    /// opens a fresh command buffer for one dispatch and `to_scalar` waits on a
    /// second submit -- §11.2a's fixed ~0.5 ms round trip, measured at
    /// **441 µs/token** in §11.5a.
    ///
    /// With this on the `argmax` is enqueued while the forward pass's command
    /// buffer is still open, so the readback reads an already-computed value.
    /// **One command-buffer boundary moved by one dispatch**: no
    /// `MTLSharedEvent`, no token ring, no dependency on §11.5's bounded
    /// window.
    ///
    /// > **#567 deleted the per-token drain this used to lean on** (`DESIGN.md`
    /// > §11.5r). Two clauses here said the queue was *"already drained"* by a
    /// > `device.synchronize()` and that *"the existing `synchronize()` drains
    /// > both"*; both are struck, because there is no longer a per-token one.
    /// > **The arm still does what it claims** — the argmax rides the forward
    /// > pass's buffer either way, which is the whole mechanism — but what
    /// > forces the value to be ready is now `to_scalar`'s own
    /// > `flush_and_wait_current` rather than a drain that ran first. The
    /// > **441 µs it is measured against was taken under the drain** and is not
    /// > re-derived here.
    #[arg(long)]
    sample_in_buffer: bool,

    /// Run the decode loop on §11.5g's non-blocking token path (lloom #567).
    ///
    /// The sampled token is left on the device, written into a persistent input
    /// slot in place, and the EOS decision is made GPU-side.
    ///
    /// # Why this flag is on THIS harness, which is the whole of #567
    ///
    /// The three async arms — this, `--nonblocking-eos` and `--completion-wait`
    /// — were selectable only from `lfm2-release`, which reports **no
    /// `step_ms_per_token`**; the two epilogues are selectable only here and on
    /// `lfm2-determinism`, neither of which carried an async arm. So **no
    /// binary could select all four**, and §2.5 requires the saving be quoted in
    /// `step_ms_per_token` and never in `wall` (#299's double count). The
    /// measurement every one of these mechanisms owes was not takeable.
    ///
    /// **They compose, which is why one arm at a time is not enough.** The
    /// epilogues remove dispatches and the async arms remove a wait; #503
    /// measured that a barrier costs the overlap it forbids, so removing
    /// dispatches while a blocking wait remains — or the wait while the
    /// dispatches remain — each measures something other than the design.
    ///
    /// **Off by default and the default arm is byte-identical to the path that
    /// ships** (§7.1a). Requires `--temperature 0`: `sample_argmax_on_device` is
    /// the only sampler that leaves the token on the device, so a temperature
    /// run would take the host readback while reporting the ring arm — §7.1a-i's
    /// wrong-arm shape, which compares equal to every real run of that arm.
    #[arg(long)]
    token_ring: bool,

    /// How the host waits for the ring to publish, on `--token-ring` runs.
    ///
    /// `spin`, `spin-hint`, `wfe` (aarch64), `sleep`, or `sleep:<micros>`.
    /// §11.5g reports `surviving_count` = 3 on *which arm is best* and that is
    /// still not resolved. §11.5l measures the arms as indiscriminable **on the
    /// arm as wired**, because nothing spins — and §11.5o records that they
    /// become discriminable once `--nonblocking-eos` composes with an async
    /// `--completion-wait`, which is a combination only this harness can now
    /// select.
    ///
    /// An unrecognised value is an error rather than a fallback (§2.4, after
    /// #69's vacuous determinism run).
    ///
    /// **Unset takes `PollMechanism::default()` (#578).** This carried
    /// `default_value = "spin"`, which shadowed the type's own default — so
    /// flipping the default would have changed nothing here and the `RESULT`
    /// line would have gone on reporting the CLI's pin. One choice, one source.
    #[arg(long)]
    poll: Option<String>,

    /// Read the EOS flag **without blocking**, on `--token-ring` runs (#535).
    ///
    /// §11.5j records the ring as wired installing *"a hint before a blocking
    /// read, not a spin-until-published loop"* — `wait()` fires once per token
    /// and the flag read that follows reaches `to_cpu`'s
    /// `flush_and_wait_current` anyway. This is the replacement §11.5k names as
    /// **the largest single candidate** on this seam.
    ///
    /// Off by default (§7.1a). Requires `--token-ring`; refused otherwise
    /// rather than silently ignored, since the flag it polls is published by the
    /// ring and without it there is nothing to poll.
    #[arg(long)]
    nonblocking_eos: bool,

    /// How the host waits for a committed command buffer (§11.5o, issue #559;
    /// §11.5t, issue #583).
    ///
    /// `synchronize` (the default and the incumbent), `shared-event`,
    /// `shared-event-sleep[:<micros>]`, `shared-event-wait[:<timeout_ms>]`, or
    /// `shared-event-listener[:<timeout_ms>]`.
    ///
    /// **The five arms are four designs, not five tunings** (#583). The two
    /// middle arms are **poll loops** over `signaledValue`; `shared-event-wait`
    /// is `waitUntilSignaledValue:timeoutMS:` and **parks in the runtime**, so
    /// it reports `completion_polls=0` structurally. Its period is a **timeout
    /// in milliseconds** — a fail-safe that falls back to blocking, not a
    /// latency knob — where the sleeping arm's is a sleep in microseconds.
    ///
    /// **`shared-event-listener` is the fourth design and the only one where
    /// the completion is DELIVERED rather than waited for** —
    /// `notifyListener:atValue:block:` runs a block on a GCD worker, which
    /// wakes this thread through a condvar. It also reports
    /// `completion_polls=0`, and **that zero is not its engagement proof**: a
    /// listener that has stopped firing falls back to blocking, is correct,
    /// moves no digest, and reads the same 0. `completion_fires` is what
    /// separates them, and a run selecting this arm with `completion_fires=0`
    /// has **not taken it**.
    ///
    /// **A result from this arm carries a blind spot that must be stated where
    /// it is used**: the block runs on a GCD queue, and §15.1i measures that
    /// TSan cannot model the happens-before edge through Metal's dispatch — a
    /// report there survives `-Zbuild-std` and is still false, so a genuine
    /// ordering defect at that boundary would be indistinguishable.
    ///
    /// **Three axes on one seam, at different heights**, and they compose
    /// rather than substitute:
    ///
    /// | flag | what it changes |
    /// |---|---|
    /// | `--poll` | how the host idles *between* reads |
    /// | `--nonblocking-eos` | whether the **flag read** blocks (#535) |
    /// | `--completion-wait` | whether the **command-buffer wait underneath** blocks (#559) |
    ///
    /// `--nonblocking-eos` removes the block from the *one* read the ring makes
    /// per token; this removes it from every `flush_and_wait_current` on the
    /// path — which is what the baseline arm reaches through the sampler and
    /// what `to_cpu` reaches at every other call site. **So this one is not
    /// gated on `--token-ring`**: it is the only async arm that changes the
    /// default path, which is precisely what makes it measurable against the
    /// epilogues.
    ///
    /// **`synchronize` stays the default.** Every canonical digest and every
    /// `.bench/` row belongs to it, and §7.1a's discipline is that a default
    /// flip is its own argued decision with a number in hand.
    #[arg(long, default_value = "synchronize")]
    completion_wait: String,

    /// Run the front end's per-token seam on the decode thread (lloom #412).
    ///
    /// Reproduces ambrogio's streaming shape — `Tokenizer::decode` over the
    /// pending run, `frag.to_string()`, `blocking_send` into a
    /// `tokio::sync::mpsc::channel(32)` a real consumer drains — so what #412
    /// asks the price of is an **arm in a rotation** rather than an argument.
    /// `seam.rs` says which lines it is a copy of and what a copy costs.
    ///
    /// **Off is the null arm and it is the shipped decode loop**, not a seam
    /// that does nothing: with this off no channel exists, no runtime exists,
    /// and no `String` is allocated. The difference between the two arms is the
    /// whole quantity.
    #[arg(long)]
    seam: bool,

    /// The seam's channel capacity. 32 matches `lfm2.rs`'s `channel(32)`.
    ///
    /// Exposed because it is the term that decides whether `blocking_send` ever
    /// **blocks**, which is #412's second question — a slow consumer applying
    /// backpressure to generation. A capacity of 1 is the arm that makes a stall
    /// reachable; the default is the shipped value and must stay it.
    #[arg(long, default_value_t = 32)]
    seam_capacity: usize,

    /// Detokenize the pending run even on the null arm (lloom #412).
    ///
    /// On by default, because ambrogio's decode loop calls `Tokenizer::decode`
    /// whether or not anyone is streaming — it is how the reply text is built at
    /// all. Attributing it to the seam would charge the streaming arm for a cost
    /// both arms pay, which is §15.2 #11a's shape. `--no-detokenize` removes it
    /// from **both** arms, which is what prices the term itself.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    detokenize: bool,

    /// Write row 0's generated token ids, one per line, to this path.
    ///
    /// **The gate for `--sample-in-buffer`** (issue #319). That arm moves which
    /// command buffer carries the `argmax` and must not change which token it
    /// selects, so the two arms' streams are compared directly rather than
    /// through a timing. This harness has no digest of its own -- `sha256.rs`
    /// is `lfm2-determinism`'s -- and a file of ids is `cmp`-able, which is what
    /// the gate needs. Off the timed path: written after the loop.
    #[arg(long)]
    dump_tokens: Option<PathBuf>,

    /// Label printed with the result, to tag a run inside a batch.
    #[arg(long, default_value = "")]
    label: String,

    // ---- the run store (lloom issue #171) -------------------------------
    /// Append a serialized run record to this JSONL file, at exit.
    ///
    /// Nothing is written without it, and a write failure is a warning rather
    /// than a failed run: the measurement has already happened by then.
    #[arg(long)]
    run_store: Option<String>,

    /// Run id, shared with the `lloom-sample` telemetry for the same run so the
    /// two files can be joined without guessing.
    #[arg(long)]
    run_id: Option<String>,

    /// Path of the `lloom-sample` telemetry recorded around this run, recorded
    /// in the run record so a reader can find it.
    #[arg(long)]
    telemetry_path: Option<String>,

    // ---- variant axes (`DESIGN.md` §7.1) --------------------------------
    //
    // The same selectors `lfm2-dispatch-trace` carries, so a timing run and a
    // dispatch-count run can name the same configuration. Without them a
    // ms/token figure is silently all-defaults, which is what makes today's
    // `.bench/` rows unreproducible (#105).
    //
    // Each defaults to the shipped default, so an unflagged run is the
    // configuration that produced 18.763 ms/token.
    /// Serve decode activations from an activation arena (`DESIGN.md` §9.2).
    ///
    /// The first `--arena-record-steps` decode steps run on the pool to derive
    /// the plan; the arena is installed after them. Those steps are excluded
    /// from the steady-state average in addition to `--warmup`, since they
    /// measure a different allocator.
    #[arg(long)]
    arena: bool,

    /// `packed` (#69's, slots reused by liveness) or `non-aliasing` (§9.3's
    /// reference layout, every value in its own slot).
    #[arg(long, default_value = "packed")]
    arena_layout: String,

    /// Where arena offsets are computed: `cpu` (#69) or `gpu` (#70's bump
    /// allocator). Requires `--arena` and `--arena-layout non-aliasing`.
    #[arg(long, default_value = "cpu")]
    arena_offsets: String,

    /// How intra-encoder hazards are keyed: `pointer` (candle's default) or
    /// `range` (#69's route (a), interval overlap).
    ///
    /// Left unset the selector is untouched, so `CANDLE_METAL_HAZARD_KEY` is
    /// honoured -- calling the setter unconditionally consumes the `OnceLock`
    /// and silently pins the default, which is the vacuous-instrument failure
    /// §2.4 records from #69.
    #[arg(long)]
    hazard_key: Option<String>,

    /// LFM2 decode attention: `generic` (the default path), `sdpa` (#97's
    /// GQA-native `sdpa_vector`) or `flash` (#116's FlashDecoding).
    ///
    /// `flash` was added by #116 and this help text still named two arms until
    /// #241 — the same omission as the `RESULT` line's, in the place a user
    /// reads to find out what the flag takes.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV tokens per page for `--attn flash` — the **allocation** granularity.
    ///
    /// §10.4 proposes 256 and marks it **UNVERIFIED**. A flag rather than a
    /// constant because §10.3d establishes page size as two dispatch-tier
    /// numbers, so 16, 256 and 1024 are one field apart — which is what
    /// *"must not foreclose it"* requires. Verifying which wins is #61's axis.
    #[arg(long, default_value_t = 256)]
    flash_page_size: usize,

    /// Pages per chunk for `--attn flash` — the `k` of
    /// `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk equal **by fiat**; §9.1d records that
    /// a page (allocation) and a tile (computation) are optimised against
    /// disjoint cost functions, so **a sweep holding `k = 1` cannot separate a
    /// page-size effect from a tile-size one**. 1 is what §10.4 specifies; the
    /// flag is what makes the two separable.
    #[arg(long, default_value_t = 1)]
    flash_k: usize,

    /// How `--attn flash` sizes the scratch class: `grow` (the default),
    /// `reserve` or `bucket` — `DESIGN.md` §9.1a's three policies, issue #234.
    ///
    /// **Three arms have been compiled since #71 and none reached a model path
    /// until now.** #116 sized these buffers to the live `kv_len` on every
    /// call, which is `Grow` in effect and by consequence rather than by
    /// choice; this flag is what turns three compiled arms into three
    /// measurable ones. `grow` is what shipped, so the default allocates
    /// byte-for-byte what it did.
    ///
    /// `reserve` reserves for the **cache's** capacity (`--kv-capacity`, 4096
    /// by default) rather than for `max_position_embeddings`: the first is the
    /// largest `kv_len` a run can reach and the second is the model's
    /// positional ceiling, 128000. `bucket` takes the smallest rung of
    /// `BUCKET_LADDER` (8k/32k/128k in tokens) that covers the step.
    #[arg(long, default_value = "grow")]
    flash_scratch_sizing: String,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset -- issue #142).
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// KV cache element width: `off` (the f16 default), `int8` or `int4`
    /// (issue #218, `DESIGN.md` §10.7).
    ///
    /// **A moved digest is EXPECTED on this axis and is not itself the
    /// finding.** Quantization is a real precision loss rather than a
    /// reordering, so what is owed is the ulp delta's **magnitude** and its
    /// **growth slope** against §10.7a's ladder, per §2.3.3c's tier-2
    /// sub-distinction. Magnitude is primary: #128's defect ratio is 0.787, so
    /// reading growth first would have called a real defect benign.
    #[arg(long, default_value = "off")]
    kv_quant: String,

    /// How decode writes conv state: `shuffle` (§6.1's `narrow` + `Tensor::cat`,
    /// the default), `ring[:K[:slack]]` (the sliding window, §10.2e) or
    /// `rotating[:K]` (§10.2a's rotating index, §10.2g).
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one fused
    /// dispatch (`ConvFused`, §12.2 row 1): `shortconv_step` on the `shuffle`
    /// state, `shortconv_ring` on `rotating`. Moves the logits digest.
    #[arg(long, default_value_t = false)]
    conv_fused: bool,

    /// The configuration preset the axis flags apply over: `none` (this
    /// harness's defaults), `crown` (what `lloom chat` decodes at one
    /// sequence) or `pool` (the rows' preset). The presets are the engine's
    /// own functions (`lloom::models::lfm2_presets`); under one, only the axis
    /// flags given on the command line override it, and the resolved axes are
    /// on the `configuration:` line.
    #[arg(long, default_value = "none")]
    preset: String,

    /// Fuse SiLU onto `w1`'s GEMV store (#484's kernel, wired by #509,
    /// `DESIGN.md` §12.2i). Moves the digests by construction.
    #[arg(long, default_value_t = false)]
    gemv_silu: bool,

    /// Fuse the two residual adds onto their producing GEMV's store (#509,
    /// §12.2i). **Needs no compile flag since #516**; it used to require
    /// `-DLLOOM_509_RESIDUAL_EPILOGUE` through `LLOOM_434_METAL_CFLAGS` and
    /// failed at pipeline load without it. Moves the digests by construction.
    #[arg(long, default_value_t = false)]
    gemv_residual: bool,

    /// How the SwiGLU product is dispatched (`GemvGate`, #732): `off` |
    /// `mul` | `swiglu`. Moves the digests by construction.
    #[arg(long, default_value = "off")]
    gemv_gate: String,

    /// Which SiLU implementation the 30 `usilu` dispatches resolve to
    /// (#599, `DESIGN.md` §12.2l): `exact` | `recip` | `literal` | `coarse4` |
    /// `coarse1` | `hardswish` | `relu`, in ascending error.
    ///
    /// **`exact` is the default and resolves the name that always shipped.**
    /// Every arm but `exact` and `recip` moves `logits_sha256` by
    /// construction — an approximation to SiLU computes different numbers, and
    /// that is the axis rather than a defect.
    #[arg(long, default_value = "exact")]
    silu_impl: String,

    /// Decode steps recorded before the arena is installed.
    ///
    /// Two, not one: comparing two steps is what separates an activation from
    /// session state, since an allocation whose size moved between them is
    /// sized by `kv_len` (§9.1, #68 finding 4).
    #[arg(long, default_value_t = 2)]
    arena_record_steps: usize,

    /// Serve the arena's session-state declines from an owned region (#47,
    /// `DESIGN.md` §9.2x).
    ///
    /// # What this is, against `--arena` which it requires
    ///
    /// The arena is already an owned slab — one `MTLBuffer`, slots at fixed
    /// offsets — but it serves only the ordinals whose size is **the same every
    /// token**. Measured on this tree, every decline it makes is
    /// `declined(session)`: the `Tensor::cat` KV copies under `--kv-append cat`,
    /// which grow one token wider each step and carry **84.1 % of the residual
    /// overhang** (§9.2u). This flag gives that population a second owned
    /// allocation, sized once from the **measured** per-token stride to
    /// `--kv-capacity` tokens, so no later token allocates for it.
    ///
    /// **`off` is the default and is what ships.** An unflagged run is the path
    /// every recorded figure belongs to.
    ///
    /// Requires `--arena`: the reservations are derived from the arena
    /// recording's two steps, which is the only thing in the tree that measures
    /// a per-token stride rather than assuming one.
    #[arg(long, default_value_t = false)]
    session_region: bool,

    /// Tokens of KV reserved per sequence under `--kv-append in-place`.
    ///
    /// # Why this flag has to exist for issue #61, and why it is not a policy
    ///
    /// `Cache::new` hard-wires `DEFAULT_KV_CAPACITY`, which is **4096**, and
    /// that value is deliberately a placeholder: `Cache::new_with`'s doc
    /// comment says so in as many words -- *"above the 2720 ceiling with
    /// headroom ... and small enough that nobody mistakes it for a considered
    /// long-context answer"*, with **"what would decide it is a context-length
    /// curve -- issue #61"**. So the `InPlace` arm could not be measured past
    /// `kv_len` 4096 by any harness, which is a **quarter of the way** to the
    /// smallest interesting point of that curve and 3 % of the 128k target
    /// §2.1 names.
    ///
    /// **This selects an existing parameter; it does not choose the policy.**
    /// `Cache::new_with` already takes the capacity and already refuses loudly
    /// when it is exhausted (§6.2b finding 4); all that was missing was a way
    /// for a *measurement* to ask for a different one. `DEFAULT_KV_CAPACITY`
    /// is untouched, so an unflagged run is byte-for-byte what shipped --
    /// §7.1a's shape, applied to a harness flag rather than to an axis.
    ///
    /// **It is `Reserve`, so it is allocated whether or not it is reached**:
    /// at 16 KiB/token (§5.6's geometry, not §5.6's table) a capacity of
    /// 131072 is **2.147 GB** allocated up front. That is inside every
    /// prediction in `measurements/issue-61-raw/sweep_admission.py`, and the
    /// prediction is what §9.5j item 1 requires *before* the arm runs.
    ///
    /// Ignored under `--kv-append cat`, which has no capacity: it reallocates
    /// the whole cache every token (§6.2), which is the thing being measured.
    #[arg(long, default_value_t = lloom::models::lfm2::DEFAULT_KV_CAPACITY)]
    kv_capacity: usize,

    /// Print a progress line to stderr every N decode tokens. 0 disables it.
    ///
    /// Issue #61 requires a long-context run to report **where it stopped and
    /// why** rather than truncating silently, and a run that dies having printed
    /// nothing cannot. The line carries `kv_len` and the pool's live / free /
    /// **pending** bytes, the last being §9.5k's *"capped by nothing"* term and
    /// the one whose unbounded growth §3.4a-iv names as the diagnostic shape.
    ///
    /// Default 0 so a normal timed run is byte-for-byte unchanged: the write is
    /// outside the per-token window but inside the loop, and a measurement
    /// should not pay for an instrument it did not ask for (§6.4a).
    #[arg(long, default_value_t = 0)]
    progress_every: usize,

    /// Write one JSON object per `--progress-every` sample to this path.
    ///
    /// The human progress line is for watching a run; this is the artifact.
    /// `CONTRIBUTING.md` §3.2a #4 requires the raw series be committed beside
    /// any plot so it can be re-analysed, and §2.6's rule is that a number
    /// existing only in a PR body is unmeasured.
    ///
    /// **Bytes are written as bytes**, not rounded to GB: issue #206's second
    /// acceptance item asks for a per-token create/destroy *rate*, which is a
    /// difference of two cumulative counters, and a series pre-rounded for
    /// display cannot be differenced to recover it.
    ///
    /// Ignored without `--progress-every`, since that is what decides when a
    /// sample is taken.
    #[arg(long)]
    mem_jsonl: Option<PathBuf>,

    /// Record #171's by-class memory timeline to this path, as JSONL.
    ///
    /// # What this is, against `--mem-jsonl` which sits beside it
    ///
    /// They are **not** two spellings of one thing, and running both is the
    /// point of issue #205 rather than redundancy:
    ///
    /// | | `--mem-jsonl` | `--run-telemetry-jsonl` |
    /// |---|---|---|
    /// | schema | one wide row per sample | `lloom-sample`'s long rows, so the two files concatenate |
    /// | reports | pool, **`device_allocated`**, **`phys_footprint`** | §9.1's five classes, on the `CLOCK_UPTIME_RAW` axis |
    /// | sees the retention | **yes** -- it reads outside the allocator | **no, by construction** |
    ///
    /// **That last row is the finding this flag exists to make visible, not a
    /// defect in it.** §6.3e established that Metal retains every buffer the
    /// pool evicts, and that those bytes are *"outside every quantity
    /// `occupancy()` reports, by construction -- and outside §9.1's five
    /// classes, all of which are ours. A by-class timeline would have shown
    /// five flat curves."* So this series is the instrument #171 built and
    /// §16 P0 #7 waited on, exercised at long context for the first time, and
    /// the honest result is a **bound on what it can answer**. Recording it
    /// beside `--mem-jsonl` is what makes that bound a measurement rather than
    /// an assertion: five flat curves against a device reading of tens of GB,
    /// in one plot.
    ///
    /// Requires the `metal-run-telemetry` feature -- which did not exist for
    /// `candle-examples` until #205 and is why this had never been run. Without
    /// it the run refuses rather than writing an empty file, because a build
    /// that *cannot* record and one that *did not* are different states (§2.4,
    /// after #69's vacuous determinism arm).
    #[arg(long)]
    run_telemetry_jsonl: Option<PathBuf>,
}

/// Bytes of language-model weights, computed from the loaded config rather than
/// quoted from `DESIGN.md` §5.5.
///
/// Recomputing it here means the roofline denominator tracks whatever
/// configuration actually ran: if `intermediate_size` had silently fallen back
/// to candle's 8192, the byte count would change and the reconciliation would
/// visibly stop working, instead of quietly comparing against the wrong number.
///
/// Counts each weight once. `tie_word_embeddings` is true for this checkpoint,
/// so the embedding matrix is also the lm_head and is not double counted.
fn language_weight_bytes(config: &Config, layer_is_attn: &[bool], dtype_bytes: usize) -> usize {
    let d = config.hidden_size;
    let ffn = config.intermediate_size;
    let kv_dim = config.num_key_value_heads * (d / config.num_attention_heads);

    let mut params = 0usize;
    // Tied embedding: also the lm_head.
    params += config.vocab_size * d;
    // Final norm.
    params += d;

    for &is_attn in layer_is_attn {
        // Both layer kinds: two norms and the SwiGLU triple.
        params += 2 * d;
        params += 2 * ffn * d; // w1 (gate), w3 (up)
        params += ffn * d; // w2 (down)

        if is_attn {
            params += d * d; // q_proj
            params += 2 * kv_dim * d; // k_proj, v_proj
            params += d * d; // out_proj
            params += 2 * (d / config.num_attention_heads); // q/k layernorm
        } else {
            params += 3 * d * d; // in_proj [3d, d], already fused BCx
            params += d * 3; // depthwise conv, k=3
            params += d * d; // out_proj
        }
    }

    params * dtype_bytes
}

/// Bytes one attention layer's FlashDecoding partials **ask for** at the
/// selected sizing arm (`DESIGN.md` §9.1a, issue #234).
///
/// # Why this is on the `RESULT` line
///
/// §2.4: an instrument that cannot be shown to have engaged has measured
/// nothing, and the proof must come **from a quantity the flag should have
/// changed** rather than from the flag being echoed back. #257 found a third
/// species of that failure — an axis rendering correctly for a mechanism that
/// **did not run** — and `ScratchSizing` is exposed to exactly it: the arm is
/// inert on `--attn generic` and `--attn sdpa`, which allocate from this class
/// not at all.
///
/// So this figure is **0 unless FlashDecoding is selected**, and otherwise
/// differs per arm: the live count under `Grow`, the reservation under
/// `Reserve`, a rung under `Bucket`. Two runs differing only in this flag whose
/// `scratch_asks_bytes` agree did not select what they claim to have.
///
/// # It is a prediction, and it says so
///
/// Computed from the arm and `kv_len` rather than read from the allocator, so
/// it is *what the policy asks for* and not *what was allocated* — §5.5a's
/// caution about a harness quoting its own computed figure as evidence. The
/// observed side is `allocated_bytes` under `--mem-jsonl`, which knows nothing
/// about this arithmetic. The field is named `asks` for that reason.
///
/// `+ 2` per (head, chunk) is the online-softmax `m` and `l`, and the count is
/// **query** heads because a partial is downstream of GQA's register broadcast
/// (§9.1a: using the 8 KV heads would under-size the class 4×).
fn flash_scratch_ask_bytes(
    axes: &Axes,
    config: &Config,
    kv_len: usize,
    kv_capacity: usize,
) -> usize {
    if config.attn_impl != AttnImpl::FlashDecoding {
        return 0;
    }
    let chunk_size = (config.flash_page_size * config.flash_pages_per_chunk).max(1);
    let live = kv_len.div_ceil(chunk_size);
    // The same three arms `Sizing::sized_chunks` computes, and the same bounds:
    // `Reserve` against the cache's capacity, `Bucket` against the ladder in
    // chunks. A refusal is impossible here — the run completed, so the op
    // accepted these inputs — and is rendered as the live count rather than
    // panicking after a successful measurement.
    let chunks = match axes.flash_scratch_sizing {
        FlashScratchSizing::Grow => live,
        FlashScratchSizing::Reserve => kv_capacity.div_ceil(chunk_size).max(live),
        FlashScratchSizing::Bucket => [8_192usize, 32_768, 131_072]
            .iter()
            .map(|&kv| kv.div_ceil(chunk_size))
            .find(|&rung| live <= rung)
            .unwrap_or(live),
    };
    let head_dim = config.hidden_size / config.num_attention_heads;
    config.num_attention_heads * chunks * (head_dim + 2) * 4
}

/// What a `--speculate` run accumulated, and the denominator the per-token
/// fields are computed over.
///
/// **`accepted` is the denominator §10.2i says does not exist here.** That
/// section states the obstacle exactly: the profiler's loop *"emits one token
/// per step and times it, where a verify pass consumes K and emits between 1 and
/// K, so `wall_ms_per_token` needs a denominator that does not exist there."*
/// This is it. A window's wall time is divided by the tokens it *kept*, not by
/// the positions it *proposed* — which is what makes the figure GPU-busy **per
/// accepted token**, the quantity #273's prediction is stated in and the one
/// §5.6-style byte accounting cannot substitute for.
///
/// `proposed` is carried beside it rather than derived, because the two answer
/// different questions: `accepted / proposed` is the acceptance rate a proposer
/// is judged on, and `accepted` alone is what a token cost is divided by.
#[derive(Clone, Copy, Debug, Default)]
struct SpecCounters {
    /// Verify passes run. **Zero after a `--speculate` run is a vacuous
    /// instrument** (§2.4, after #69's determinism run reported a passing digest
    /// for a path the flag never reached), so it is asserted rather than printed.
    windows: usize,
    /// Positions proposed across every window — the denominator of the
    /// acceptance rate.
    proposed: usize,
    /// Positions whose state survived `resolve`. **The per-token denominator.**
    accepted: usize,
    /// Proposals the `wrong:<N>` arm deliberately corrupted. The engagement
    /// proof for that arm: an arm that cannot be shown to have rejected anything
    /// has not exercised the rollback at all (#89).
    corrupted: usize,
    /// Steps at which two batch rows accepted different prefix lengths, with the
    /// row and the two lengths. `None` is the expected result and the theorem
    /// below says why.
    accept_divergence: Option<(usize, usize, usize, usize)>,
}

/// Record the token stream this configuration emits **without** speculation.
///
/// Runs an ordinary decode on a **throwaway cache** and returns row 0's tokens.
/// Untimed, outside every measurement window, and dropped before the timed cache
/// is touched — so the timed pass starts from the same state an unflagged run
/// would, rather than continuing one.
///
/// # Why a whole decode rather than a cheaper oracle
///
/// Because the proposer must be *the target's own output*, and nothing cheaper
/// is. That is the property #89 rests on: **greedy speculation is
/// output-identical by construction**, so a proposer that proposes what the
/// target would emit makes acceptance total and turns the mechanism's cost into
/// the only variable. A proposer drawn from anywhere else measures the
/// proposer's quality confounded with the mechanism's cost, which is the thing
/// #284 exists to separate.
///
/// It costs one untimed decode of `n` tokens per run. That is real and it is
/// stated rather than hidden: a `--speculate` run is roughly twice the wall time
/// of an unflagged one, all of the excess outside the window.
#[allow(clippy::too_many_arguments)]
fn build_oracle_sequence<P: Lfm2Shape>(
    model: &Model<P>,
    config: &Config,
    dtype: DType,
    device: &Device,
    kv_capacity: usize,
    prompt_ids: &[u32],
    batch: usize,
    n: usize,
    seed: u64,
    temperature: f64,
    top_p: f64,
) -> Result<Vec<u32>> {
    // Its own cache, and its own `Cache::new_with` — so **admission runs on this
    // one too** (§9.5l). A configuration whose predicted peak exceeds the budget
    // is refused before the oracle is paid for rather than after.
    let mut cache = State::<P>::new_with(true, dtype, config, device, kv_capacity)
        .context("allocating the oracle's KV cache")?;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: top_p,
            temperature,
        }
    };
    // A fresh processor at the same seed, so the oracle samples exactly as the
    // timed run will. Sharing one would advance its RNG state and make the timed
    // run's sampling depend on how long the oracle ran (§2.3.3 #7's shape, in the
    // harness rather than in the model).
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    let prefill_ids: Vec<u32> = prompt_ids.repeat(batch);
    let input = Tensor::new(prefill_ids.as_slice(), device)?.reshape((batch, prompt_ids.len()))?;
    let mut logits = model
        .forward(&input, 0, &mut cache)
        .context("oracle prefill")?;
    let mut kv_len = prompt_ids.len();

    let mut tokens: Vec<u32> = Vec::with_capacity(n);
    let mut rows: Vec<u32> = vec![0; batch];
    while tokens.len() < n {
        for (r, slot) in rows.iter_mut().enumerate() {
            let row = logits.narrow(0, r, 1)?.squeeze(0)?;
            *slot = logits_processor.sample(&row).context("oracle sampling")?;
        }
        tokens.push(rows[0]);
        let input = Tensor::new(rows.as_slice(), device)?.reshape((batch, 1))?;
        logits = model
            .forward(&input, kv_len, &mut cache)
            .context("oracle decode")?;
        kv_len += 1;
    }
    // The cache is dropped here, before the caller's timed one is used.
    Ok(tokens)
}

/// Everything the speculative loop touches, bundled so the signature does not
/// grow past what a reader can hold.
struct SpecRun<'a, P: Lfm2Shape> {
    k: usize,
    proposer: &'a str,
    accept_check: bool,
    n: usize,
    batch: usize,
    profiling: bool,
    oracle: &'a [u32],
    model: &'a Model<P>,
    device: &'a Device,
    logits_processor: &'a mut LogitsProcessor,
    cache: &'a mut State<P>,
    logits: &'a mut Tensor,
    kv_len: &'a mut usize,
    tokens: &'a mut Vec<u32>,
    batch_rows: &'a mut Vec<u32>,
    steps: &'a mut Vec<(f64, f64, u64)>,
    step_stamps: &'a mut Vec<(u64, usize)>,
    eos_ids: &'a [u32],
    hit_eos: &'a mut bool,
    spec: &'a mut SpecCounters,
    batch_divergence: &'a mut Option<(usize, usize, u32, u32)>,
    last_token_kernels: &'a mut Vec<(String, u64)>,
}

/// The verify loop: propose K, verify in one pass, accept a prefix, roll back
/// K−n — `DESIGN.md` §10.2a's contract, ported from `lfm2-determinism` (#89)
/// onto the harness that times it (#284, §14.5 step 1).
///
/// # The window is what is timed, and that is the port's whole point
///
/// The decode loop below times one **token**; this times one **window**, then
/// divides by the tokens the window kept. §10.2i names that denominator as the
/// thing the profiler lacked, and `SpecCounters::accepted` is it. A window that
/// accepts 1 of 8 costs what it costs and produced one token, which is exactly
/// the arithmetic #89's `elapsed_ms` could not do — that harness's own docs say
/// the field *"is not comparable between arms"*, so the most-cited speculative
/// figures in this project (K=2's 1.245×, K=8's 0.587×) rest on a quantity their
/// own harness disclaims.
fn run_speculative_decode<P: Lfm2Shape>(r: SpecRun<'_, P>) -> Result<()> {
    let SpecRun {
        k,
        proposer,
        accept_check,
        n,
        batch,
        profiling,
        oracle,
        model,
        device,
        logits_processor,
        cache,
        logits,
        kv_len,
        tokens,
        batch_rows,
        steps,
        step_stamps,
        eos_ids,
        hit_eos,
        spec,
        batch_divergence,
        last_token_kernels,
    } = r;

    // `wrong:<N>` corrupts every `N`-th proposal. Parsed here rather than at the
    // flag so the error names the flag and the value together.
    let wrong_every: Option<usize> = match proposer {
        "oracle" => None,
        s => match s.strip_prefix("wrong:") {
            Some(rest) => {
                let every: usize = rest.parse().map_err(|_| {
                    anyhow::anyhow!("--spec-proposer wrong:<N> needs an integer, got {rest:?}")
                })?;
                anyhow::ensure!(every > 0, "--spec-proposer wrong:0 would corrupt nothing");
                Some(every)
            }
            None => {
                anyhow::bail!("--spec-proposer must be `oracle` or `wrong:<N>`, got {proposer:?}")
            }
        },
    };
    let vocab_size = logits.dim(logits.rank() - 1)? as u32;

    // **A position predicts the token that FOLLOWS the one it reads**, and the
    // resulting off-by-one is the failure this loop is shaped to avoid. #89
    // records what getting it wrong looks like, and it is not a crash: comparing
    // position `j` against `window[j]` rather than against the proposal for the
    // step it predicts produced a full, stable token stream with an acceptance
    // rate of 0.26 **under a perfect oracle** — which reads as a bad proposer.
    // That is why the oracle arm's acceptance is asserted rather than printed.
    while tokens.len() < n {
        let done = tokens.len();
        if done >= oracle.len() {
            break;
        }

        // Step `done`'s token, from the logits already in hand. It costs
        // nothing — the pass that produced it has already been paid for — and
        // feeding it back is what conditions the window on it.
        for (row, slot) in batch_rows.iter_mut().enumerate() {
            let l = logits.narrow(0, row, 1)?.squeeze(0)?;
            *slot = logits_processor.sample(&l).context("sampling")?;
        }
        let emitted = batch_rows[0];
        if accept_check && batch > 1 && batch_divergence.is_none() {
            if let Some((row, &t)) = batch_rows.iter().enumerate().find(|(_, &t)| t != emitted) {
                *batch_divergence = Some((done, row, emitted, t));
            }
        }
        if eos_ids.contains(&emitted) {
            *hit_eos = true;
        }
        tokens.push(emitted);

        // **The emitted stream and the oracle must agree, and that is the
        // mechanism's own guarantee rather than a coincidence.** Under greedy
        // decoding every emitted token is the target's argmax, which is what
        // non-speculative decoding emits, which is what the oracle recorded. A
        // rejection changes *which proposals are wasted*, never *what is
        // emitted*. So a divergence here is the verifier failing — not a
        // rejected proposal — and the run stops rather than continuing against
        // proposals for positions the model never reached.
        anyhow::ensure!(
            emitted == oracle[done],
            "speculative decoding diverged from the non-speculative sequence at step \
             {done}: emitted {emitted}, oracle {}. Under greedy decoding these are \
             identical by construction (#89), so this is a defect in the verifier and \
             not a rejected proposal.",
            oracle[done]
        );

        if tokens.len() >= n || tokens.len() >= oracle.len() {
            break;
        }

        // The proposals for the steps after it, clipped to the oracle.
        let want = k.min(oracle.len() - tokens.len());
        let mut props: Vec<u32> = oracle[done + 1..done + 1 + want].to_vec();
        if let Some(every) = wrong_every {
            for (j, p) in props.iter_mut().enumerate() {
                if (done + 1 + j) % every == every - 1 {
                    *p = p.wrapping_add(1) % vocab_size;
                    spec.corrupted += 1;
                }
            }
        }
        let width = props.len();
        if width == 0 {
            break;
        }

        // `[emitted, props[..width-1]]` — `width` positions, verifying
        // `width - 1` proposals and predicting one bonus token.
        let mut ids: Vec<u32> = Vec::with_capacity(width);
        ids.push(emitted);
        ids.extend_from_slice(&props[..width - 1]);
        // Every row runs the same window: `--batch` is uniform lockstep
        // (§13.4b), so the rows share a proposal set exactly as they share a
        // prompt.
        let batched_ids: Vec<u32> = ids.repeat(batch);

        // ---- the timed window ------------------------------------------
        let tok = cache.advance(width)?;
        let step_start = std::time::Instant::now();
        let input = Tensor::new(batched_ids.as_slice(), device)?.reshape((batch, width))?;
        let all = model
            .forward_all(&input, *kv_len, cache)
            .context("speculative verify pass")?;
        // One synchronization per window, for the decode loop's reason: the
        // acceptance test reads the logits back, so the serialization is
        // inherent to the workload rather than an artifact of measuring it.
        device.synchronize()?;
        let wall = step_start.elapsed().as_secs_f64();
        let (gpu, disp) = if profiling {
            let s = profile::snapshot();
            profile::reset();
            *last_token_kernels = s.by_label.clone();
            (s.gpu_busy_union_s, s.dispatches)
        } else {
            (0.0, 0)
        };
        *kv_len += width;

        // ---- the accept test, per row ----------------------------------
        //
        // **Uniformity is a theorem here, not an assumption, and that is why the
        // check asserts rather than handles.** #284 anticipates rows accepting
        // different prefix lengths — *"a rollback affects some rows and not
        // others"* — which is true of a **ragged** batch and is exactly what
        // §13.4b puts out of scope. Under uniform lockstep with identical
        // prompts, identical rows produce identical logits (that IS
        // `--batch-check`), hence identical argmaxes, hence the same accepted
        // length. So a divergence is a **batch leak** — §2.3.3 #7's forbidden
        // dependence of a sequence on its neighbours — and is reported as a
        // defect rather than accommodated.
        //
        // It is also the only shape the state can express: `KvSlot` carries one
        // `len` for the whole batch and `Cache` one `conv_phase`, so `resolve`
        // moves both for every row together. A genuinely ragged accept would
        // need a per-row length that does not exist, which is #148's paging
        // question rather than this harness's.
        // `[B, width, vocab] -> [vocab]` for one (row, position). `narrow` twice
        // and `squeeze` twice rather than `IndexOp::i`, matching the sampling
        // above: both are views, neither copies, and it needs no trait import.
        let row_pos = |all: &Tensor, row: usize, j: usize| -> Result<Tensor> {
            Ok(all
                .narrow(0, row, 1)?
                .squeeze(0)?
                .narrow(0, j, 1)?
                .squeeze(0)?)
        };
        let mut accepted = 0usize;
        for (j, proposed) in props.iter().enumerate() {
            accepted += 1;
            let target = logits_processor
                .sample(&row_pos(&all, 0, j)?)
                .context("sampling the verify pass")?;
            if target != *proposed {
                break;
            }
        }
        if accept_check && batch > 1 && spec.accept_divergence.is_none() {
            for row in 1..batch {
                let mut row_accepted = 0usize;
                for (j, proposed) in props.iter().enumerate() {
                    row_accepted += 1;
                    let target = logits_processor
                        .sample(&row_pos(&all, row, j)?)
                        .context("sampling the verify pass")?;
                    if target != *proposed {
                        break;
                    }
                }
                if row_accepted != accepted {
                    spec.accept_divergence = Some((done, row, accepted, row_accepted));
                    break;
                }
            }
        }

        // Emit every accepted position except the last, whose logits become the
        // carried prediction for the next window's first token.
        //
        // **Two loops rather than one**, and #89 records why: a single loop that
        // both emitted position `j` and assigned `logits = row` emitted the same
        // prediction twice, which showed up as an acceptance rate that looked
        // like a mediocre proposer rather than as a bug. Separating them makes
        // the invariant structural — exactly one row is carried, every other
        // accepted row is emitted.
        let mut carry = None;
        for j in 0..accepted {
            // `[B, width, vocab] -> [B, vocab]`, position `j` for every row.
            let row = all.narrow(1, j, 1)?.squeeze(1)?.contiguous()?;
            if j + 1 == accepted {
                carry = Some(row);
                break;
            }
            if tokens.len() >= n || tokens.len() >= oracle.len() {
                break;
            }
            for (b, slot) in batch_rows.iter_mut().enumerate() {
                let l = row.narrow(0, b, 1)?.squeeze(0)?;
                *slot = logits_processor.sample(&l).context("sampling")?;
            }
            let target = batch_rows[0];
            if accept_check && batch > 1 && batch_divergence.is_none() {
                if let Some((rr, &t)) = batch_rows.iter().enumerate().find(|(_, &t)| t != target) {
                    *batch_divergence = Some((tokens.len(), rr, target, t));
                }
            }
            if eos_ids.contains(&target) {
                *hit_eos = true;
            }
            tokens.push(target);
        }
        if let Some(row) = carry {
            *logits = row;
        }

        // The pass appended `width` positions and `accepted` of them produced
        // tokens that were kept, so `width - accepted` are discarded — a length
        // decrement and a phase move, neither of which clears a byte (§10.2a).
        cache.resolve(tok, accepted)?;
        *kv_len -= width - accepted;

        // **One `steps` entry per window, and the per-token divide happens
        // later.** Recording it per accepted token instead would make the
        // sd and the median describe a quantity no window produced.
        steps.push((wall, gpu, disp));
        step_stamps.push((run_record::now_ns(), *kv_len));
        spec.windows += 1;
        spec.proposed += width;
        spec.accepted += accepted;
    }
    Ok(())
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches)?;
    execute(args, matches)
}

fn execute(args: Args, matches: ArgMatches) -> Result<()> {
    validate_run_telemetry_request(&args)?;
    // A gate states its contract: deterministic, unless LLOOM_PROFILE names one.
    candle_examples::pin_profiled_contract(
        "lfm2-decode-profile",
        candle::tuning::Contract::Deterministic,
    )?;
    let key = shape_key(&args, &matches)?;
    println!("PROFILE {key}");
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => run::<P>(args, matches))
}

fn run<P: Lfm2Shape>(args: Args, matches: ArgMatches) -> Result<()> {
    let contract =
        candle::tuning::gate_contract("lfm2-decode-profile").map_err(anyhow::Error::msg)?;
    println!("CONTRACT {}", contract.report());
    let given = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);

    let profiling = profile::enabled();

    // Before anything loads, so it is the load the run *began* under. The
    // closing reading is taken at the record, and the pair is what shows a
    // build arriving midway through.
    let load_at_start = load_average();

    // Cloned rather than moved out: `args` is read again below, both to resolve
    // the variant axes and to print them on the RESULT line.
    let model_dir = args.model_dir.clone().or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    // **The completion arm is parsed and installed BEFORE the device exists**
    // (`DESIGN.md` §11.5o, issue #559, wired here by #567). `Commands::new`
    // reads the arm once at construction through a `OnceLock`, for the reason
    // every other per-token setting is read once (§2.4: a per-call `var()` on
    // the hot path is what #69's vacuous determinism run risks) -- so setting
    // it afterwards would silently leave the incumbent installed while the
    // `RESULT` line reported otherwise. That is §7.1a-i's wrong-arm shape.
    //
    // Parsed first, so an unrecognised value fails here rather than panicking
    // inside the device constructor: the error names the flag the operator
    // typed. Under `cfg(metal)` because the type lives behind the feature --
    // §14.1 makes not breaking the non-Metal build a hard constraint -- and a
    // CPU run has no command buffer to wait for, so the flag is inert rather
    // than merely unavailable.
    #[cfg(feature = "metal")]
    let completion_wait =
        candle::metal_backend::CompletionWait::from_str_arm(&args.completion_wait)
            .map_err(anyhow::Error::msg)?;
    // SAFETY: single-threaded, before any device or thread exists. `set_var` is
    // unsafe from Rust 2024 because a concurrent `getenv` is UB; there is no
    // other thread here, and the value is read once immediately below.
    #[cfg(feature = "metal")]
    unsafe {
        std::env::set_var("CANDLE_METAL_COMPLETION_WAIT", &args.completion_wait)
    };

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // **Read back from the RESOLVED value, and refuse if it disagrees**
    // (§7.1a-i). The check is that what the device installed is what was asked
    // for, not that the string parsed: a wrong arm compares equal to every run
    // of that arm, where an absent one compares equal to nothing.
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        anyhow::ensure!(
            m.completion_wait_name() == completion_wait.name(),
            "--completion-wait {} resolved to {} inside the device: the arm reported \
             would not be the arm executed (DESIGN.md §7.1a-i)",
            completion_wait.name(),
            m.completion_wait_name()
        );
        // An arm that needs an event and did not get one runs the blocking
        // fallback -- correct, but not what was asked for, and a null measured
        // on it would be attributed to the wrong mechanism (§2.4).
        anyhow::ensure!(
            !completion_wait.needs_event() || m.completion_event_present(),
            "--completion-wait {} needs an MTLSharedEvent and the device declined to \
             create one, so every wait would take the blocking fallback",
            completion_wait.name()
        );
    }

    // f16 on Metal is what ambrogio loads. Measuring f32 here would measure a
    // path nothing runs, and would move twice the bytes per token.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => match &device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        },
    };

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_raw)?;

    // The preset first, then the axis flags over it: every flag when there is
    // no preset, only the flags given on the command line when there is one
    // (`lfm2-determinism --preset` has the same rule).
    let preset = candle_examples::lfm2_setup::preset(&args.preset)?;
    if let Some(apply) = preset {
        apply(&mut config);
        config.conv_prefill_ctx = true;
    }
    let set = |id: &str| preset.is_none() || given(id);

    // `AttnImpl` is a construction-tier axis on the config (§7.1, #97), so it
    // is selected before the model is built and both arms stay compiled.
    let conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    if set("conv_state") {
        config.conv_state = conv_state;
    }
    if set("conv_fused") {
        config.conv_fused = args.conv_fused;
    }
    // The two GEMV epilogue fusions (#509's wiring of #484's and #509's
    // kernels), construction-tier like `AttnImpl` beside them.
    if set("gemv_silu") {
        config.gemv_silu = args.gemv_silu;
    }
    if set("gemv_residual") {
        config.gemv_residual = args.gemv_residual;
    }
    let gemv_gate =
        lloom::models::lfm2::GateFusion::parse(&args.gemv_gate).map_err(anyhow::Error::msg)?;
    if set("gemv_gate") {
        config.gemv_gate = gemv_gate;
    }
    // #599's SiLU arm. Refused rather than defaulted, for `--conv-state`'s
    // reason: an unrecognized value silently pinning `exact` would render
    // `SiluImpl=exact` on a RESULT line for a run the caller asked to be a
    // variant, and that line would then pool with every real `exact` run
    // (§2.4, §7.1a-i).
    #[cfg(feature = "metal")]
    {
        let silu = candle::kernels::SiluImpl::parse(&args.silu_impl).ok_or_else(|| {
            anyhow::anyhow!(
                "--silu-impl must be one of `exact`, `recip`, `literal`, `coarse4`, \
                 `coarse1`, `hardswish`, `relu`, got `{}`",
                args.silu_impl
            )
        })?;
        // The SiLU arm rides `Config::silu_impl` now (RESTRUCTURING §1.6):
        // `Model::new` pushes the field into the kernel cache, so setting the
        // atomic directly here would be clobbered by that push.
        if set("silu_impl") {
            config.silu_impl = silu;
        }
    }
    let attn = match args.attn.as_str() {
        "generic" => lloom::models::lfm2::AttnImpl::Generic,
        "sdpa" => lloom::models::lfm2::AttnImpl::Sdpa,
        // Issue #116. Selectable and never a default: 10.4's argument for it
        // is structural rather than measured, and the kv_len at which it pays
        // is #61's to find.
        "flash" => lloom::models::lfm2::AttnImpl::FlashDecoding,
        other => anyhow::bail!("--attn must be `generic`, `sdpa` or `flash`, got `{other}`"),
    };
    if set("attn") {
        config.attn_impl = attn;
    }
    if set("flash_page_size") {
        config.flash_page_size = args.flash_page_size;
    }
    if set("flash_k") {
        config.flash_pages_per_chunk = args.flash_k;
    }
    // §9.1a's sizing policy, reaching a model path for the first time (#234).
    // Refused rather than defaulted, for the same reason as `--attn`: a
    // silently-ignored flag makes an A/B arm measure the wrong thing (§2.4).
    let sizing =
        FlashScratchSizing::parse(&args.flash_scratch_sizing).map_err(anyhow::Error::msg)?;
    if set("flash_scratch_sizing") {
        config.flash_scratch_sizing = sizing;
    }
    // Refused rather than defaulted, for the same reason as `--attn`.
    let kv_append = match args.kv_append.as_str() {
        "cat" => lloom::models::lfm2::KvAppend::Cat,
        "in-place" => lloom::models::lfm2::KvAppend::InPlace,
        "paged" => lloom::models::lfm2::KvAppend::Paged,
        other => anyhow::bail!("--kv-append must be `cat`, `in-place` or `paged`, got `{other}`"),
    };
    if set("kv_append") {
        config.kv_append = kv_append;
    }

    // Refused rather than defaulted, for `--kv-append`'s reason: an
    // unrecognized value silently pinning the default would report the f16
    // arm's digest under a quantized arm's name -- #69's vacuous run (§2.4).
    let kv_quant = KvQuant::parse(&args.kv_quant).map_err(anyhow::Error::msg)?;
    if set("kv_quant") {
        config.kv_quant = kv_quant;
    }

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::<P>::new(&config, vb).context("constructing LFM2 model")?;

    // §9.5's admission, installed only when asked for (issue #249).
    //
    // **The batch is what makes this reachable.** §9.5b's table is flat in
    // every B=1 configuration — everything under 12.7 % of the machine — so a
    // B=1 harness installing a budget would never see it refuse, and §9.5l
    // records that no harness installs one at all. `KV × B` is the only class
    // that can consume the machine (§9.5b), so `--batch` is the axis that
    // walks the prediction into the refusal.
    //
    // Placed **after** `Model::new` because that is where the weights are
    // actually allocated, and §9.5m's `reconcile_weights` compares the figure
    // passed here against what the weight-bearing pool holds — a check that
    // reads an empty pool reports nothing (#244). §7.1b records the ordering as
    // a fact of all four harnesses: `Model::new` precedes `Cache::new_with` in
    // 4 of 4, and this is the fifth.
    if let Some(fraction) = args.memory_budget {
        anyhow::ensure!(
            fraction > 0.0 && fraction <= 1.0,
            "--memory-budget must be in (0, 1], got {fraction}"
        );
        let la: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| matches!(t, lloom::models::lfm2::LayerType::FullAttention))
            .collect();
        let mut budget = lloom::models::lfm2::MemoryBudget::new(language_weight_bytes(
            &config,
            &la,
            dtype.size_in_bytes(),
        ));
        budget.batch = args.batch;
        budget.fraction = fraction;
        config.memory_budget = Some(budget);
    }

    // `new_with` rather than `new`, so #61's context curve can reach past
    // `DEFAULT_KV_CAPACITY`'s 4096 on the `InPlace` arm. Unflagged this passes
    // exactly `DEFAULT_KV_CAPACITY`, so it is the same call `new` makes.
    //
    // **This is where admission runs**, so a configuration whose predicted peak
    // exceeds the budget is refused here — before the first token and before
    // any KV is allocated — rather than discovered by dying (§9.5a: the machine
    // has been panicked twice by finding a limit that way).
    let mut cache = State::<P>::new_with(true, dtype, &config, &device, args.kv_capacity)
        .context("allocating KV cache")?;

    // ---- variant axes, validated before the clock starts -----------------
    //
    // Parsed and checked here so an unsatisfiable combination fails before the
    // model runs rather than after a five-minute measurement.
    let axes = Axes::resolve(&args, &device, &config)?;
    axes.announce();

    // ---- #171's by-class telemetry, switched on (#205) --------------------
    //
    // Refused rather than silently degraded when the feature is absent. §2.4's
    // rule after #69's vacuous determinism run: an instrument that cannot be
    // shown to have engaged has measured nothing, and a run that asks for a
    // timeline and receives an empty file is precisely that failure -- worse
    // here, because an empty series and a **flat** series are different
    // findings and only the second answers §16 P0 #7.
    //
    // `compiled()` is a `const fn` over `cfg!`, so this is the *build's*
    // capability rather than a flag read back to itself; `engaged()` at the end
    // of the run is the other half, and it is checked before the file is
    // written.
    #[cfg(feature = "metal")]
    if args.run_telemetry_jsonl.is_some() {
        candle::kernels::metal::run_telemetry::set_enabled(true);
    }

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }
    // **The positive control for the ring's EOS cross-check** (lloom #567), and
    // it exists because without it that check is **unexercisable**.
    //
    // The cross-check in the loop compares the ring's device-side
    // `broadcast_eq` against the host's `Vec::contains` at every step. On this
    // harness's prompt at `--temperature 0` the model does not reach a real EOS
    // inside any run length worth measuring — 300 tokens read `hit_eos=false` —
    // so both sides read `false` at every step and the comparison is
    // **vacuously true**. Measured: a mutation corrupting the ring's uploaded
    // id set SURVIVED the ordinary probe for exactly that reason, and was
    // killed only once an id the prompt actually emits was in the set.
    //
    // That is §11.5l's lesson in this alphabet — *a fixture that does not vary
    // the axis under test cannot see a defect pinned to that axis* — and
    // §3.1b rule 2's repair: a shape invariant needs an arm that pins the
    // level, not merely an invariant that holds.
    //
    // Env-gated rather than a flag, deliberately: it is a **test instrument**,
    // not a variant axis. It selects no kernel and appears in no `config=[…]`
    // field, and it must never be set for a measured run — with it set the run
    // stops early on a token that is not really EOS. Announced on stderr so a
    // run that sets it by accident says so rather than quietly generating a
    // short sequence (§2.4: a silent fallback produces a run that reports one
    // thing and executes another).
    if let Ok(v) = std::env::var("LLOOM_567_FAKE_EOS") {
        let extra: u32 = v
            .parse()
            .map_err(|e| anyhow::anyhow!("LLOOM_567_FAKE_EOS must be a token id: {e}"))?;
        eprintln!(
            "LLOOM_567_FAKE_EOS={extra}: treating token {extra} as EOS. This is the EOS \
             cross-check's positive control and MUST NOT be set for a measured run."
        );
        eos_ids.push(extra);
    }

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );

    // **TTFT opens here** — see the block below for the definition. The clock
    // starts before the tokenize because tokenizing is work the request causes:
    // a caller hands over text, not ids. It is small (§13.7 measures the
    // detokenize direction at 3.4-7.6 µs) and it is inside rather than argued
    // away, because a bracket that excludes a term because it is small is the
    // shape §2.5 exists to prevent.
    let ttft_start = std::time::Instant::now();

    let prompt_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    // ---- TTFT: prompt submitted -> first token available -----------------
    //
    // **The request layer** (`DESIGN.md` §2.5.1), and the one row of that table
    // that read *"no. Appears in prose repeatedly, measured never."* §2.5.4
    // defines it and this is that definition, bracketed:
    //
    //     prompt submitted  ->  prefill  ->  one decode step  ->  the first
    //                                                            sample
    //
    // # §2.5.4's decomposition names one term this loop does not have
    //
    // **There is no decode step before the first token, and the bracket is
    // what shows it.** §2.5.4 writes TTFT as `prefill + one decode step + the
    // first sample`; read against this loop, the decode loop is
    // **sample-then-forward** — `sample()` at the top of iteration *i* consumes
    // the logits iteration *i-1* produced, and for *i* = 0 it consumes **the
    // prefill's**, which the `--sample-in-buffer` bootstrap above says in so
    // many words. So the first token is the prefill's own argmax and the first
    // decode forward runs only to produce the *second*.
    //
    // Measured, that is the difference between the two readings: `ttft -
    // prefill_ms` is ~4.2-4.7 ms against a step of ~18 ms. Had the definition
    // been implemented as written, TTFT would have been reported ~18 ms too
    // large — which is §2.5's own thesis (*"two figures with different brackets
    // subtracted"*) landing on the section that states it.
    //
    // # Three things the definition pins, each of which this clock obeys
    //
    // **It does not drain.** §2.5.4 names the profiler's own prefill bracket as
    // the trap — `synchronize()` on both sides reads ~90 ms of *execution* where
    // a user's loop does not drain and sees ~30 ms of *submission* (§14.5b
    // finding 3) — and says *"the release-harness bracket is the one that
    // matches what a user waits for."* So this clock opens before the tokenize
    // and closes after the first token exists, with **no synchronize of its
    // own**: the drain that does happen inside it is the one the decode loop
    // was always going to do, at `to_scalar`.
    //
    // **It contains the sample.** *"The first token is not available until it
    // has been sampled, so TTFT contains the sample by construction"* — which
    // is the layer confusion the taxonomy exists to prevent appearing inside
    // the definition of the quantity itself. Built from `wall_ms_per_token` it
    // would understate by ~0.44 ms.
    //
    // **It is single-shot, so §14.5b's `n`-drift does not apply** — that is a
    // property of the engine-cold layer, and this is the request layer.
    //
    // # Why it starts here and not at `main`
    //
    // A user's request arrives with the model already loaded; what they wait
    // for is their prompt being answered, not the process starting. Model load
    // is §2.4a's quantity and is measured there (page-in 7.01 s of a 6.66 s–16.5 s
    // cold load, attacked by #298's `madvise`). Putting it inside TTFT would
    // fold two quantities whose optimisations are unrelated — and would make
    // the figure a property of the page cache, which §2.4's own
    // *"cold-start quantity needs its cold state PROVED"* is about.
    //
    // **The consequence is that this figure is WARM-PROCESS TTFT and must be
    // labelled one.** §2.5.4's *"what would make it wrong"* names *"quoting it
    // from a warm process"* as a failure — for a *cold-start* TTFT, which is a
    // different row. Both are real; the artifact states which this is, and the
    // cold one is `prefill + this` plus §2.4a's load, not a re-bracket.
    //
    // # The one drain inside it, and why it is not the trap §2.5.4 names
    //
    // The prefill `synchronize()` below IS inside this window, and it stays
    // there rather than being made conditional. §2.5.4's trap is about
    // reporting `prefill_ms` — a *component* — from a draining bracket and then
    // comparing it against a harness that does not drain, which is §14.5b
    // finding 3's ~90 ms against ~30 ms. **TTFT is the total to the first
    // token, and that total is drain-invariant to first order**: the first
    // token cannot exist until the GPU has executed the prefill, so draining
    // early moves work *within* the window rather than adding work to it. What
    // it can cost is overlap — the CPU could have been encoding decode step 0
    // while the prefill ran — and at a 34-token prompt that is bounded by the
    // ~0.9 ms non-GPU term of one step against an ~87 ms prefill.
    //
    // **It is not made a flag**, deliberately. Removing the call would change
    // `prefill_ms`, which is a field on every `RESULT` line this project has
    // recorded (§7.1a's lesson about a quantity moving under a name), and this
    // change is meant to ADD a layer rather than perturb one. The bound above
    // is stated in the artifact instead, and the release-harness comparison
    // §2.5.4 asks for is `lfm2-release`'s to make — which is a separate finding
    // this PR records as absent from the monorepo.

    // ---- prefill, timed into its own bucket ------------------------------

    // Synchronize before starting the clock so no earlier work (weight upload,
    // cache allocation) lands inside the window.
    device.synchronize()?;
    if profiling {
        profile::reset();
    }

    let prefill_start = std::time::Instant::now();
    // `[B, prompt_len]` — the same prompt in every row.
    //
    // **Built by repeating rather than by broadcasting.** `Tensor::broadcast_as`
    // would give a view whose stride along dim 0 is zero, and `Embedding` is a
    // gather that would then read one row and produce a non-contiguous result
    // the rest of the stack has to materialise anyway. Repeating the ids costs
    // `B * prompt_len` u32 once, off the per-token path.
    //
    // At `B = 1` this is `prompt_ids` unchanged, so the default path allocates
    // and computes exactly what it did before.
    let batch = args.batch;
    let prefill_ids: Vec<u32> = prompt_ids.repeat(batch);
    let input = Tensor::new(prefill_ids.as_slice(), &device)?.reshape((batch, prompt_ids.len()))?;
    // `[B, vocab]` — one row of logits per sequence. `forward` narrows to the
    // last position and returns `[b_sz, vocab]`, so the `squeeze(0)` the B=1
    // path used is exactly the `B = 1` case of keeping the batch dimension.
    let mut logits = model
        .forward(&input, 0, &mut cache)
        .context("prefill forward pass")?;
    // Force the prefill to complete before the clock stops; candle is
    // asynchronous, so without this the time recorded is submission, not
    // execution.
    device.synchronize()?;
    let prefill_wall = prefill_start.elapsed();

    let prefill_profile = if profiling {
        let s = profile::snapshot();
        profile::reset();
        Some(s)
    } else {
        None
    };

    let mut kv_len = prompt_ids.len();

    // ---- decode, one timed window per token ------------------------------

    // Per token: (wall seconds, gpu busy seconds, dispatches).
    let mut steps: Vec<(f64, f64, u64)> = Vec::with_capacity(args.n);
    // Per-step cross-encoder fence-map reads, hits, and waits (lloom #732).
    let mut fence_steps: Vec<(u64, u64, u64)> = Vec::with_capacity(args.n);
    // Per token, for the run record: `CLOCK_UPTIME_RAW` at the end of the token
    // and the `kv_len` it ran at. Pre-reserved, so the push inside the loop
    // cannot reallocate mid-window; the stamp itself is one
    // `clock_gettime_nsec_np` against a 24 MHz timebase (`DESIGN.md` §3.4b).
    let mut step_stamps: Vec<(u64, usize)> = Vec::with_capacity(args.n);
    // Row 0's stream. At `B = 1` it is *the* stream, and at `B > 1` it is the
    // one every other row is asserted equal to, so the reported text and the
    // EOS test mean the same thing at every `B`.
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n);
    // Every row's stream for this step, reused across steps so the per-token
    // path allocates nothing. `batch_rows[r]` is row `r`'s token.
    let mut batch_rows: Vec<u32> = vec![0; batch];
    // The first step at which two rows disagreed, if any. Recorded rather than
    // returned immediately so the run still reports where it got to — a
    // divergence is a finding and its `kv_len` is part of it.
    let mut batch_divergence: Option<(usize, usize, u32, u32)> = None;
    let mut hit_eos = false;
    // Time to first token — the request layer (`DESIGN.md` §2.5.1, §2.5.4).
    // `None` until the first token is pushed; see the close site for why it is
    // there and not at the sample.
    //
    // **`None` is also the honest value on the speculative arm**, which has its
    // own loop and does not set this. A window emits between 1 and K tokens, so
    // "the first token" is the first of a batch the user could not have had
    // sooner — a different quantity, and #284's is the harness that would own
    // it. Reporting a number there would be reporting this one under that
    // arm's name (§2.4's *"ask what a field's name promises"*).
    let mut ttft: Option<std::time::Duration> = None;
    let mut last_token_kernels: Vec<(String, u64)> = Vec::new();
    // The speculative arm's counters. `spec.windows == 0` after a `--speculate`
    // run is the vacuity check §2.4 requires, asserted rather than printed.
    let mut spec = SpecCounters::default();

    // ---- ring-arm state, allocated before the loop (lloom #567) -----------
    //
    // `ring` is the persistent input slot plus the device-resident EOS set.
    // Allocated on first use rather than here: `TokenRing::new` wants a first
    // input, and the slot's contents before the first `publish` are never read.
    //
    // **The counters are the reason a null on this axis is readable at all.**
    // §11.5c's first sweep measured the blit against itself and read
    // `unresolved` on every dimension -- it was caught by `to_cpu_fast=0` on
    // both arms and by nothing else. #500 is the same shape one level out:
    // every gate was green because the mechanism had no callers, and a run
    // reporting `ring_publishes=0` under `--token-ring` is that state made
    // visible instead of being discovered by a census.
    let mut ring: Option<TokenRing> = None;
    let mut ring_publishes: u64 = 0;
    // EOS tokens published. Counted rather than inferred from `hit_eos`: this
    // harness deliberately generates PAST EOS (the point is a steady-state
    // measurement), so the bool says only *whether* one was seen and the
    // reconciliation below needs *how many*.
    let mut ring_eos_publishes: u64 = 0;
    let mut poll_iters: u64 = 0;
    // Steps on which the non-blocking poll gave up and took the blocking read
    // (#535). **A nonzero value is not a failure** -- the fallback computes the
    // same answer -- but it says the peek was declining, which is what makes
    // the arm's engagement readable rather than assumed (§2.4).
    let mut poll_fallbacks: u64 = 0;
    // `flush_and_prune`'s returned surviving count, maxed over the run.
    //
    // #493 repaired the unbounded `Commands::in_flight` and returns this count
    // **so a caller can assert rather than assume** (§2.4). §11.5o reads it
    // flat at 14-17 across a 7.8x span of `n` against 63 at `n` = 64 with the
    // prune removed, which is the bound this arm must reproduce.
    //
    // **Read on every arm since #567**, not only the ring's: with the per-token
    // drain deleted (§11.5r) the queue bound is a property of this loop rather
    // than of one flag, and it reads **14** here against the **1** every
    // pre-#567 run recorded.
    let mut max_in_flight: usize = 0;

    // **Whether every command buffer's GPU interval was actually collected**
    // (lloom #567, `DESIGN.md` §11.5r).
    //
    // The per-token drain used to guarantee this by construction: nothing could
    // still be in flight when the snapshot ran. Without it that guarantee is an
    // **assumption**, and §2.4's rule is that an instrument which cannot be
    // shown to have engaged has not measured anything -- so it is counted
    // rather than assumed, and asserted after the loop.
    //
    // `untimed` is the number of committed buffers whose completion handler had
    // not yet run when their window was snapshotted. **A nonzero value means
    // `gpu_ms_per_token` is under-counting**, which is exactly the failure the
    // drain's removal risks and the one no digest and no other counter can see.
    let mut timed_cbs: u64 = 0;
    let mut untimed_cbs: u64 = 0;

    // The oracle sequence the proposer draws from — the tokens this
    // configuration emits **without** speculation.
    //
    // # Why it is generated here rather than read from a file
    //
    // `lfm2-determinism` takes its oracle from `--force-tokens`, and that is not
    // a mechanism this harness owes. That flag exists there to make two arms'
    // logit dumps **position-comparable** (§2.3.9a): it pins the sequence so
    // `max|Δlogit|` at position `i` is a property of the arithmetic rather than
    // of two conversations drifting apart. The proposer is a *reuse* of it.
    //
    // Here there is no second arm and no dump to subtract. What speculation
    // needs is only *a sequence the target would have emitted*, and that is
    // derivable in-process from the very thing this harness already does. So the
    // **proposer** is ported and the **file** is not — which also removes the
    // two-run record-then-replay protocol a file would have imposed, and with it
    // the chance of measuring against an oracle taken under a different
    // configuration.
    //
    // # Why it gets its own cache
    //
    // The pre-pass mutates KV and conv state, so the timed pass cannot inherit
    // it — a continuation is not a repeat (§2.4's rule for prefill, in the state
    // rather than in the page cache). A second `Cache` at B=1 is 12.7 % of the
    // machine by §9.5b's own table, and it is dropped before the timed cache is
    // built. **Admission runs on it too**, so a configuration that would be
    // refused is refused here rather than after the oracle has been paid for.
    //
    // Untimed and outside every window: nothing below `steps` sees it.
    let oracle: Vec<u32> = if args.speculate > 0 {
        build_oracle_sequence(
            &model,
            &config,
            dtype,
            &device,
            args.kv_capacity,
            &prompt_ids,
            batch,
            args.n,
            args.seed,
            args.temperature,
            args.top_p,
        )
        .context("recording the oracle sequence the proposer draws from")?
    } else {
        Vec::new()
    };

    // Per token: (sample seconds, EOS-test seconds). Issue #172's bucket,
    // ported from `bench/issue-172-sample-cost` per §11.5a.
    //
    // The `wall` window below opens AFTER the sample on the default path, so
    // §11.2's non-GPU figure -- wall minus GPU busy -- structurally cannot
    // contain it. Timed here so a step is `wall + sample + eos` and the term is
    // an observed interval rather than a share inferred from a busy series.
    //
    // A timer pair rather than a counter: §6.4a prices `Instant::now()` at
    // 43 ns, negligible against a readback that blocks on
    // `flush_and_wait_current()`, and it is the same instrument the wall window
    // uses -- so the two are commensurable by construction.
    let mut sample_steps: Vec<(f64, f64)> = Vec::with_capacity(args.n);
    // Per token under `--sample-split`: (argmax-and-sync seconds, readback seconds).
    let mut split_steps: Vec<(f64, f64)> = Vec::with_capacity(args.n);

    // The front end's seam (lloom #412). Three terms per token, kept apart
    // because they are three mechanisms — see `seam.rs`.
    let mut seam_steps: Vec<seam::SeamStep> = Vec::with_capacity(args.n);
    // Built before the loop so a runtime that cannot be created fails now
    // rather than mid-generation, and so no allocation lands in a timed window.
    let mut seam_tx = if args.seam {
        Some(seam::Seam::new(args.seam_capacity)?)
    } else {
        None
    };
    // The not-yet-emitted run, exactly as `generate_raw` holds it: tokens are
    // held back until they decode to complete text, because a multi-byte
    // character can span several of them. Reproducing that is what makes the
    // detokenize term the right size — decoding the whole sequence every step
    // instead would be quadratic and would price a bug ambrogio does not have.
    let mut seam_pending: usize = 0;
    let mut seam_text_bytes: usize = 0;

    // Context switches across the whole decode window (lloom #412).
    //
    // Read in-process because the split is only available here: measured on
    // this machine `proc_pidinfo(PROC_PIDTASKINFO)`'s `pti_csw` — the one an
    // out-of-process sampler can read — is exactly `ru_nvcsw + ru_nivcsw`, the
    // sum with no split, and it is the **involuntary** half that discriminates
    // preemption from a thread parking at a syscall. `seam.rs` has the probe.
    //
    // Two readings around the steady-state window rather than one per token: a
    // per-token pair would put a syscall inside the quantity it measures, and
    // the counter is monotone so a difference over the window divides exactly.
    let mut csw_at_warmup: Option<(u64, u64)> = None;
    // `warmup`'s two inputs, available here where `warmup` itself is not — see
    // the branch inside the loop for why the third (`.min(steps.len())`) cannot
    // bite, and the assertion after the loop that checks it did not.
    let seam_warmup_floor = args
        .warmup
        .max(if axes.arena { axes.record_steps } else { 0 });

    #[cfg(feature = "metal")]
    let metal_device = match &device {
        Device::Metal(d) => Some(d.clone()),
        _ => None,
    };

    // Buffered, and flushed after the loop rather than per sample: at
    // `--progress-every 1` on a long run this is thousands of writes, and an
    // unbuffered one inside the loop would be an instrument sitting in the
    // window it measures (§2.4, §6.4a).
    //
    // Opened BEFORE the loop so a path that cannot be created fails now rather
    // than after a run that may take an hour.
    // Bytes last written to the by-class timeline, so the final report can say
    // whether it reached disk without re-reading it.
    let mut rt_written: usize = 0;

    // **The `--sample-in-buffer` bootstrap** (issue #319).
    //
    // That arm reads each step's token at the BOTTOM of the loop, out of the
    // argmax that rode the forward pass's command buffer. The first iteration
    // has no previous bottom, so its token comes from the prefill logits here.
    //
    // This one sample is a second submit -- the prefill's `synchronize()` has
    // already drained the queue -- and that is correct rather than a leak: the
    // default arm pays exactly one such submit per token including this one, so
    // paying it once outside the loop is what makes the two arms generate the
    // same stream from the same prefill. It is off the per-token path and
    // outside every timed window.
    if args.sample_in_buffer {
        for (r, slot) in batch_rows.iter_mut().enumerate() {
            let row = logits.narrow(0, r, 1)?.squeeze(0)?;
            *slot = logits_processor.sample(&row).context("prefill sampling")?;
        }
    }

    let mut mem_jsonl = match args.mem_jsonl.as_ref() {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p)
                .with_context(|| format!("creating memory series at {}", p.display()))?,
        )),
        None => None,
    };

    // ---- the speculative arm (#284, porting #89's verify loop) ------------
    //
    // Written as a separate loop rather than a branch inside the one below, for
    // the reason `lfm2-determinism` gives at the same seam: the shapes genuinely
    // differ. An ordinary step consumes one token and emits one; a verify pass
    // consumes K and emits between 1 and K. Folding them would put a
    // `seq_len`-dependent branch on the path every recorded figure belongs to,
    // to serve a caller that path does not have — §11.3l finding 4's shape.
    if args.speculate > 0 {
        run_speculative_decode(SpecRun {
            k: args.speculate,
            proposer: &args.spec_proposer,
            accept_check: args.spec_accept_check,
            n: args.n,
            batch,
            profiling,
            oracle: &oracle,
            model: &model,
            device: &device,
            logits_processor: &mut logits_processor,
            cache: &mut cache,
            logits: &mut logits,
            kv_len: &mut kv_len,
            tokens: &mut tokens,
            batch_rows: &mut batch_rows,
            steps: &mut steps,
            step_stamps: &mut step_stamps,
            eos_ids: &eos_ids,
            hit_eos: &mut hit_eos,
            spec: &mut spec,
            batch_divergence: &mut batch_divergence,
            last_token_kernels: &mut last_token_kernels,
        })?;
    }

    while args.speculate == 0 && tokens.len() < args.n {
        // **Sampled per row**, because `LogitsProcessor::sample` is rank-0 by
        // construction: `sample_argmax` is `argmax(D::Minus1)?.to_scalar()`, and
        // `to_scalar` requires rank 0, so a `[B, vocab]` tensor is not something
        // it can take. That is the one genuine B=1 assumption on this path
        // (issue #249), and it is the harness's rather than the model's.
        //
        // **`B` GPU argmaxes rather than one batched one**, and it is stated
        // rather than elided: `fast_argmax_f32` is 1 dispatch of ~517 (§11.2a),
        // so this makes the dispatch count `517 + (B - 1)` rather than flat.
        // The alternative — one `argmax` over `[B, vocab]` and one `to_vec1` —
        // would be flat, and it would also change what the B=1 arm dispatches,
        // which is the arm every recorded figure belongs to. Keeping B=1
        // byte-identical is worth more than a flat sampling term, and §2.4's
        // engagement proof is computed over the FORWARD PASS for exactly this
        // reason — see the `dispatches_per_token_forward` note below.
        //
        // Each row gets its own narrowed view rather than a copy.
        //
        // On the `--sample-in-buffer` arm the tokens for this step were already
        // sampled at the bottom of the previous iteration, while the forward
        // pass's command buffer was still open (issue #319) -- or, for the first
        // iteration, out of the prefill logits just above the loop. `batch_rows`
        // holds them, so there is nothing to do here and no second submit.
        let sample_start = std::time::Instant::now();
        // ---- the ring arm's sample (lloom #567, `DESIGN.md` §11.5g) --------
        //
        // **In the `sample` bucket, not in `wall`, and that is what makes the
        // arms comparable.** `wall` is the forward pass's window on every
        // recorded figure this project holds (§11.2a's disjointness), and the
        // baseline's readback is timed here; the ring's publish-and-poll
        // replaces exactly that readback, so it belongs in exactly that bucket.
        // Putting it in `wall` would make this arm's `wall` incomparable with
        // every committed row while measuring the same work.
        //
        // The token is sampled into a **device** tensor and never comes back:
        // `publish` writes it into the persistent input slot in place and
        // computes the EOS flag against the uploaded id set, both GPU-side.
        // §11.5g's write leg is *deleted rather than relocated* -- there is
        // nothing to upload because nothing came down.
        //
        // `ring_token` carries the device tensor out to the forward pass below,
        // which reads `r.input()` instead of building a `[1, 1]` from a host
        // `u32`. `None` on every other arm, so the default path is untouched.
        let mut ring_is_eos = false;
        if args.token_ring {
            let sampled = logits_processor
                .sample_argmax_on_device(&logits.narrow(0, 0, 1)?.squeeze(0)?)
                .context("device-resident argmax")?;
            let r = match ring.as_ref() {
                Some(r) => r,
                None => {
                    ring = Some(TokenRing::new(0, &eos_ids, &device)?);
                    ring.as_ref().unwrap()
                }
            };
            // On the non-blocking arm the publish is STAMPED with this step's
            // ordinal, so the host can poll for *this* step rather than for a
            // flag it cannot date (#535). Ordinals start at 1 because 0 is the
            // unpublished sentinel.
            //
            // Without the stamp `eos == 0` is two states a non-blocking reader
            // cannot separate -- *published, not EOS* and *not yet published*
            // -- and the blocking read was silently discharging that ambiguity
            // by waiting. The blocking arm is unchanged and still calls the
            // unstamped `publish`.
            if args.nonblocking_eos {
                ring_publishes += 1;
                r.publish_step(&sampled, Some(ring_publishes as u32))
                    .context("publishing the token")?;
            } else {
                r.publish(&sampled).context("publishing the token")?;
                ring_publishes += 1;
            }

            // Commit without waiting, and prune. **This is the call that makes
            // the arm non-blocking, and it is also the only thing pruning
            // `Commands::in_flight` once the per-token readback is gone**
            // (§11.5h): the `Vec`'s only two shrink sites are inside the two
            // blocking calls this arm exists to stop calling. Buffer
            // *reclamation* is safe without a wait -- the completion handler
            // marks the epoch asynchronously -- and the `Vec` is not, which is
            // the asymmetry #493 found.
            #[cfg(feature = "metal")]
            if let Device::Metal(m) = &device {
                let in_flight = m.flush_and_prune()?;
                max_in_flight = max_in_flight.max(in_flight);
            }

            ring_is_eos = if args.nonblocking_eos {
                let generation = ring_publishes as u32;
                // **Bounded, and the bound falls back rather than failing.**
                // `poll_eos` returns `None` both for *not published yet* and
                // for *this buffer is not CPU-mapped*, and the second is an
                // allocator property (§15.2 #12a) that #454 measures as
                // unstable run to run -- so an unbounded loop would HANG on a
                // run where the pool happened to hand back a large bucket, and
                // it would hang silently. The blocking read is the fallback,
                // which is the same fail-closed shape §11.5c's `contents()`
                // test uses: the slow path is always correct, so the fast one
                // is free to decline.
                const POLL_BUDGET: u64 = 50_000_000;
                let mut spins: u64 = 0;
                // The watched word, resolved once (#578). `Wfe` arms the
                // exclusive monitor on it; every other arm ignores it. `None`
                // when the allocation is unmapped, and then the address-free
                // wait is used and only the monitor is lost.
                //
                // SAFETY: the pointer is valid while `r` is alive, and `r`
                // outlives this loop.
                let watched = unsafe { r.step_addr() };
                loop {
                    poll_iters += 1;
                    match r.poll_eos(generation) {
                        Some(eos) => break eos,
                        None => {
                            spins += 1;
                            if spins > POLL_BUDGET {
                                // Counted, not silent: a fallback nobody counts
                                // is a null that reads as a result (§2.4).
                                poll_fallbacks += 1;
                                break r.eos_flag().to_vec1::<u8>()?[0] != 0;
                            }
                            match watched {
                                // Wait for the counter to CHANGE from what the
                                // poll just saw. `expect` is the observed value
                                // rather than `generation - 1`: the two are the
                                // same only when no step was missed, and the
                                // point of the compare is to skip the wait when
                                // the word has already moved -- which needs the
                                // value actually read, not one derived from the
                                // ordinal. A `WFE` can also retire on its own
                                // (§3.7h), so the loop re-reads and re-tests
                                // after every return either way.
                                Some(p) => {
                                    let seen = unsafe { p.read_volatile() };
                                    unsafe { axes.poll.wait_on(p, seen) }
                                }
                                None => axes.poll.wait(),
                            }
                        }
                    }
                }
            } else {
                axes.poll.wait();
                poll_iters += 1;
                r.eos_flag().to_vec1::<u8>()?[0] != 0
            };
            if ring_is_eos {
                ring_eos_publishes += 1;
            }
            // **The token id is read back here, once, and it is OUTSIDE the
            // sample bucket's purpose but inside its window -- so it is stated
            // rather than hidden.** `lfm2-release` avoids this entirely by
            // accumulating tokens device-side and reading them after the clock;
            // this harness cannot, because `tokens` drives the EOS test, the
            // seam's detokenize, `--dump-tokens` and the per-token progress
            // line, all of which are this harness's reason to exist.
            //
            // So **the ring arm as measured here still makes one 4-byte round
            // trip per token**, and what it removes is the *write* leg plus the
            // blocking flag read -- not the read leg. A reader must not take
            // this arm's number for `lfm2-release`'s. It is recorded on the
            // `RESULT` line as `ring_reads_token=true` so the difference
            // travels with the figure rather than living only here.
            // Rank 0 already: the input is `[vocab]` and `argmax(D::Minus1)`
            // reduces it away, which is the same shape `sample_argmax` calls
            // `to_scalar` on.
            batch_rows[0] = sampled
                .to_scalar::<u32>()
                .context("reading the published token")?;
        }
        if !args.token_ring && !args.sample_in_buffer {
            for (r, slot) in batch_rows.iter_mut().enumerate() {
                // `[B, vocab] -> [vocab]`. `narrow` + `squeeze` rather than
                // `IndexOp::i`, which would need a trait import for one call; both
                // are views and neither copies. At `B = 1` this is `squeeze(0)`,
                // the exact shape the B=1 path always passed.
                let row = logits.narrow(0, r, 1)?.squeeze(0)?;
                *slot = if args.sample_split {
                    // `argmax` here enqueues ONE dispatch over the vocab and
                    // `to_scalar` then blits 4 bytes back and blocks on
                    // `flush_and_wait_current()`. Synchronizing between them
                    // attributes the two separately -- which is what decides
                    // whether the cost is the submit or the transfer (§11.2a).
                    //
                    // **This `synchronize` stays, and #567 changed what precedes
                    // it** (`DESIGN.md` §11.5r). This comment used to open *"the
                    // forward pass has already been synchronized by the previous
                    // iteration"* — true until the per-token drain was deleted,
                    // and **false now**: the previous iteration commits without
                    // waiting. The split this arm exists to take still comes
                    // from the `synchronize` *below* rather than from any
                    // earlier one, so the attribution mechanism is intact — but
                    // the argmax term now also absorbs whatever of the previous
                    // forward pass had not finished, so **this arm's split is no
                    // longer comparable with the figures §11.2a took under the
                    // drain.** Stated rather than repaired: the repair is a
                    // separate measurement and this arm is off the default path.
                    let split_start = std::time::Instant::now();
                    let idx = row.argmax(candle::D::Minus1).context("argmax")?;
                    device.synchronize()?;
                    let argmax_s = split_start.elapsed().as_secs_f64();
                    let read_start = std::time::Instant::now();
                    let tok = idx.to_scalar::<u32>().context("argmax readback")?;
                    if r == 0 {
                        split_steps.push((argmax_s, read_start.elapsed().as_secs_f64()));
                    }
                    tok
                } else {
                    logits_processor.sample(&row).context("sampling")?
                };
            }
        }
        let sample_s = sample_start.elapsed().as_secs_f64();
        let next = batch_rows[0];

        // **The correctness gate** (issue #249): identical prompts must give
        // identical streams, so a row differing from row 0 means the batch
        // dimension is leaking. Checked here — before the token is used — so
        // the failure names the step it happened at rather than a digest at the
        // end.
        if args.batch_check && batch > 1 && batch_divergence.is_none() {
            if let Some((r, &t)) = batch_rows.iter().enumerate().find(|(_, &t)| t != next) {
                batch_divergence = Some((tokens.len(), r, next, t));
            }
        }

        let eos_start = std::time::Instant::now();
        let is_eos = eos_ids.contains(&next);
        let eos_s = eos_start.elapsed().as_secs_f64();
        // **The GPU-side EOS decision is checked against the host one rather
        // than replacing it** (lloom #567). This harness generates PAST EOS by
        // design -- the point is a steady-state measurement -- so the flag is
        // not a control-flow input here and substituting it would test nothing.
        // Compared, it is an absolute reference for the ring's device-side
        // `broadcast_eq` against the `Vec::contains` every other arm runs, at
        // **every step rather than once**: §3.1b rule 1 prefers an absolute
        // reference to a shape, and this is the only place in the tree where
        // the two EOS mechanisms are computed side by side on the same token.
        //
        // A disagreement is a hard failure. §11.5g's whole design rests on the
        // uploaded id set matching the host's, and a silent divergence would
        // present as a run whose `hit_eos` is right and whose ring flag is
        // wrong -- invisible to every digest, because this harness does not
        // stop on EOS.
        if args.token_ring {
            anyhow::ensure!(
                ring_is_eos == is_eos,
                "step {}: the ring's device-side EOS flag reads {} and the host's \
                 `eos_ids.contains` reads {} for token {} -- the uploaded id set and \
                 the host's disagree (DESIGN.md §11.5g)",
                tokens.len(),
                ring_is_eos,
                is_eos,
                next
            );
        }
        sample_steps.push((sample_s, eos_s));
        if is_eos {
            // The point is a steady-state decode measurement, so generation
            // continues past the model's natural stopping point. Recorded
            // because past EOS the text degenerates, and a reader should know
            // the tail was not on-distribution.
            hit_eos = true;
        }
        tokens.push(next);

        // **TTFT closes here** (`DESIGN.md` §2.5.4, lloom #388 item 2).
        //
        // *After* `tokens.push`, because that is the instant the first token is
        // available to the caller — the definition is *"prompt submitted ->
        // first token available"*, and a token that has been sampled but not
        // yet handed over is not yet available. In this harness "handed over"
        // is the push onto `tokens`, which is what the seam, the digest and the
        // EOS test all read.
        //
        // `is_none()` so it records the FIRST step only. A single-shot quantity
        // (§2.5.4: *"the drift with `n` does not apply"*), so there is no mean
        // and no warmup — one clock, one reading, and a run reports it whatever
        // `--n` is.
        if ttft.is_none() {
            ttft = Some(ttft_start.elapsed());
        }

        // **The front end's seam** (lloom #412), in `generate_raw`'s own
        // position: after the token is pushed and before the next forward pass
        // is encoded. That placement is the measurement — the seam sits on the
        // decode thread between two tokens, which is why a consumer that blocks
        // can stall generation at all.
        //
        // Outside the `wall` window deliberately, and for the same reason the
        // sample bucket is (§11.2a's disjointness): `wall` is the forward pass's
        // window on every recorded figure this project holds, and moving a term
        // into it would make this arm's `wall` incomparable to all of them. A
        // step is `wall + sample + eos + seam`.
        //
        // The detokenize term is timed on **both** arms, because ambrogio's loop
        // runs it whether or not anyone is streaming; only the allocate and send
        // terms are the streaming arm's. Charging the first to the seam would be
        // §15.2 #11a's shape — a per-axis count read as a per-axis budget.
        {
            let mut step = seam::SeamStep::default();
            if args.detokenize {
                let (chunk, dt) = seam::time_detokenize(|| {
                    tokenizer
                        .decode(&tokens[seam_pending..], false)
                        .map_err(|e| anyhow::anyhow!("decoding tokens: {e}"))
                });
                let chunk = chunk?;
                step.detokenize_s = dt;
                // A trailing replacement char means the run ends mid-character,
                // so the tokens are held for the next step — `generate_raw`'s
                // rule, reproduced because it is what sets how often the seam
                // fires. A seam that emitted every token would be a different
                // and larger quantity.
                if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                    seam_text_bytes += chunk.len();
                    if let Some(s) = seam_tx.as_ref() {
                        let e = s.emit(&chunk);
                        step.alloc_s = e.alloc_s;
                        step.send_s = e.send_s;
                    }
                    seam_pending = tokens.len();
                }
            } else if let Some(s) = seam_tx.as_ref() {
                // `--no-detokenize` prices the channel alone, on a fragment of
                // fixed size. Not the shipped shape and labelled so: it exists
                // to separate the two mechanisms, since a null on the total
                // cannot say which half was null.
                let e = s.emit("x");
                step.alloc_s = e.alloc_s;
                step.send_s = e.send_s;
                seam_text_bytes += 1;
            }
            seam_steps.push(step);
        }

        // The steady-state context-switch window opens exactly where every
        // other steady-state mean does, so `invol_csw / token` divides by the
        // same denominator (lloom #412).
        //
        // The threshold is recomputed from the same two inputs `warmup` is
        // derived from below rather than read from it, because that binding is
        // clamped by `steps.len()` and does not exist until the loop has ended.
        // The clamp cannot bite here: this only fires while steps are still
        // being produced, so `warmup`'s `.min(steps.len())` is the identity at
        // every point the branch is reachable. Asserted after the loop rather
        // than argued — a mismatch would mean the csw window and every other
        // steady-state mean divide by different denominators.
        if csw_at_warmup.is_none() && tokens.len() > seam_warmup_floor {
            csw_at_warmup = Some(seam::csw());
        }

        // Arena recording spans the first `record_steps` decode steps.
        //
        // `next_arena_recording_step` between them is load-bearing rather than
        // bookkeeping: a plan needs **two** steps, because comparing them is what
        // separates an activation from session state -- an allocation whose size
        // moved between them is sized by `kv_len` and must not enter the arena
        // (§9.1, #68 finding 4). Recording one step makes the planner refuse and
        // return no allocations, which presents as an arena arm that silently
        // ran on the pool.
        #[cfg(feature = "metal")]
        if axes.arena {
            if let Some(dev) = metal_device.as_ref() {
                let step = tokens.len() - 1;
                if step == 0 {
                    dev.begin_arena_recording();
                } else if step < axes.record_steps {
                    dev.next_arena_recording_step();
                }
                dev.begin_decode_step();
            }
        }

        let step_start = std::time::Instant::now();
        // `[B, 1]` — one token per row. At `B = 1` this is `[1, 1]`, which is
        // what `Tensor::new(&[next])?.unsqueeze(0)?` produced.
        // **On the ring arm the input is the ring's own slot, not a fresh
        // upload** (lloom #567, `DESIGN.md` §11.5g). That is the write leg
        // being *deleted rather than relocated*: `publish` wrote this step's
        // token into the persistent `[1, 1]` buffer in place, so there is
        // nothing to upload and `Tensor::new` is not called at all.
        //
        // Held as a reference to the ring's tensor on that arm and as an owned
        // one otherwise, so the default path's construction is byte-identical.
        let owned_input = if args.token_ring {
            None
        } else {
            Some(Tensor::new(batch_rows.as_slice(), &device)?.reshape((batch, 1))?)
        };
        // The ring's input is a `[1, 1]` view of the current input tile
        // (`TokenRing::input_for`), not one stable buffer: the input cycles
        // `depth + 1` tiles so a stage never overwrites a tile an in-flight
        // forward reads. The view is held here for the forward.
        let ring_input = match (&owned_input, ring.as_ref()) {
            (None, Some(r)) => Some(r.input_for(1)?),
            _ => None,
        };
        let input = match (&owned_input, ring_input.as_ref()) {
            (Some(t), _) => t,
            // `ring` is `Some` here whenever `owned_input` is `None`: the ring
            // is allocated in the sample block above, which runs before this on
            // every iteration of the ring arm.
            (None, Some(t)) => t,
            (None, None) => anyhow::bail!(
                "the ring arm reached the forward pass with no ring allocated, which is \
                 unreachable: the sample block allocates it on first use"
            ),
        };
        // **The ring's slot is identified by POINTER, not by value**, and this
        // assertion is the instrument that makes the arm's ROUTING testable.
        //
        // A mutation that publishes correctly and then feeds the forward pass a
        // host-built `Tensor::new(batch_rows)` is **invisible to every
        // value-based check**: the two tensors carry the same id by
        // construction, so the token stream, both digests, `lfm2-smoke`'s text
        // and `ring_publishes` are all unchanged. Measured on this tree — that
        // mutant survived the composed probe, and **the dispatch census does
        // not kill it either**: 677.0 dispatches/token on both arms, because
        // the publish still happens and is merely ignored. §15.1a's
        // INEXPRESSIVE class, and the repair is a different instrument.
        //
        // This is #500's and #536's shape exactly — the mechanism intact and
        // the *routing* mutated dead — which §11.5o records as the one mutation
        // no test of the mechanism's own module can see, and which no gate in
        // this tree could otherwise catch.
        //
        // `ptr::eq` is sound here and is **not** #560's defect: that was an
        // address comparison on a `&'static str` reached through a `const`,
        // where every mention is a fresh rvalue the compiler may materialise
        // more than once. These are two references to one live heap
        // allocation, and identity is exactly the question being asked.
        //
        // One pointer compare per token, off the GPU path.
        // The pointer compare this arm carried (`ptr::eq(input, ring.input())`)
        // has nothing to compare any more: the input is a per-step view of a
        // cycling tile, and a `Tensor`'s storage identity is not public. What is
        // asserted instead is the routing itself: on the ring arm no host tensor
        // was built for this step, so the forward can only be reading the view
        // the ring handed out.
        if ring.is_some() {
            anyhow::ensure!(
                owned_input.is_none(),
                "step {}: the ring arm built a host input tensor, so the token is being \
                 uploaded rather than published in place -- the mechanism is wired and \
                 the routing is dead (DESIGN.md §11.5g, #500)",
                tokens.len()
            );
        }
        // `[B, vocab]`, kept batched rather than squeezed: the per-row sampling
        // above indexes it, and at `B = 1` the squeeze happens there instead.
        logits = model
            .forward(input, kv_len, &mut cache)
            .context("decode forward pass")?;

        // **Issue #319: the argmax rides the forward pass's command buffer.**
        //
        // Enqueued HERE -- after `forward`, before the commit below -- the
        // `argmax` dispatch lands in the command buffer the forward pass is
        // still filling, and the `to_scalar` further down reads a value the GPU
        // has already computed.
        //
        // On the default path this same `argmax` runs at the TOP of the next
        // iteration -- so it opens a fresh command buffer for one dispatch and
        // `to_scalar` waits on a second submit. That is §11.2a's fixed ~0.5 ms
        // round trip, measured at 441 µs/token in §11.5a. **The dispatch is the
        // same and the arithmetic is the same; only which buffer carries it
        // moves.**
        //
        // The `to_scalar` is deliberately NOT here: it is the readback, and it
        // belongs after the commit rather than before it.
        //
        // > **#567 deleted the per-token drain these clauses named** (§11.5r).
        // > They said the `synchronize()` below *"drains both"* and that the
        // > default arm's argmax runs *"after `synchronize()` has drained the
        // > queue"*. **The arm's mechanism is unchanged** — which command buffer
        // > carries the argmax is decided by where it is enqueued, not by what
        // > drains afterwards — but what makes the value ready is now
        // > `to_scalar`'s own `flush_and_wait_current` rather than a drain that
        // > ran first, and the **441 µs was measured under the drain**.
        let pending_argmax = if args.sample_in_buffer {
            let mut idx = Vec::with_capacity(batch);
            for r in 0..batch {
                let row = logits.narrow(0, r, 1)?.squeeze(0)?;
                idx.push(row.argmax(candle::D::Minus1).context("argmax")?);
            }
            Some(idx)
        } else {
            None
        };

        // **The per-token drain is DELETED** (lloom #567, `DESIGN.md` §11.5r).
        //
        // What stood here was `device.synchronize()?`, justified as *"what the
        // decode loop does anyway: sampling reads the logits back to the CPU,
        // so the serialization is inherent to the workload, not an artifact of
        // measuring it."* **That second clause was true when it was written and
        // is false now.** §11.5n built the non-blocking read (`to_cpu_fast` 65
        // -> 1) and §11.5o made the completion wait a seam, so the engine no
        // longer requires the stall -- and the harness was asserting one, then
        // measuring arms whose entire purpose is removing it.
        //
        // **And the `max_in_flight = 1` that §11.5p attributed to this drain is
        // NOT the drain's.** A mutation arm restoring the `synchronize` while
        // keeping the prune below still reads **13**
        // (`measurements/issue-567-raw/mutate-nodrain.txt`, ARM A). The 1 was
        // the OBSERVATION POINT: the pre-existing `flush_and_prune` sits in the
        // sample block at the *top* of the iteration (`:3162`), which under the
        // old code ran immediately after the previous step's drain and so read
        // an empty queue by construction. Three arms, one machine:
        //
        // | drain | prune at bottom | `max_in_flight` |
        // |---|---|---|
        // | yes | no | **1** |
        // | yes | **yes** | **13** |
        // | no | yes | **15** |
        //
        // §11.5r corrects the attribution. The deletion is still right, for the
        // reason above rather than for the queue reading.
        //
        // **There is deliberately no `synchronize` arm.** A flag reproducing
        // today's `wall` byte-for-byte would preserve the defect under a name --
        // every figure taken through that call measures a decode loop carrying
        // a forced stall the engine does not need. §11.5r records that fact for
        // the corpus instead, which is what makes the pre-change rows readable
        // rather than comparable.
        //
        // **What replaces it is a commit without a wait**, which is exactly the
        // call §11.5h built for this and which `lfm2-release` already makes
        // (`main.rs:897`): `flush_and_prune` submits the encoder and drops the
        // `in_flight` entries whose GPU work has finished, returning the
        // surviving count. **The pruning is the load-bearing half** -- once
        // nothing blocks per token, `in_flight`'s only two shrink sites are
        // inside the two blocking calls this change stops making, so without it
        // the `Vec` grows one retained command buffer per token for the length
        // of a generation (§11.5h). Errors surface here for the same reason:
        // with no blocking readback on this path this is the only place a
        // failed command buffer is observed, and a dropped error presents as
        // wrong output rather than as a failure (§3.5).
        //
        // Called on **every** arm rather than only the ring's, so the queue
        // bound is a reading this harness takes rather than a property of one
        // flag -- and so `max_in_flight` means the same thing here as it does
        // on release.
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &device {
            let in_flight = m.flush_and_prune()?;
            max_in_flight = max_in_flight.max(in_flight);
        }
        let wall = step_start.elapsed().as_secs_f64();

        // The readback, on the `--sample-in-buffer` arm only. Timed into the
        // same `sample_steps` bucket the default arm uses, so `wall + sample +
        // eos` means the same thing on both arms and the two are comparable
        // (§11.5a's disjointness requirement).
        //
        // Outside the `wall` window deliberately: `wall` is the forward pass's
        // window on every recorded figure this project holds, and moving a term
        // into it would make this arm's `wall` incomparable to all of them.
        if let Some(idx) = pending_argmax {
            let read_start = std::time::Instant::now();
            for (r, slot) in batch_rows.iter_mut().enumerate() {
                *slot = idx[r].to_scalar::<u32>().context("argmax readback")?;
            }
            let read_s = read_start.elapsed().as_secs_f64();
            // Replaces this step's placeholder push, which recorded only the
            // (zero) cost of the skipped top-of-loop sample.
            if let Some(last) = sample_steps.last_mut() {
                last.0 = read_s;
            }
        }

        // **The GPU-side window, and it is the one the drain's removal does NOT
        // move** (lloom #567, `DESIGN.md` §11.5r).
        //
        // `profile::reset`'s own doc states a precondition this loop has just
        // stopped satisfying: call it *"after a device synchronization: an
        // in-flight command buffer's completion handler fires later and would
        // otherwise land in the next window."* **So whether the intervals are
        // still all collected is something this change owes an answer to**, and
        // the counters accumulated here are that answer. The assertion site
        // after the loop carries the table and the control arm.
        //
        // **Two readings, and the first one was wrong.** The run-final
        // `command buffers N (N timed)` line reads `14 (14 timed)` -- but it
        // describes the **last window only**, and accumulating over every
        // window finds a residue. §3.1c on my own instrument: a summary line
        // answering a different question from the one being asked.
        //
        // **Then a control arm found the residue is not this change's.** Rebuilt
        // with the drain restored and these counters kept, it is the same size
        // or larger (21/4357 draining against 14/4409 not, at `n` = 256).
        // `record_command_buffer` increments *both* compared counts, from
        // inside the completion handler itself, so what separates them is the
        // zero-GPU-work filter on the next line and never a buffer still in
        // flight.
        //
        // **What matters for this change is that the quantity does not move**:
        // `gpu busy / step` reads 17.211 / 17.250 / 17.570 ms at `n` = 64 / 128
        // / 256 against the draining baseline's 17.246 / 17.450 / 17.615, taken
        // on the same machine minutes apart -- within **0.24 %**, where the
        // CPU-side `wall` moves by a factor of ten. **The GPU quantity is
        // drain-invariant and the CPU one is not**, which is the whole finding
        // and why the fields below are renamed rather than left to mean
        // something new under an old name.
        let (gpu, disp, fence_counts) = if profiling {
            let s = profile::snapshot();
            profile::reset();
            // Kept from the last iteration: the counters are reset per token, so
            // after the loop there is nothing left to read. This is one token's
            // kernel mix, which is what the inventory should be.
            last_token_kernels = s.by_label.clone();
            timed_cbs += s.timed_command_buffers;
            untimed_cbs += s.command_buffers.saturating_sub(s.timed_command_buffers);
            (
                s.gpu_busy_union_s,
                s.dispatches,
                (s.fence_probes, s.fence_hits, s.fence_waits),
            )
        } else {
            (0.0, 0, (0, 0, 0))
        };

        // The arena is derived from the first `record_steps` decode steps and
        // installed after them, so those steps run on the pool and are excluded
        // from the steady-state average below. Placed after the timing snapshot
        // so the install itself is not counted into a token.
        #[cfg(feature = "metal")]
        if axes.arena {
            if let Some(dev) = metal_device.as_ref() {
                dev.end_decode_step();
                // `tokens.len() - 1` is this step's index, so the install fires
                // on the last recorded step -- the same `record_steps - 1`
                // condition the dispatch-trace harness uses.
                if tokens.len() - 1 == axes.record_steps - 1 {
                    let excluded = dev.arena_recording_excluded();
                    let by_test = dev.arena_recording_excluded_by_test();
                    // Read BEFORE `finish_arena_recording`, which `take`s the
                    // recorder — the reservations are derived from the same two
                    // steps the plan is, and after the take there is nothing
                    // left to derive them from.
                    let reservations = if axes.session_region {
                        dev.arena_recording_session_reservations(kv_len, args.kv_capacity)
                    } else {
                        None
                    };
                    let plan = dev
                        .finish_arena_recording(axes.layout)
                        .context("arena recording produced no allocations")?;
                    let (covered, total) = plan.covered();
                    if let Some((n_excl, n_all)) = excluded {
                        eprintln!(
                            "arena: {n_all} allocations recorded over {} steps, \
                             {n_excl} excluded as session state",
                            axes.record_steps
                        );
                        // Split by detector, because neither subsumes the other
                        // and a zero on either side is a result: size growth
                        // finds the KV cache, cross-step liveness finds the conv
                        // state, which never grows (§5.7, §9.2c).
                        if let Some((by_size, by_step)) = by_test {
                            eprintln!(
                                "arena:   {by_size} caught by size growth (kv_len), \
                                 {by_step} by outliving the step"
                            );
                        }
                    }
                    let monotone = plan.is_bump_reproducible();
                    eprintln!(
                        "arena: {covered} of {total} ordinals served -> {} slots, {} B ({:?})",
                        plan.slots().len(),
                        plan.arena_bytes(),
                        axes.layout,
                    );
                    dev.install_arena(plan, axes.layout)
                        .context("installing the arena")?;
                    // #47's session region, after the arena because it serves
                    // what the arena declines: the offer order in `new_buffer`
                    // is arena first, region second, pool last.
                    if let Some(res) = reservations {
                        if res.is_empty() {
                            // A result, not a nuisance: it means no ordinal grew
                            // between the two recorded steps, which is what
                            // `--kv-append in-place` produces. Said out loud
                            // because an installed-but-empty region and an
                            // uninstalled one are different states and only the
                            // counters below distinguish them (#500).
                            eprintln!(
                                "session region: no growing ordinals were recorded, so nothing \
                                 is reserved -- under `--kv-append in-place` the KV cache does \
                                 not reallocate and this axis has no population to serve"
                            );
                        }
                        let bytes = dev
                            .install_session_region(&res)
                            .context("installing the session region")?;
                        eprintln!(
                            "session region: {} tenants reserved to kv_capacity {}, {} B",
                            res.len(),
                            args.kv_capacity,
                            bytes
                        );
                    }
                    if axes.gpu_offsets {
                        // Checked against the plan that was recorded, not
                        // against the layout's name: a plan that somehow is not
                        // monotone must fail before a wrong offset can be bound
                        // rather than after (§9.3 -- there is no other
                        // detector).
                        if !monotone {
                            anyhow::bail!(
                                "the recorded plan's offsets are not monotone, so a bump \
                                 allocator cannot reproduce them"
                            );
                        }
                        let served = dev
                            .install_gpu_arena_offsets()
                            .context("computing arena offsets on the GPU")?;
                        // The engagement check §2.4 requires: this reads `Gpu`
                        // only after the table was verified equal to the plan
                        // element-wise, so it is evidence the GPU allocator ran
                        // and agreed -- not that a flag was passed.
                        eprintln!(
                            "arena: offsets computed on the GPU and verified against the plan \
                             ({served} ordinals); source now {:?}",
                            dev.arena_offsets()
                        );
                    }
                }
            }
        }

        steps.push((wall, gpu, disp));
        fence_steps.push(fence_counts);
        // Stamped after the timing snapshot, so the clock read is outside the
        // window it describes rather than inside it.
        step_stamps.push((run_record::now_ns(), kv_len));
        kv_len += 1;

        // ---- progress, and it is a MEASUREMENT rather than a courtesy ------
        //
        // A long-context run can fail, and issue #61 requires that **where it
        // stopped is reported rather than silently truncated**. A run that dies
        // at token N having printed nothing says only that it died; one that has
        // printed its progress says *at which `kv_len`*, which is the number the
        // stopping point is expressed in.
        //
        // The pool figures are printed with it because they are the mechanism
        // that would explain such a stop. §9.5k: `free_bytes` is capped by
        // `set_free_budget` and **`pending_bytes` is capped by nothing** — it is
        // §6.3b's stranding (11.6 buffers/token), and under `KvAppend=Cat` the
        // size just freed is never the size next requested, so a freed cache
        // strands instead of being reused. §3.4a-iv names the diagnostic shape
        // exactly: *"a curve that climbs without plateauing as `--n` or `kv_len`
        // rises is a leak or an allocator defect"*, and records that this
        // document has seen that shape **three times and late every time**.
        //
        // On stderr so it cannot land in the middle of the `RESULT` line a
        // parser keys on, and every `--progress-every` tokens rather than every
        // token: at one line per token a 32k run prints 32k lines, and the write
        // itself would sit inside the loop being timed.
        if args.progress_every > 0 && tokens.len().is_multiple_of(args.progress_every) {
            #[cfg(feature = "metal")]
            let sample = metal_device
                .as_ref()
                .map(|d| MemProbe::read(d, tokens.len() as u64, kv_len as u64, wall));
            #[cfg(not(feature = "metal"))]
            let sample: Option<MemProbe> = None;
            match sample {
                Some(s) => {
                    // The by-class timeline (#171) is fed from the same reading
                    // as the wide row, deliberately: two samplers on two
                    // cadences would put the five classes and the device figure
                    // at different instants, and the whole finding is a
                    // COMPARISON between them at one instant. One read, two
                    // sinks.
                    s.emit_telemetry(KV_BYTES_PER_TOKEN);
                    // Written here rather than only at exit, for the reason
                    // `flush_run_telemetry` documents: the arm this timeline
                    // exists to explain is one that dies, and the first version
                    // of this code left a zero-byte file when it did.
                    flush_run_telemetry(&args, &mut rt_written)?;
                    eprintln!("{}", s.human());
                    if let Some(w) = mem_jsonl.as_mut() {
                        let _ = writeln!(w, "{}", s.jsonl());
                        // Flushed at every sample, deliberately, and this is the
                        // one place a buffered writer would be wrong. **The run
                        // being measured is one that dies** — #204's `Cat` arm
                        // exhausted memory at `kv_len` 2284 — and the samples
                        // nearest the failure are the whole point. A buffer
                        // holding them when the process is killed loses exactly
                        // the tail the artifact exists for.
                        //
                        // The cost is bounded by `--progress-every`, which is
                        // the flag that decides how often this happens at all,
                        // and the write is outside the per-token timing window.
                        let _ = w.flush();
                    }
                }
                None => eprintln!(
                    "progress: token {:>7}  kv_len {:>7}  wall {:7.3} ms",
                    tokens.len(),
                    kv_len,
                    wall * 1e3
                ),
            }
        }
    }

    // ---- #171's timeline: the FINAL flush ---------------------------------
    //
    // The series has been written incrementally in the loop; this is the last
    // one plus the engagement report.
    //
    // **It was written at exit only, and running it is what found that wrong.**
    // The first `Cat` arm died of `kIOGPUCommandBufferCallbackErrorOutOfMemory`
    // at `kv_len` 2259 and wrote a **zero-byte** timeline, because `?` on the
    // forward pass returns before this point -- so the instrument was absent
    // from exactly the run it exists for. That is #206's own lesson at a second
    // sink: *"the run being measured is one that dies, and the samples nearest
    // the failure are the whole point."* Its `--mem-jsonl` flushes per sample
    // for that reason and survived the same failure with 89 rows; this one did
    // not, and now does.
    //
    // **`engaged()` is checked and reported rather than assumed.** It is
    // `ENABLED && accepted > 0`, so it distinguishes a build that could not
    // record, from one that could and was not asked to, from one that recorded
    // -- the discrimination §2.4 requires after #69's determinism arm consumed
    // its own `OnceLock` and reported a passing digest for the unchanged path.
    #[cfg(feature = "metal")]
    if args.run_telemetry_jsonl.is_some() {
        use candle::kernels::metal::run_telemetry as rt;
        let accepted = rt::accepted();
        anyhow::ensure!(
            rt::engaged(),
            "--run-telemetry-jsonl was given and the recorder never engaged \
             (compiled={}, accepted={accepted}). A build that cannot record and a run \
             that did not are different states, and an empty timeline is not a \
             measurement. Check that --progress-every is non-zero: the sampler is \
             driven by it.",
            rt::compiled()
        );
        flush_run_telemetry(&args, &mut rt_written)?;
        eprintln!("run-telemetry: {accepted} samples+marks recorded");
    }

    // ---- summarize -------------------------------------------------------

    // Under `--arena` the first `record_steps` tokens run on the pool, because
    // the arena does not exist until its plan does. Averaging them in would mix
    // two allocators into one ms/token figure and attribute the difference to
    // the arena -- the confound this harness exists to remove. They are dropped
    // in addition to `--warmup`, not instead of it: they are a different
    // allocator, where warmup is a cold one.
    let arena_skip = if axes.arena { axes.record_steps } else { 0 };
    let warmup = args.warmup.max(arena_skip).min(steps.len());
    // The csw window's threshold, checked rather than assumed to be this one
    // (lloom #412). If the clamp ever bites, `invol_csw_per_token` would divide
    // a window that opened at one index by a token count taken from another —
    // a figure that is internally consistent and wrong, which is §2.4's shape.
    anyhow::ensure!(
        seam_warmup_floor == warmup || steps.is_empty(),
        "the csw window opened at {seam_warmup_floor} where the steady-state window \
         opens at {warmup}; invol_csw_per_token would divide by the wrong denominator"
    );
    let steady = &steps[warmup..];
    anyhow::ensure!(
        !steady.is_empty(),
        "no steady-state tokens left after {warmup} excluded tokens \
         (--warmup {}, arena recording {arena_skip}); raise --n",
        args.warmup
    );

    let n_steady = steady.len() as f64;
    let wall_mean: f64 = steady.iter().map(|s| s.0).sum::<f64>() / n_steady;
    let gpu_mean: f64 = steady.iter().map(|s| s.1).sum::<f64>() / n_steady;
    let disp_mean: f64 = steady.iter().map(|s| s.2 as f64).sum::<f64>() / n_steady;
    // The fence-map traffic per step over the same steady window (lloom #732):
    // reads of the cross-encoder map, reads that found a writer, and
    // `waitForFence` calls. Zero with profiling off.
    let (fence_probes_mean, fence_hits_mean, fence_waits_mean) = {
        let fs = fence_steps
            .get(warmup.min(fence_steps.len())..)
            .unwrap_or(&[]);
        let n = fs.len().max(1) as f64;
        (
            fs.iter().map(|f| f.0 as f64).sum::<f64>() / n,
            fs.iter().map(|f| f.1 as f64).sum::<f64>() / n,
            fs.iter().map(|f| f.2 as f64).sum::<f64>() / n,
        )
    };

    // ---- the speculative denominator (#284, §10.2i) ----------------------
    //
    // **The means above are per STEP, and under `--speculate` a step is a
    // window.** §10.2i states the problem exactly: a verify pass *"consumes K
    // and emits between 1 and K, so `wall_ms_per_token` needs a denominator that
    // does not exist there."* This is that denominator — tokens **accepted** per
    // window — and it is what makes the reported figure GPU-busy per **accepted
    // token** rather than per window.
    //
    // # Why the ratio and not a per-window divide
    //
    // Because acceptance varies between windows and a mean of per-window ratios
    // is not the per-token cost. §3.4a's own rule, established for traffic and
    // then found to generalise (§3.4a-vii's energy case): **sum over the band and
    // divide once.** A mean of `wall_j / accepted_j` weights a 1-accept window
    // as heavily as an 8-accept one; the ratio of sums weights each window by
    // what it produced, which is the quantity a token cost is.
    //
    // At `K = 0` this is exactly 1.0 — a step *is* a token — so
    // `wall_ms_per_token` keeps its name, its meaning and its comparability with
    // every row already recorded. The corpus does not re-interpret.
    let spec_tokens_per_step: f64 = if spec.windows > 0 {
        // Steady-state windows only, matching the means: the accepted total is
        // over every window, so scale it by the fraction that survived warmup.
        // Exact when no warmup is excluded, which is the measured case for a
        // speculative run.
        let steady_frac = n_steady / spec.windows.max(1) as f64;
        (spec.accepted as f64 * steady_frac) / n_steady
    } else {
        1.0
    };
    // Per **accepted token**, which is the quantity #273's prediction is stated
    // in and the one no speculative run has ever produced (#284's opening).
    let wall_per_token = wall_mean / spec_tokens_per_step;
    let gpu_per_token = gpu_mean / spec_tokens_per_step;

    // ---- the step, bracketed by OBSERVATION (lloom #567, §11.5r) ---------
    //
    // **`wall` no longer bounds a token, and this is the field that does.**
    //
    // With the per-token drain deleted, `wall` is the interval from opening the
    // step clock to the *submit* returning -- CPU encode plus whatever
    // back-pressure the queue applies -- and it is **not** a bracket around a
    // token's execution. Measured on this tree at `--token-ring`, same machine,
    // minutes apart:
    //
    // | `n` | draining `wall`/step | non-draining `wall`/step | `gpu busy`/step, both |
    // |---|---|---|---|
    // | 64 | 18.047 | **1.793** | 17.246 / 17.211 |
    // | 128 | 18.565 | **2.638** | 17.450 / 17.250 |
    // | 256 | 18.752 | **3.619** | 17.570 / 17.570 |
    //
    // The draining figure is flat across `n` and the non-draining one **climbs**
    // -- it is measuring how full the queue is, which is an `n`-dependent
    // property of the harness rather than a per-token cost. **A `tok/s` computed
    // from it would read ~558 at `n` = 64**, a fabricated 10x, which is §1.3's
    // *"a speedup that is too large is a bug signal"* arriving as a bracket
    // error rather than as a corrupted cache.
    //
    // **So the denominator becomes the GPU's own clock**, which is the
    // observation route: `GPUStartTime`/`GPUEndTime` per command buffer, unioned
    // over the window (`profile.rs`), already recorded on every arm and shown
    // above to be **drain-invariant** -- it reads within 0.24 % of the draining
    // baseline at all three `n`. That is what §2.5.1's *engine* layer is, and it
    // is the layer the mechanisms this harness selects actually move.
    //
    // The step's own binding is below, once the sample, EOS and seam means it
    // sums exist — see `step_per_token`.

    // Standard deviation, so a mean is never reported without its spread.
    let wall_sd = (steady
        .iter()
        .map(|s| (s.0 - wall_mean).powi(2))
        .sum::<f64>()
        / n_steady)
        .sqrt();
    let gpu_sd = (steady.iter().map(|s| (s.1 - gpu_mean).powi(2)).sum::<f64>() / n_steady).sqrt();

    let mut walls: Vec<f64> = steady.iter().map(|s| s.0).collect();
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wall_min = walls[0];
    let wall_med = walls[walls.len() / 2];
    let wall_max = walls[walls.len() - 1];

    // ---- the sample bucket (lloom #172, #319) ----------------------------
    //
    // **The median as well as the mean, and the median is what compares.**
    // §11.5a establishes that `sample_ms_per_token` has §6.6b's shape -- a floor
    // plus a one-sided tail -- so its mean rises over a long run while its
    // median is flat, and *"a mean of a right-tailed distribution is not a
    // location estimate"*. §11.2a reports this term as flat in `kv_len` and that
    // is correct on the median only.
    let sample_steady = &sample_steps[warmup.min(sample_steps.len())..];
    let (sample_mean, sample_med, eos_mean) = if sample_steady.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let n = sample_steady.len() as f64;
        let mean = sample_steady.iter().map(|s| s.0).sum::<f64>() / n;
        let eos = sample_steady.iter().map(|s| s.1).sum::<f64>() / n;
        let mut v: Vec<f64> = sample_steady.iter().map(|s| s.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (mean, v[v.len() / 2], eos)
    };

    // The seam's three terms over the same steady-state window (lloom #412).
    //
    // **The median beside the mean, for §6.6b's reason and it is sharper here.**
    // A `blocking_send` into a channel that is not full is a lock and a memcpy;
    // one into a channel that IS full is a park until the consumer drains. Those
    // are different populations, so the mean of the mixture is not a location
    // estimate of either — and the gap between mean and median is what says
    // whether the second population was ever entered. A seam whose mean and
    // median agree never blocked.
    let seam_steady = &seam_steps[warmup.min(seam_steps.len())..];
    let (seam_detok_mean, seam_alloc_mean, seam_send_mean, seam_send_med, seam_send_max) =
        if seam_steady.is_empty() {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let n = seam_steady.len() as f64;
            let detok = seam_steady.iter().map(|s| s.detokenize_s).sum::<f64>() / n;
            let alloc = seam_steady.iter().map(|s| s.alloc_s).sum::<f64>() / n;
            let send = seam_steady.iter().map(|s| s.send_s).sum::<f64>() / n;
            let mut v: Vec<f64> = seam_steady.iter().map(|s| s.send_s).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max = v.last().copied().unwrap_or(0.0);
            (detok, alloc, send, v[v.len() / 2], max)
        };
    // The terms only the streaming arm pays, and they are ONE timed window:
    // `emit` times `to_string` and `blocking_send` together because the timer is
    // larger than the allocation (see `seam.rs::emit`). `seam_alloc_mean` is
    // therefore structurally 0.0 and is not added — adding it would double-count
    // nothing, but reading it as a term would claim a resolution the instrument
    // does not have. Detokenize is excluded for the reason the loop's own
    // comment gives: both arms pay it.
    debug_assert_eq!(
        seam_alloc_mean, 0.0,
        "emit times allocate and send as one window; a nonzero alloc mean means \
         that changed and `seam_mean` is now missing a term"
    );
    let seam_mean = seam_send_mean;

    // **The step, and it is what survived #567's deletion intact**
    // (`DESIGN.md` §11.5r).
    //
    // §11.2b defines a step as `wall + sample + eos` (plus #412's seam) and this
    // harness sums them because the windows are disjoint. **Deleting the
    // per-token drain did not change that** — it moved the wait from the first
    // term into the second, `sample / token` going **0.4204 -> 16.8235 ms**,
    // and left the sum where it was: **18.6779 ms against the draining
    // baseline's 18.4674**, same machine, minutes apart.
    //
    // So this, and **not** `wall`, is the denominator every rate and share is
    // taken against from here on. It is the one quantity on this harness that
    // stays comparable with every pre-#567 row, which is what makes the
    // deletion checkable against the corpus rather than merely asserted.
    //
    // Bound once, so the `RESULT` line, the store and the human-readable block
    // cannot drift into three different denominators — §2.4's defect, and one
    // that would be invisible because each number is individually plausible.
    let step_per_token = wall_per_token + sample_mean + eos_mean + seam_mean;

    // Context switches over the steady-state window, per token.
    //
    // Read here rather than at the last token so the window closes after the
    // final `synchronize()` — a switch taken while the last forward pass drains
    // belongs to the window it happened in.
    let csw_end = seam::csw();
    let steady_tokens = seam_steady.len().max(1) as f64;
    let (vol_csw_per_token, invol_csw_per_token) = match csw_at_warmup {
        Some((v0, i0)) => (
            (csw_end.0.saturating_sub(v0)) as f64 / steady_tokens,
            (csw_end.1.saturating_sub(i0)) as f64 / steady_tokens,
        ),
        None => (0.0, 0.0),
    };
    // What the consumer saw. `seam_received` differing from the number of
    // fragments sent means the channel closed early and every send after that
    // point measured an error return rather than a hand-off (§2.4).
    let seam_consumer = seam_tx.take().map(|s| s.finish()).unwrap_or_default();

    // The instrument, priced against the terms (§2.4, §6.4a). Run here — after
    // the decode loop, after the channel is closed, before anything is printed
    // — so it is outside every measured window and its ~20 ms of looping cannot
    // reach a figure. Both arms run it, so the two RESULT lines carry the same
    // self-cost and a reader can see it did not move between them.
    let (seam_instant_pair_s, seam_alloc_priced_s) = seam::self_cost("hello");

    // Dispatch counts must be identical token to token if the sequence is
    // stable; report the range so a reader can see whether it was.
    let disp_min = steady.iter().map(|s| s.2).min().unwrap_or(0);
    let disp_max = steady.iter().map(|s| s.2).max().unwrap_or(0);

    // `layer_types` is the authoritative source for which layers hold attention
    // (`DESIGN.md` §5.2: `full_attn_idxs` is null in this checkpoint, so indices
    // must not be hardcoded).
    let layer_is_attn: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| matches!(t, LayerType::FullAttention))
        .collect();
    let n_attn = layer_is_attn.iter().filter(|b| **b).count();
    let weight_bytes = language_weight_bytes(&config, &layer_is_attn, dtype.size_in_bytes());

    // Row 0's stream, for the `--sample-in-buffer` gate (issue #319). After the
    // loop, so the write is outside every timed window.
    if let Some(path) = args.dump_tokens.as_ref() {
        use std::io::Write as _;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("creating token dump at {}", path.display()))?,
        );
        for t in &tokens {
            writeln!(f, "{t}")?;
        }
        f.flush()?;
    }

    println!("=== machine / configuration ===");
    println!("dtype                 {dtype:?}");
    println!(
        "device                {}",
        if args.cpu { "cpu" } else { "metal" }
    );
    println!(
        "layers                {} ({} attention, {} conv)",
        config.num_hidden_layers,
        n_attn,
        config.num_hidden_layers - n_attn
    );
    println!(
        "hidden / ffn          {} / {}",
        config.hidden_size, config.intermediate_size
    );
    println!("vocab                 {}", config.vocab_size);
    println!(
        "lm weight bytes       {weight_bytes} ({:.3} GB)",
        weight_bytes as f64 / 1e9
    );
    println!(
        "profiling             {}",
        if profiling {
            "on (CANDLE_METAL_PROFILE=1)"
        } else if cfg!(feature = "metal") {
            "OFF -- rerun with CANDLE_METAL_PROFILE=1 for the CPU/GPU split"
        } else {
            "unavailable (Metal profiling is not compiled); CPU wall-time remains measured"
        }
    );
    println!();

    println!("=== prefill ({} tokens) ===", prompt_ids.len());
    println!(
        "wall                  {:.2} ms",
        prefill_wall.as_secs_f64() * 1e3
    );
    if let Some(p) = &prefill_profile {
        println!("gpu busy (union)      {:.2} ms", p.gpu_busy_union_s * 1e3);
        println!("gpu busy (sum)        {:.2} ms", p.gpu_busy_sum_s * 1e3);
        println!("dispatches            {}", p.dispatches);
        println!("encoders              {}", p.encoders);
        println!(
            "command buffers       {} ({} timed)",
            p.command_buffers, p.timed_command_buffers
        );
    }
    println!();

    // **Time to first token, decomposed rather than asserted** (§2.5.4, #388
    // item 2). The residual is printed because it is the whole check on the
    // claim beside the definition: this loop is sample-then-forward, so the
    // first token is the prefill's own argmax and TTFT is `tokenize + prefill +
    // one sample` rather than §2.5.4's `prefill + one decode step + one
    // sample`. A residual near the sample bucket confirms that; a residual near
    // a step would mean the bracket is closing a step late, which is the error
    // this print makes visible instead of leaving to be inferred.
    if let Some(t) = ttft {
        let ttft_ms = t.as_secs_f64() * 1e3;
        let prefill_ms = prefill_wall.as_secs_f64() * 1e3;
        println!("=== time to first token (request layer, §2.5.1) ===");
        println!("ttft                  {ttft_ms:.2} ms   <- prompt in -> first token out");
        println!("  of which prefill    {prefill_ms:.2} ms");
        println!(
            "  residual            {:.2} ms   <- tokenize + the first sample",
            ttft_ms - prefill_ms
        );
        println!(
            "  cf. one step        {:.2} ms   <- NOT inside ttft: the first token is",
            step_per_token * 1e3
        );
        println!("                                     the PREFILL's argmax (see the bracket)");
        println!("warm process: model already loaded. A cold-start TTFT adds §2.4a's load.");
        println!();
    }

    // #47's session region, reported whenever one is installed.
    //
    // **This is the engagement proof, and without it a null is unreadable**
    // (#500): an installed region that served nothing and an uninstalled one
    // produce identical pool counters, and only `offers`/`hits` separate them.
    // The two declines are printed separately because they want opposite
    // responses -- `unplanned` says widen the plan, `capacity` says the
    // reservation was too small -- and a single number could mean either.
    #[cfg(feature = "metal")]
    if let Some(dev) = metal_device.as_ref() {
        if let Some(c) = dev.session_region_counters() {
            println!();
            println!("=== session region (#47, DESIGN.md §9.2x) ===");
            println!(
                "  tenants             {}  reserving {} B",
                dev.session_region_tenants().unwrap_or(0),
                dev.session_region_bytes().unwrap_or(0)
            );
            println!(
                "  offers / hits       {} / {}  ({:.1} % served)",
                c.offers,
                c.hits,
                if c.offers > 0 {
                    100.0 * c.hits as f64 / c.offers as f64
                } else {
                    0.0
                }
            );
            println!(
                "  declined            {} unplanned, {} past the reservation",
                c.declined_unplanned, c.declined_capacity
            );
            println!();
        }
    }

    println!(
        "=== decode, steady state ({} steps, {} warmup excluded, B={batch}) ===",
        steady.len(),
        warmup
    );
    // **"step" rather than "token", and at `B > 1` they are different
    // quantities** (`DESIGN.md` §13.4a). A step advances every row by one
    // token, so it produces `B` tokens; the whole weight-amortisation claim is
    // that step time barely moves while `B` rises. Calling this "wall / token"
    // at `B > 1` would report a per-step figure under a per-token name, which
    // is §2.4's *"check whether the measurement tool measures the thing its
    // output names"* — the defect that removed `tok_per_s` from the sibling
    // harness (#102).
    // **Labelled `submit` since #567** (`DESIGN.md` §11.5r): with the per-token
    // drain deleted this closes when the encoder is submitted, not when the
    // token is done. It is printed because it is a real quantity and a useful
    // one -- it is how the queue's back-pressure becomes visible, climbing
    // 1.79 -> 3.62 ms as `n` goes 64 -> 256 where the draining figure was flat
    // -- but it is **not** the denominator anything is quoted against.
    println!(
        "submit / step         {:.3} ms  (sd {:.3}, min {:.3}, med {:.3}, max {:.3})",
        wall_mean * 1e3,
        wall_sd * 1e3,
        wall_min * 1e3,
        wall_med * 1e3,
        wall_max * 1e3
    );
    println!("                      ^ CPU encode+submit only; NOT a token's duration (§11.5r)");
    // The figure §13.4a's table is in: `B / step_time`. At `B = 1` it is the
    // per-sequence rate this harness has always printed, so the row does not
    // change meaning for an unflagged run.
    //
    // **Over the STEP rather than the submit window** (#567, §11.5r). Dividing
    // by the latter printed **540 tok/s** against a true ~54, which is §1.3's
    // too-large-to-be-real signal arriving as a bracket error. The step is the
    // denominator that survived the deletion intact -- the wait moved from
    // `wall` into `sample` and the sum is unchanged -- so this row still means
    // what it has always meant and still compares with every recorded one.
    //
    // Needs no `profiling` guard: all three terms are CPU clocks the harness
    // takes whether or not the GPU counters ran.
    let step_mean = wall_mean + sample_mean + eos_mean + seam_mean;
    println!(
        "aggregate throughput  {:.2} tok/s  ({batch} seq x {:.2} tok/s)",
        batch as f64 / step_mean,
        1.0 / step_mean
    );
    if batch > 1 {
        println!(
            "per-sequence          {:.3} ms/token  ({:.2} tok/s)",
            step_mean * 1e3,
            1.0 / step_mean
        );
    }
    // `step_mean` is the per-STEP sum and `step_per_token` the per-accepted-token
    // one; at `K = 0` they are equal, and under `--speculate` only the second
    // is a token cost (§10.2i). Asserted rather than assumed so the two cannot
    // drift into meaning different things on one line.
    debug_assert!(
        spec.windows > 0 || (step_mean - step_per_token).abs() < 1e-12,
        "the step's per-step and per-token sums disagree at K = 0"
    );
    if profiling {
        println!(
            "gpu busy / step       {:.3} ms  (sd {:.3})",
            gpu_mean * 1e3,
            gpu_sd * 1e3
        );
        // **The non-GPU term is taken against the STEP since #567**, not
        // against `wall` (§11.5r). It was `wall - gpu`, and `wall` is now the
        // submit window while `gpu` is the token's execution -- neither
        // contains the other, so that subtraction is the one §2.5.2 obligation
        // 2 forbids and it read **-15.42 ms** on this change's first run.
        //
        // The step does contain both, because the wait the drain used to hold
        // is now inside `sample` and the step is still their sum. So the term
        // is recovered rather than lost, and **against a denominator that is
        // one token rather than one submission** -- which is arguably what it
        // should always have been: §11.2b records that `wall` and `sample` are
        // disjoint windows and that quoting a share of `wall` is #299's double
        // count in miniature.
        println!(
            "non-gpu / step        {:.3} ms  ({:.1}% of step)",
            (step_mean - gpu_mean) * 1e3,
            100.0 * (step_mean - gpu_mean) / step_mean
        );
        println!(
            "dispatches / step     {:.1}  (min {}, max {})",
            disp_mean, disp_min, disp_max
        );
        println!(
            "fence map / step      {:.1} reads, {:.1} hits, {:.1} waits",
            fence_probes_mean, fence_hits_mean, fence_waits_mean
        );
        // **The engagement proof §2.4 requires, and the reason it is computed
        // rather than read off the line above** (issue #249).
        //
        // The claim under test is that one weight sweep serves `B` rows, so the
        // *forward pass* must issue the same dispatches at every `B` — if it
        // does not, the batch is a loop over sequences and the harness is
        // measuring `B` separate decodes rather than one batched one.
        //
        // The counters span sampling (§11.2a: they reset at `:717` and snapshot
        // at `:716`, where the wall clock does not), and this harness samples
        // **per row** — `B` `fast_argmax_f32` dispatches, one per row. So the
        // raw count rises by `B - 1` for a reason that has nothing to do with
        // the forward pass, and reading it as-is would report a batch that
        // *is* flat as one that is not. Subtracting the known sampling term is
        // what makes the quantity discriminate.
        //
        // **Whether it discriminates in THIS regime is checked rather than
        // assumed** — #248 found a dispatch-count proof that expires, because
        // two `AttnImpl` arms coincide at 547 above `kv_len` 1023. Here the
        // discriminator is a *difference between B arms of one binary*, not
        // between two kernels, and the failure it must detect is a per-row loop,
        // which would scale the forward term by `B` — 517 against 1034 at B=2,
        // not a coincidence two arms could land on.
        let sampling_dispatches = batch.saturating_sub(1) as f64;
        let disp_forward = disp_mean - sampling_dispatches;
        println!(
            "  -- of which sampling {:.1}  ({batch} rows x 1 argmax, {:.1} left for the \
             forward pass)",
            sampling_dispatches + 1.0,
            disp_forward
        );
        println!(
            "dispatches / fwd pass {:.1}  <- FLAT IN B iff the weight read is shared",
            disp_forward
        );
        if disp_mean > 0.0 {
            println!(
                "non-gpu / dispatch    {:.1} us",
                (wall_mean - gpu_mean) * 1e6 / disp_mean
            );
        }
        println!();
        println!("=== roofline ===");
        // **Per STEP, not per token, and that is the whole point.** §13.4a:
        // *"weight traffic is per step, not per token"* — the same 5.394 GB
        // serves every row, so this figure does not scale with `B` while the
        // aggregate throughput above does.
        println!("weights per step      {:.3} GB", weight_bytes as f64 / 1e9);
        if batch > 1 {
            println!(
                "weights per token     {:.3} GB  (per step / B -- the term batching \
                 amortises)",
                weight_bytes as f64 / batch as f64 / 1e9
            );
        }
        if gpu_mean > 0.0 {
            println!(
                "implied bandwidth     {:.1} GB/s  (weights / gpu busy)",
                weight_bytes as f64 / gpu_mean / 1e9
            );
        }
        println!(
            "implied bw vs wall    {:.1} GB/s  (weights / wall)",
            weight_bytes as f64 / wall_mean / 1e9
        );
    }

    // The correctness gate's verdict, printed whether it fired or not — a gate
    // that reports only on failure is one a reader cannot tell was run (§2.4).
    if batch > 1 {
        println!();
        println!("=== batch correctness (issue #249) ===");
        match args.batch_check {
            false => println!("row streams           NOT CHECKED (--batch-check false)"),
            true => match batch_divergence {
                None => println!(
                    "row streams           IDENTICAL across all {batch} rows, {} steps",
                    tokens.len()
                ),
                Some((step, row, want, got)) => println!(
                    "row streams           **DIVERGED** at step {step}: row 0 emitted {want}, \
                     row {row} emitted {got}"
                ),
            },
        }
    }

    // ---- the speculative report (#284) ------------------------------------
    //
    // Printed whether or not anything was rejected, for the batch gate's reason:
    // a gate that reports only on failure is one a reader cannot tell was run
    // (§2.4).
    if args.speculate > 0 {
        println!();
        println!("=== speculative verify (issue #284, mechanism #89) ===");
        println!("K                     {}", args.speculate);
        println!("proposer              {}", args.spec_proposer);
        println!("windows               {}", spec.windows);
        println!(
            "proposed / accepted   {} / {}  (accept rate {:.4})",
            spec.proposed,
            spec.accepted,
            if spec.proposed > 0 {
                spec.accepted as f64 / spec.proposed as f64
            } else {
                0.0
            }
        );
        println!(
            "tokens per window     {spec_tokens_per_step:.4}  \
             (the per-token denominator, §10.2i)"
        );
        if spec.corrupted > 0 {
            println!(
                "corrupted             {}  (the wrong-proposer arm's engagement proof)",
                spec.corrupted
            );
        }
        if batch > 1 {
            match (args.spec_accept_check, spec.accept_divergence) {
                (false, _) => {
                    println!("per-row accept        NOT CHECKED (--spec-accept-check false)")
                }
                (true, None) => println!(
                    "per-row accept        IDENTICAL across all {batch} rows, {} windows",
                    spec.windows
                ),
                (true, Some((step, row, want, got))) => println!(
                    "per-row accept        **DIVERGED** at step {step}: row 0 accepted \
                     {want}, row {row} accepted {got}"
                ),
            }
        }

        // **The vacuity checks, asserted rather than printed** (§2.4, after
        // #69's determinism run reported a passing digest for a path its flag
        // never reached). A `--speculate` run that opened no window has measured
        // the ordinary decode loop under the speculative arm's name, and a
        // `wrong:<N>` arm that corrupted nothing is the oracle arm wearing a
        // different label — the case #89 names as the stronger of its two,
        // because a mechanism transparent only when nothing is rejected has not
        // been shown to roll back at all.
        anyhow::ensure!(
            spec.windows > 0,
            "--speculate {} was passed and no verify window ran, so every figure above is \
             the ordinary decode loop's under the speculative arm's name",
            args.speculate
        );
        anyhow::ensure!(
            args.spec_proposer == "oracle" || spec.corrupted > 0,
            "--spec-proposer {} corrupted nothing, so this arm is the oracle arm under \
             another name and the rollback was never exercised",
            args.spec_proposer
        );
        // **The output-identity gate, and it is the claim the whole mechanism
        // rests on** (#89, #284): under greedy decoding a speculative run must
        // emit *exactly* what the non-speculative run emits, whatever the
        // proposer does. The oracle sequence is that non-speculative run — this
        // harness generated it, from this configuration, minutes ago — so the
        // comparison is direct rather than against a recorded digest.
        //
        // Asserted over the whole stream rather than trusted from the per-step
        // check inside the loop: that one fires at the *carried* token, and the
        // tokens emitted from inside a window are the half a rollback could
        // corrupt. This covers both.
        anyhow::ensure!(
            tokens.as_slice() == &oracle[..tokens.len()],
            "the speculative stream is not the non-speculative one. Under greedy \
             decoding they are identical by construction (#89), so this is a defect in \
             the verifier — a rejection changes which proposals are wasted, never what \
             is emitted."
        );
        println!(
            "output identity       IDENTICAL to the non-speculative stream over {} tokens",
            tokens.len()
        );

        // **The oracle arm's acceptance is asserted, and this is the check that
        // catches the off-by-one #89 records.** A perfect proposer accepts every
        // position, so anything less means the verifier is comparing position
        // `j` against the wrong proposal — which produces a stable token stream
        // and an acceptance rate that reads as a mediocre proposer rather than
        // as a defect. The stream being correct is exactly why this needs its
        // own assertion.
        if args.spec_proposer == "oracle" {
            anyhow::ensure!(
                spec.accepted == spec.proposed,
                "the oracle proposer accepted {} of {} positions. A proposer that proposes \
                 what the target emits is accepted at every position by construction \
                 (#89), so a shortfall is a misalignment in the verifier — not a \
                 rejection.",
                spec.accepted,
                spec.proposed
            );
        }
    }
    println!();

    if args.per_token {
        println!("=== per token ===");
        // `kv_len` is printed beside the index because #61's whole question is
        // ms/token **as a function of kv_len**, and the two are not the same
        // number: the index starts at 0 and `kv_len` starts at the prompt
        // length. Deriving one from the other downstream means a plot's x-axis
        // depends on a prompt length recorded somewhere else, which is exactly
        // the reconstruction §3.4a-iii prices at 15 %. Here it is measured.
        for (i, (w, g, d)) in steps.iter().enumerate() {
            let tag = if i < warmup { " [warmup]" } else { "" };
            let kv = step_stamps.get(i).map(|s| s.1).unwrap_or(0);
            println!(
                "{i:4}  kv_len {kv:7}  wall {:8.3} ms  gpu {:8.3} ms  dispatches {d}{tag}",
                w * 1e3,
                g * 1e3
            );
        }
        println!();
    }

    if profiling && !last_token_kernels.is_empty() {
        // One decode token's kernel mix, captured before the per-token reset.
        let total: u64 = last_token_kernels.iter().map(|(_, c)| c).sum();
        println!("=== kernels in one decode token ({total} dispatches) ===");
        for (name, count) in &last_token_kernels {
            println!("{count:6}  {name}");
        }
        println!();
    }

    // ---- the sample bucket, reported (lloom #172, #319) -------------------
    //
    // **A step is `wall + sample + eos`, DISJOINT rather than a share.** The
    // `wall` window opens after the sample on the default arm and closes before
    // the readback on the `--sample-in-buffer` arm, so §11.2's non-GPU figure
    // structurally cannot contain either term (§11.2a, §11.5a).
    {
        let step_ms = (wall_mean + sample_mean + eos_mean + seam_mean) * 1e3;
        println!("=== sample and EOS (lloom #172, #319) ===");
        println!(
            "  arm                 {}",
            if args.sample_in_buffer {
                "in-buffer  (argmax rides the forward's command buffer)"
            } else {
                "default    (argmax opens its own command buffer)"
            }
        );
        println!(
            "  sample / token      {:.4} ms  (med {:.4})",
            sample_mean * 1e3,
            sample_med * 1e3
        );
        println!("  EOS test / token    {:.4} ms", eos_mean * 1e3);
        println!(
            "  step = wall+sample+eos+seam  {step_ms:.4} ms   -> {:.2} tok/s effective",
            if step_ms > 0.0 { 1e3 / step_ms } else { 0.0 }
        );
        if args.sample_split && split_steps.len() > warmup {
            let s = &split_steps[warmup..];
            let n = s.len() as f64;
            let argmax_mean = s.iter().map(|x| x.0).sum::<f64>() / n;
            let read_mean = s.iter().map(|x| x.1).sum::<f64>() / n;
            println!();
            println!("  --- split (DIAGNOSTIC: the extra sync inflates the total) ---");
            println!(
                "  argmax + sync       {:.4} ms  <- the submit",
                argmax_mean * 1e3
            );
            println!(
                "  4-byte readback     {:.4} ms  <- blit + flush_and_wait_current",
                read_mean * 1e3
            );
        }
        println!();
    }

    // The seam (lloom #412). Printed on BOTH arms, because a null arm reporting
    // nothing and a null arm reporting zeros are different claims and only the
    // second is checkable.
    {
        println!("=== the front end's seam (lloom #412) ===");
        println!(
            "  arm                 {}",
            if args.seam {
                "streaming  (to_string + blocking_send into mpsc(cap))"
            } else {
                "null       (the shipped decode loop; no channel, no String)"
            }
        );
        println!(
            "  detokenize / token  {:.4} ms  <- BOTH arms pay this",
            seam_detok_mean * 1e3
        );
        println!(
            "  to_string + send    {:.4} ms  (med {:.4}, max {:.4})  <- ONE window",
            seam_send_mean * 1e3,
            seam_send_med * 1e3,
            seam_send_max * 1e3
        );
        println!(
            "  seam / token        {:.4} ms  <- to_string + send, the streaming arm's own",
            seam_mean * 1e3
        );
        // §2.4's *"check whether the measurement tool is the cost"*, run rather
        // than assumed — and here it fires. The allocation is NOT timed
        // separately because the timer is larger than it; this prints both
        // numbers so a reader can check that decision rather than take it.
        // Outside every measured window: after the loop, after every mean.
        {
            let (pair_s, alloc_s) = (seam_instant_pair_s, seam_alloc_priced_s);
            println!("  --- instrument self-cost (§6.4a, §2.4) ---");
            println!(
                "  Instant pair        {:.1} ns   <- the timer",
                pair_s * 1e9
            );
            println!(
                "  to_string(5B)       {:.1} ns   <- the term, {:.1}x SMALLER than the timer",
                alloc_s * 1e9,
                if alloc_s > 0.0 { pair_s / alloc_s } else { 0.0 }
            );
            println!(
                "  => the allocation is not separately resolvable by a timer, and is priced\n     \
                    at count x price instead: 1 fragment/token x {:.1} ns = {:.6} ms/token",
                alloc_s * 1e9,
                alloc_s * 1e3
            );
        }
        // The engagement proof, and it is what makes a null readable (§2.4).
        // A streaming arm whose consumer received nothing measured a closed
        // channel; a null arm whose consumer received something is not null.
        println!(
            "  consumer received   {} fragments, {} bytes, {} boundaries  (producer emitted {} bytes)",
            seam_consumer.received, seam_consumer.bytes, seam_consumer.boundaries, seam_text_bytes
        );
        println!(
            "  csw / token         {:.3} involuntary, {:.3} voluntary",
            invol_csw_per_token, vol_csw_per_token
        );
        println!();
    }

    // The configuration is part of the RESULT line, not a separate note.
    // #99's diagnosis was that the harness emits a near-complete cache key
    // missing the commit and the variant axes; both are here, so a row can name
    // the configuration it was taken under rather than being read as
    // all-defaults by whoever finds it later.
    // **The existing field names are kept even though they now say "token"
    // where the quantity is per STEP.** At `B = 1` — every run this project has
    // ever recorded, and the default — the two are the same number, so the
    // corpus keeps parsing and older rows stay comparable (`ingest.rs` splits
    // `config=[…]` keyed rather than positionally, and these are top-level
    // fields a rename would break). The batched quantities are **added**
    // alongside with names that say what they are, which is the same choice
    // §7.1a records for `config_line()`: append rather than re-spell, so an
    // older line parses unchanged and a newer one carries more.
    //
    // `batch=` is on the line for the reason #171's `UNRECORDED` exists: a run
    // taken before this field existed is not thereby *at* B=1, it is a run the
    // question cannot be asked of. Now it can.
    // **The engagement proof for `ScratchSizing`, and it is not the config
    // line** (#234, and #257's third species: an axis that renders correctly
    // for a mechanism that did not run). §2.4 requires a flag be shown to have
    // engaged **from a quantity it should have changed**, and what this arm
    // changes is *how many bytes one attention layer's partials occupy*.
    //
    // Computed from the resolved arm and the final `kv_len` rather than
    // measured at the allocation, which is a weaker check and an honest one:
    // it says what the selected policy *asks for*, so it discriminates the
    // three arms — 0 on `Generic`/`Sdpa`, `live` on `Grow`, the reservation on
    // `Reserve` and a rung on `Bucket` — and a reader comparing it against
    // `allocated_bytes` in `--mem-jsonl` has the observed side. **A figure the
    // harness computes is a prediction that agrees, not an observation**
    // (§5.5a's own caution), and it is labelled `asks` for that reason.
    let scratch_asks_bytes = flash_scratch_ask_bytes(&axes, &config, kv_len, args.kv_capacity);
    #[cfg(feature = "metal")]
    let (to_cpu_fast, to_cpu_blit) = candle::metal_backend::to_cpu_counts();
    #[cfg(not(feature = "metal"))]
    let (to_cpu_fast, to_cpu_blit) = (0u64, 0u64);

    // ---- #567's engagement counters --------------------------------------
    //
    // **Each of the three mechanisms has a different one, and a silent
    // fallback in any of them makes the whole run a null for the wrong
    // reason.** That is the issue's own requirement and it is #500's finding
    // stated per-arm: a mechanism with no callers regresses nothing, so an
    // unmoved number on a green board carries no information at all.
    //
    // | arm | counter | what a zero means |
    // |---|---|---|
    // | `--token-ring` | `ring_publishes` | the ring never published |
    // | `--nonblocking-eos` | `peek_hit` + `poll_reads` | neither read ran |
    // | `--completion-wait` | `completion_signals` | no signal was encoded |
    //
    // `to_cpu_fast`/`to_cpu_blit` are already on the line above and are the
    // fourth: #454 measures which readback arm fires as unstable run to run
    // (blit 12/1/2/5 across four runs of one binary), which is the same order
    // as the effect -- so **if the split differs between arms the comparison
    // moved the allocator, not the mechanism**, and the result is void
    // regardless of sign.
    #[cfg(feature = "metal")]
    let (peek_hit, peek_unmapped) = candle::metal_backend::peek_counts();
    #[cfg(not(feature = "metal"))]
    let (peek_hit, peek_unmapped) = (0u64, 0u64);
    // #578 split the peek into two arms and `peek_hit` counts only one of them.
    // The poll now reads through an address resolved once, which calls nothing
    // that increments `peek_hit` -- so on a mapped run that counter reads
    // **zero for a path that ran every iteration**, which is exactly the shape
    // §3.1c warns about: a zero that is indistinguishable from a path that
    // never ran. `poll_reads` is the counter for the arm that actually carried
    // the run, and the pair is what makes either readable.
    let poll_reads = lloom::generation::token_ring::poll_read_count();
    // Read back from the device rather than from the flag (§7.1a-i), and at a
    // different height from `peek_hit` rather than a duplicate of it:
    // `peek_hit` counts reads that skipped a wait, this counts waits that were
    // satisfied without blocking.
    #[cfg(feature = "metal")]
    let (cw_signals, cw_polls) = match &device {
        Device::Metal(m) => m.completion_counts(),
        _ => (0, 0),
    };
    #[cfg(not(feature = "metal"))]
    let (cw_signals, cw_polls) = (0u64, 0u64);
    // The listener arm's own counter (#583), and it is a THIRD one because the
    // two above cannot see this arm's failure. `shared-event-listener` reports
    // `(signals > 0, polls == 0)` whether its notification block delivered or
    // never fired and the wait fell back to blocking -- and the fallback is
    // correct, so the values agree and both digests are unmoved either way.
    // `completion_fires` is the only quantity that separates them.
    #[cfg(feature = "metal")]
    let (cw_fires, cw_already) = match &device {
        Device::Metal(m) => (m.completion_fires(), m.completion_already_complete()),
        _ => (0u64, 0u64),
    };
    #[cfg(not(feature = "metal"))]
    let (cw_fires, cw_already) = (0u64, 0u64);

    // **The engagement assertions, not merely the prints.** §11.5i's finding is
    // that an unmoved digest on an arm that never engaged carries no
    // information, and `lfm2-determinism` already asserts the completion arm's
    // signal count for exactly that reason. Asserted here too, so a `RESULT`
    // line from this harness cannot report an arm it did not run.
    //
    // Placed after the run rather than before it because engagement is a
    // property of what happened, and the measurement has already been taken by
    // now -- an assertion here costs a failed report, never a lost run.
    anyhow::ensure!(
        !args.token_ring || ring_publishes > 0,
        "--token-ring published nothing over {} tokens: the arm did not engage, so every \
         figure on this line belongs to the baseline path (DESIGN.md §11.5i, §2.4)",
        tokens.len()
    );
    // The peek is allowed to decline -- `poll_eos` returns `None` when the pool
    // hands back a buffer Metal did not map, which §15.2 #12a makes a SIZE
    // property and #454 measures as unstable run to run -- so the assertion is
    // that it was *reached*, not that it always succeeded. `poll_fallbacks`
    // beside it is what separates "the peek declined" from "the peek never ran".
    anyhow::ensure!(
        !args.nonblocking_eos || poll_iters > 0,
        "--nonblocking-eos ran no poll iterations, so the non-blocking read never \
         executed (DESIGN.md §11.5n, §2.4)"
    );
    #[cfg(feature = "metal")]
    anyhow::ensure!(
        axes.completion_wait == Some("synchronize") || cw_signals > 0,
        "completion arm {:?} encoded no signal, so this run's figures are the incumbent's \
         -- the mechanism did not engage (DESIGN.md §11.5o, §11.5i, §2.4)",
        axes.completion_wait
    );
    // **The listener arm needs its own assertion, and the one above cannot be
    // it** (#583). A listener that never fires falls back to the blocking wait,
    // which computes the right answer -- so `cw_signals` is nonzero, both
    // digests are unmoved, every value assertion passes, and the run reports an
    // arm it did not take. That is §7.1a-i's wrong-arm shape in the one place
    // no pre-existing counter could see it, and Apple's own guidance is that a
    // deallocated listener stops delivering silently.
    // **`completion_signals` counts COMMITS and `completion_fires` counts
    // WAITS, and the two are ~13.5:1 by construction** (#583). A signal is
    // encoded in every `commit_swap_locked` — five call sites, and
    // `CutPolicy::Counter(50)` cuts a buffer every 50 of 675 dispatches per
    // token — while only `flush_and_wait_current` waits, once per token. So
    // `fires ≈ n` is the listener firing on every wait there is, and comparing
    // it against `signals` measures the cut policy rather than the mechanism.
    //
    // Measured, `completion_already_complete` is **0**, which excludes the
    // other candidate explanation (that the readbacks arrive after the GPU has
    // retired the buffer) rather than leaving it to be assumed.
    #[cfg(feature = "metal")]
    let cw_engaged_waits = cw_signals.saturating_sub(cw_already);
    #[cfg(feature = "metal")]
    anyhow::ensure!(
        axes.completion_wait != Some("shared-event-listener")
            || cw_engaged_waits == 0
            || cw_fires > 0,
        "completion arm shared-event-listener fired no notification block over {} tokens \
         while {cw_engaged_waits} waits actually waited: the wait fell back to blocking \
         and computed the right answer, so this run's figures belong to the incumbent \
         (DESIGN.md §11.5u, §2.4)",
        tokens.len()
    );
    // **The zero-GPU-work residue, and it is the DRAIN'S INNOCENCE** (lloom
    // #567, `DESIGN.md` §11.5r).
    //
    // This counter was written to catch a defect the drain's removal was
    // predicted to cause -- an in-flight command buffer whose completion
    // handler had not run when its window was snapshotted, whose interval is
    // therefore missing from the union, and which would make
    // `gpu_ms_per_token` under-count while every digest, every other counter
    // and the token stream stayed identical (§15.1a's INEXPRESSIVE class).
    //
    // **It fired, and then a control arm falsified the diagnosis.** Rebuilt
    // with the drain RESTORED and this counter kept, the residue is still
    // there and is not smaller:
    //
    // | `n` | drain present | drain deleted |
    // |---|---|---|
    // | 64 | 5 / 1089 | 9 / 1076 |
    // | 128 | 7 / 2177 | 16 / 2236 |
    // | 256 | **21 / 4357** | **14 / 4409** |
    //
    // At `n` = 256 the *draining* arm is worse. **So the residue is a property
    // of the instrument and not of the drain**, and every `gpu_ms_per_token`
    // this project has recorded carries it. The mechanism is visible once
    // looked for: `COMMAND_BUFFERS` is incremented **inside**
    // `record_command_buffer`, which *is* the completion handler
    // (`profile.rs:98`), so both counts describe buffers that have already
    // completed -- and what separates them is the filter on the next line,
    // `start <= 0.0 || end <= start`, which drops a buffer that **encoded no
    // GPU work**. A count of not-yet-completed buffers is not a quantity this
    // instrument can express at all.
    //
    // **This is therefore a warning rather than a failure, and the reason it is
    // not silent is that it is real**: 0.1-0.5 % of buffers, non-monotone in
    // `n`, present on both arms. Made an error would fail every pre-existing
    // arm equally, which is CONTRIBUTING.md §1.2's *"weaken a test so it
    // passes"* inverted -- a gate that fails on the state the corpus chose is
    // one that gets disabled (§3.1a's own note).
    if profiling && untimed_cbs > 0 {
        eprintln!(
            "note: {untimed_cbs} of {} command buffers encoded no GPU work and are \
             excluded from the union (profile.rs:101). Present on the draining arm too \
             -- 21/4357 at n=256 against this arm's 14/4409 -- so it is the instrument's \
             residue rather than this change's (DESIGN.md §11.5r).",
            timed_cbs + untimed_cbs
        );
    }
    // #366: readbacks that found the buffer unmapped *after* the GPU wait and
    // fell back to the blit. This is the window that used to segfault; printing
    // it is what makes its frequency a number rather than a crash report.
    #[cfg(feature = "metal")]
    let to_cpu_null_after_wait = candle::metal_backend::to_cpu_null_after_wait();
    #[cfg(not(feature = "metal"))]
    let to_cpu_null_after_wait = 0u64;
    // #477's engagement quantity, read from the kernel cache after the run so
    // it counts what the decode actually compiled rather than what a plan says
    // it would (§2.4: verify an arm from a quantity the flag should have
    // changed). Under `MergedUnit=off` the sixteen non-MLX sources each hold
    // their own library; under `on` they share one.
    let (metal_library_count, metal_pipeline_count) = {
        #[cfg(feature = "metal")]
        {
            match &device {
                candle::Device::Metal(d) => d.kernels().cache_sizes(),
                // Not a Metal device, so there is no library cache. Reported as
                // zeros with the pipeline total beside them, which is the
                // reading `(0, 0)` is supposed to have.
                _ => (0, 0),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            (0usize, 0usize)
        }
    };
    println!(
        "RESULT label={} n={} warmup={} batch={} submit_ms_per_token={:.4} \
         gpu_ms_per_token={:.4} nongpu_ms_per_token={:.4} timed_cbs={} untimed_cbs={} \
         sample_ms_per_token={:.4} sample_med_ms_per_token={:.4} eos_ms_per_token={:.4} \
         sample_in_buffer={} to_cpu_fast={} to_cpu_blit={} \
         to_cpu_null_after_wait={} step_ms_per_token={:.4} \
         dispatches_per_token={:.1} \
         dispatches_per_forward={:.1} fence_probes_per_token={:.1} \
         fence_hits_per_token={:.1} fence_waits_per_token={:.1} \
         aggregate_tok_per_s={:.2} per_seq_tok_per_s={:.2} \
         batch_streams_identical={} spec_k={} spec_proposer={} spec_windows={} \
         spec_proposed={} spec_accepted={} spec_corrupted={} spec_accept_rate={:.4} \
         spec_tokens_per_step={:.4} spec_rows_agree={} prefill_ms={:.2} \
         ttft_ms={} \
         prompt_tokens={} weight_bytes={} scratch_asks_bytes={} dtype={:?} hit_eos={} \
         seam={} seam_capacity={} seam_ms_per_token={:.4} \
         seam_detok_ms_per_token={:.4} seam_alloc_ns_priced={:.1} \
         seam_send_ms_per_token={:.4} seam_send_med_ms_per_token={:.4} \
         seam_send_max_ms_per_token={:.4} seam_instant_pair_ns={:.1} \
         seam_received={} seam_bytes={} \
         invol_csw_per_token={:.3} vol_csw_per_token={:.3} \
         metal_libraries={} metal_pipelines={} \
         token_ring={} ring_publishes={} ring_eos_publishes={} ring_reads_token={} \
         poll={} poll_iters={} poll_fallbacks={} max_in_flight={} \
         nonblocking_eos={} peek_hit={} peek_unmapped={} poll_reads={} \
         completion_wait={} completion_signals={} completion_polls={} \
         completion_fires={} completion_already_complete={} \
         profiling={} config=[{}] candle_commit={} seed={} temp={} top_p={}",
        args.label,
        args.n,
        warmup,
        batch,
        // **`submit_ms_per_token`, renamed from `wall_ms_per_token` by #567**
        // (`DESIGN.md` §11.5r), because the quantity changed and §2.4's defect
        // is a field whose name promises one thing while the arithmetic
        // delivers another. With the per-token drain deleted this is the CPU's
        // encode-and-submit window, **not** a bracket around a token: it reads
        // 1.79 ms at `n` = 64 and 3.62 at 256, climbing with how full the queue
        // is, where the draining figure was flat at ~18.0-18.8.
        //
        // **A parser keying on `wall_ms_per_token` will now MISS rather than
        // silently mis-read**, which is the intended failure direction: an
        // absent field compares equal to nothing, where a wrong one compares
        // equal to every run of that arm (§7.1a-i).
        wall_per_token * 1e3,
        gpu_per_token * 1e3,
        // **`nongpu_ms_per_token` is now `step - gpu`, not `wall - gpu`**
        // (#567, §11.5r). The old subtraction went **negative** -- `-15.4189`
        // on this change's first run -- because `wall` stopped containing the
        // GPU while `gpu` still held a whole token. The step contains both, so
        // the term is recovered against a denominator that is one *token*
        // rather than one *submission*; §11.2b already records that a share of
        // `wall` was #299's double count in miniature.
        (step_per_token - gpu_per_token) * 1e3,
        // **The instrument's own residue, on the line rather than in a
        // comment.** `untimed` counts command buffers that completed having
        // encoded no GPU work and are therefore excluded from the union
        // (`profile.rs:101`). Measured present on the *draining* arm too, so it
        // is not this change's -- and it is reported because a reader comparing
        // a `gpu_ms_per_token` across two runs should be able to see whether
        // they excluded the same fraction.
        timed_cbs,
        untimed_cbs,
        // The sample bucket (#172, #319). DISJOINT from `wall`, so a step is
        // the sum of the three rather than `wall` alone -- which is why
        // `step_ms_per_token` is printed beside them rather than left to be
        // recomputed by a reader who may not know the windows do not overlap.
        sample_mean * 1e3,
        sample_med * 1e3,
        eos_mean * 1e3,
        args.sample_in_buffer,
        // #336's engagement proof, on every line: which `to_cpu` path the
        // readbacks took. A run claiming the fast arm with `to_cpu_fast=0`
        // measured the other one (§2.4, after #69's vacuous determinism run).
        to_cpu_fast,
        to_cpu_blit,
        to_cpu_null_after_wait,
        // **The step is STILL `wall + sample + eos + seam`, and the reason is
        // the finding** (lloom #567, `DESIGN.md` §11.5r).
        //
        // The obvious move once `wall` stops holding the forward pass is to
        // swap in the GPU-bounded term. **That double-counts**, and measuring
        // it is what showed why: with the per-token drain deleted,
        // `sample / token` moves **0.4204 -> 16.8235 ms**. The sample is a
        // blocking readback, so once nothing drains ahead of it *it* is what
        // waits for the GPU. **The wait did not disappear; it moved from `wall`
        // into `sample`** -- and `engine + sample` would count that same GPU
        // time twice, reading 34.19 ms against a true ~18.7.
        //
        // So the three terms remain disjoint and still sum, exactly as §11.2b
        // defines -- what changed is only *which* of them carries the wait.
        // **`step_ms_per_token` is therefore comparable with every pre-#567
        // row**, which is the one field that is, and it is why the deletion can
        // be checked against the corpus at all: 18.6779 here against the
        // draining baseline's 18.4674, same machine, minutes apart.
        //
        // `+ seam_mean` is 0.0 on the null arm and on every run this field has
        // ever carried (lloom #412).
        step_per_token * 1e3,
        disp_mean,
        // Floored at 0 rather than subtracted blindly: with profiling off
        // `disp_mean` is 0.0, and `0 - (B-1)` printed **-7.0** at B=8 — a count
        // that cannot be negative, rendered as one. The forward-pass term is
        // only meaningful when the counters ran, and a nonsense value in a
        // field a reader keys on is worse than a zero (§2.4). Found by running
        // the admission arm, which does not profile.
        if disp_mean > 0.0 {
            disp_mean - batch.saturating_sub(1) as f64
        } else {
            0.0
        },
        fence_probes_mean,
        fence_hits_mean,
        fence_waits_mean,
        // §13.4a's own quantity: `B / step_time`. Under `--speculate` a step
        // yields `spec_tokens_per_step` tokens per row rather than one, so the
        // rate is over the tokens the windows kept — the same divisor the
        // per-token fields take, applied to the same numerator.
        //
        // **Divided by the STEP since #567, not by `wall`** (§11.5r). `wall` is
        // now the submit window, and `B / 1.85 ms` would have printed **540
        // tok/s** against a true ~54 — §1.3's *"a speedup that is too large is
        // a bug signal"*, and it would have been a fabricated 10x on the field
        // this project's headline numbers are quoted in.
        //
        // The step rather than the engine term, because the step is what a
        // token costs the caller and it is the one denominator still
        // comparable with every recorded row (see `step_ms_per_token` above).
        batch as f64 / step_per_token,
        1.0 / step_per_token,
        // Tri-state rather than a bool: at `B = 1` there is nothing to compare,
        // and reporting `true` there would claim a check that did not run —
        // §15.1a's vacuous-instrument class, in a field.
        match (batch, args.batch_check, batch_divergence.is_some()) {
            (1, _, _) => "n/a",
            (_, false, _) => "unchecked",
            (_, true, false) => "true",
            (_, true, true) => "FALSE",
        },
        args.speculate,
        // Stated even at `K = 0`, where it names the arm that did not run: a
        // field absent from a line is a run the question cannot be asked of
        // (#171's `UNRECORDED`), and this one is always answerable.
        args.spec_proposer,
        spec.windows,
        spec.proposed,
        spec.accepted,
        spec.corrupted,
        // The acceptance rate a proposer is judged on — a different quantity
        // from the per-token divisor, which is why both are on the line.
        if spec.proposed > 0 {
            spec.accepted as f64 / spec.proposed as f64
        } else {
            0.0
        },
        spec_tokens_per_step,
        // The per-row accept test's verdict, tri-state for `batch_streams_identical`'s
        // reason: at `B = 1` there are no rows to compare and at `K = 0` no
        // window ran, so `true` in either case would claim a check that did not.
        match (
            args.speculate,
            batch,
            args.spec_accept_check,
            spec.accept_divergence.is_some(),
        ) {
            (0, _, _, _) | (_, 1, _, _) => "n/a",
            (_, _, false, _) => "unchecked",
            (_, _, true, false) => "true",
            (_, _, true, true) => "FALSE",
        },
        prefill_wall.as_secs_f64() * 1e3,
        // **Time to first token, the request layer** (§2.5.1, §2.5.4, #388
        // item 2). Rendered `n/a` rather than `0.00` when unset, for #171's
        // `UNRECORDED` reason applied to a field rather than to an axis: the
        // speculative arm does not set it (see the declaration), and a `0.00`
        // there would read as a measured zero — the fastest TTFT ever recorded
        // — instead of as a question that arm cannot answer. §3.4a-x's rule in
        // a timing field: an absent value is a scope question before it is a
        // magnitude one.
        match ttft {
            Some(d) => format!("{:.2}", d.as_secs_f64() * 1e3),
            None => "n/a".to_string(),
        },
        prompt_ids.len(),
        weight_bytes,
        scratch_asks_bytes,
        dtype,
        hit_eos,
        // The seam (lloom #412). `seam=` is on the line for #171's `UNRECORDED`
        // reason: a run taken before this field existed is not thereby *on the
        // null arm*, it is a run the question cannot be asked of.
        args.seam,
        args.seam_capacity,
        // The streaming arm's own cost — `to_string` plus `blocking_send`.
        // Detokenize is beside it rather than inside it because both arms pay it.
        seam_mean * 1e3,
        seam_detok_mean * 1e3,
        // **Count × price, not a timer** — one fragment per token at the
        // measured per-allocation cost. Named `_ns_priced` rather than
        // `_ms_per_token` so no reader takes it for an observed interval: the
        // timer that would have observed it is 3.3× larger than it (`seam.rs`).
        seam_alloc_priced_s * 1e9,
        seam_send_mean * 1e3,
        // The median and the max, because a bounded channel has two populations
        // and only the gap between them says whether the second was entered.
        seam_send_med * 1e3,
        seam_send_max * 1e3,
        // The timer's own cost, on the line rather than in a footnote: it is
        // what says whether the term above it was resolvable at all, and it is
        // a property of the machine and the toolchain rather than of the run.
        seam_instant_pair_s * 1e9,
        // Engagement (§2.4): a streaming arm with `seam_received=0` measured a
        // closed channel, and a null arm with a nonzero one is not null.
        seam_consumer.received,
        seam_consumer.bytes,
        // #412's preemption instrument. In-process because the vol/invol split
        // is not externally readable — `pti_csw` is the sum (see `seam.rs`).
        invol_csw_per_token,
        vol_csw_per_token,
        // #477's `MergedUnit` engagement proof, and it exists because **the
        // dispatch census cannot see this axis**. The census counts which
        // kernels ran; the merged arm changes which *library* each was
        // compiled from and dispatches an identical set — measured
        // byte-identical between the two arms, which is the correct outcome
        // and is exactly why a census-only engagement claim would be vacuous.
        // Reading §12.2i's instrument onto this axis is §15.1a's INEXPRESSIVE
        // class, which #522 paid a run for.
        //
        // The library count is what the arm moves by construction: 19 → 4.
        // **The pipeline total is what licenses it** (§14.5d): a probe that
        // compiled nothing reports `(0, 0)`, and without the second number a
        // small library count is indistinguishable from a broken read.
        metal_library_count,
        metal_pipeline_count,
        // ---- #567's three async arms and their engagement counters --------
        //
        // On the line for #171's `UNRECORDED` reason: a run taken before these
        // fields existed is not thereby *on the baseline arm*, it is a run the
        // question cannot be asked of.
        args.token_ring,
        ring_publishes,
        ring_eos_publishes,
        // **The scope difference from `lfm2-release`, on the line rather than
        // only in a source comment.** That harness accumulates tokens
        // device-side and reads them once after the clock, so its ring arm
        // makes no per-token round trip; this one needs each id inside the loop
        // for the EOS test, the seam and `--dump-tokens`, so it reads 4 bytes
        // back per token. Two runs whose `ring_publishes` agree and whose
        // `ring_reads_token` differ measured different mechanisms, and without
        // this field they would render identically.
        args.token_ring,
        poll_arm(axes.poll),
        poll_iters,
        poll_fallbacks,
        max_in_flight,
        args.nonblocking_eos,
        peek_hit,
        peek_unmapped,
        poll_reads,
        axes.completion_wait_arm(),
        cw_signals,
        cw_polls,
        cw_fires,
        cw_already,
        profiling,
        axes.config_line(),
        // Passed in by the caller rather than read from git here.
        //
        // A `git rev-parse` at runtime names the tree the binary is *run* from,
        // which is not necessarily the tree it was *built* from -- and a commit
        // field that is silently wrong is worse than one that is absent, since
        // #99's whole point is that this field decides whether a cached result
        // is valid. `unknown` is the honest default; the run script supplies it.
        std::env::var("LLOOM_CANDLE_COMMIT").unwrap_or_else(|_| "unknown".into()),
        args.seed,
        args.temperature,
        args.top_p,
    );

    // ---- the run record, at exit (lloom issue #171) ----------------------
    //
    // Deliberately here: after the last token, after every mean is computed,
    // and after `RESULT` is printed. Nothing below runs inside a measured
    // window. (This used to say *"after `device.synchronize()`"*; #567 deleted
    // that call from the loop, and the property that mattered — being outside
    // every window — is unchanged.)
    //
    // Absent `--run-store`, nothing is written and nothing fails. `lloom-sample`
    // decided the same way and gives the reason: telemetry failure must degrade
    // to "no telemetry", never to "failed measurement".
    if let Some(store) = args.run_store.as_deref() {
        let m = |v: f64| run_record::Measured::Value(v);
        // With profiling off the harness cannot compute a GPU split -- the
        // figures come from `GPUStartTime`/`GPUEndTime`, which only the
        // profiling path collects. Recorded NEVER RUN rather than as the 0.0
        // the `RESULT` line prints, which is the defect the corpus already
        // carries: a real zero standing where "not measured" was meant.
        // **The same per-accepted-token divisor the `RESULT` line takes**, and
        // that is a correctness requirement rather than a nicety: the store and
        // the line describing one run with two different denominators is the
        // shape §2.4 names — a field whose name promises one quantity while the
        // arithmetic delivers another — and it would be invisible, because both
        // numbers are individually plausible.
        // **`nongpu` is `step - gpu` since #567**, not `wall - gpu`
        // (`DESIGN.md` §11.5r). With the per-token drain deleted `wall` is the
        // submit window while `gpu` is a whole token's execution, so the old
        // subtraction is between two windows where neither contains the other
        // (§2.5.2 obligation 2) and it went **negative** — `-15.4189` on this
        // change's first run. The step contains both, because the wait moved
        // from `wall` into `sample` and the step is still their sum, so the
        // term is recovered rather than dropped.
        //
        // **The same denominator the `RESULT` line takes**, which is the
        // correctness requirement the comment above states: the store and the
        // line describing one run with two different denominators is §2.4's
        // shape and would be invisible, because both numbers are individually
        // plausible.
        let (gpu_metric, nongpu_metric, disp_metric) = if profiling {
            (
                m(gpu_per_token * 1e3),
                m((step_per_token - gpu_per_token) * 1e3),
                m(disp_mean),
            )
        } else {
            (
                run_record::Measured::NeverRun,
                run_record::Measured::NeverRun,
                run_record::Measured::NeverRun,
            )
        };

        let rec = run_record::RunRecord {
            run_id: args.run_id.clone().unwrap_or_else(|| {
                format!(
                    "{}-{}",
                    if args.label.is_empty() {
                        "run"
                    } else {
                        &args.label
                    },
                    std::process::id()
                )
            }),
            harness: "lfm2-decode-profile",
            label: args.label.clone(),
            candle_commit: std::env::var("LLOOM_CANDLE_COMMIT")
                .ok()
                .filter(|v| !v.is_empty() && v != "unknown"),
            lloom_commit: std::env::var("LLOOM_COMMIT")
                .ok()
                .filter(|v| !v.is_empty() && v != "unknown"),
            binary_uuid: run_record::self_uuid(),
            binary_path: std::env::current_exe()
                .ok()
                .map(|p| p.display().to_string()),
            axes: axes.axis_pairs(),
            machine: sysctl_str("hw.model"),
            macos_build: sysctl_str("kern.osversion"),
            load_start: load_at_start,
            load_end: load_average(),
            under_lease: std::env::var("LLOOM_ARB_LEASE").is_ok(),
            n: args.n,
            warmup,
            seed: args.seed,
            temperature: args.temperature,
            top_p: args.top_p,
            prompt_tokens: prompt_ids.len(),
            kv_len_first: step_stamps.first().map(|s| s.1).unwrap_or(0),
            kv_len_last: step_stamps.last().map(|s| s.1).unwrap_or(0),
            dtype: format!("{dtype:?}"),
            profiling_compiled: cfg!(feature = "metal"),
            profiling_enabled: profiling,
            // **The STEP, not the submit window** (lloom #567, `DESIGN.md`
            // §11.5r). This field is shared with `lfm2-release`, and
            // `lloom_runs`' store pools runs whose axis vectors agree — so
            // writing the 1.85 ms submit window into a field release fills with
            // an ~18 ms step would put two quantities differing by 10x under
            // one name. That is exactly the shape §11.5q found when release
            // rendered no `config=[…]` and its runs pooled with this
            // harness's, and it is worth avoiding twice.
            //
            // The step is the leg that still means what this field has always
            // meant, and §11.5q already records release's own figure as an
            // identity over the same three terms — so the two harnesses agree
            // here by construction rather than by coincidence.
            wall_ms_per_token: m(step_per_token * 1e3),
            gpu_ms_per_token: gpu_metric,
            nongpu_ms_per_token: nongpu_metric,
            dispatches_per_token: disp_metric,
            prefill_ms: m(prefill_wall.as_secs_f64() * 1e3),
            weight_bytes: m(weight_bytes as f64),
            steps: steps
                .iter()
                .zip(step_stamps.iter())
                .enumerate()
                .map(
                    |(i, ((wall, gpu, disp), (t_end_ns, kv)))| run_record::Step {
                        index: i,
                        warmup: i < warmup,
                        wall_ms: wall * 1e3,
                        gpu_ms: if profiling { Some(gpu * 1e3) } else { None },
                        dispatches: if profiling { Some(*disp) } else { None },
                        t_end_ns: *t_end_ns,
                        kv_len: *kv,
                    },
                )
                .collect(),
            telemetry_path: args.telemetry_path.clone(),
        };
        match rec.emit(store) {
            Ok(()) => eprintln!("run record appended to {store}"),
            // A failed write is reported and does not fail the run: the
            // measurement already happened and its `RESULT` line is printed.
            Err(e) => eprintln!("WARNING: could not write the run record to {store}: {e}"),
        }
    }

    Ok(())
}

/// The profile the run's axes name (`profiles-and-memory-design.md` §3): the
/// preset, then the `--kv-append` / `--conv-state` flags over it exactly as
/// `run` resolves them, read off `Config::profile_key` at the contract in
/// force. `run` resolves the config again in full; a divergence between the
/// two is refused by `Model::new` (`Config::check_shape`), never served.
fn shape_key(args: &Args, matches: &ArgMatches) -> Result<lloom::profile::ProfileKey> {
    let given = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);
    let model_dir = args
        .model_dir
        .clone()
        .or_else(default_model_dir)
        .context("no --model-dir given and no LFM2 snapshot in the HF cache")?;
    let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let mut config = parse_config(&raw)?;
    let preset = candle_examples::lfm2_setup::preset(&args.preset)?;
    if let Some(apply) = preset {
        apply(&mut config);
    }
    let set = |id: &str| preset.is_none() || given(id);
    if set("kv_append") {
        config.kv_append = match args.kv_append.as_str() {
            "paged" => KvAppend::Paged,
            "cat" => KvAppend::Cat,
            "in-place" => KvAppend::InPlace,
            other => {
                anyhow::bail!("--kv-append must be `cat`, `in-place` or `paged`, got `{other}`")
            }
        };
    }
    if set("conv_state") {
        config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    }
    Ok(config.profile_key())
}

#[cfg(all(test, not(feature = "metal")))]
mod cpu_telemetry_tests {
    use super::*;

    #[test]
    fn telemetry_entry_refuses_before_checkpoint_shape_or_output_access() {
        let matches = Args::command()
            .try_get_matches_from([
                "lfm2-decode-profile",
                "--cpu",
                "--model-dir",
                "/dev/null/not-a-checkpoint",
                "--run-telemetry-jsonl",
                "/dev/null/not-an-output",
            ])
            .unwrap();
        let args = Args::from_arg_matches(&matches).unwrap();
        let error = execute(args, matches).unwrap_err().to_string();
        assert!(
            error.contains("needs the `metal-run-telemetry` feature"),
            "{error}"
        );
    }

    #[test]
    fn help_remains_available_before_cpu_telemetry_setup() {
        let error = Args::command()
            .try_get_matches_from(["lfm2-decode-profile", "--help"])
            .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
        assert!(error.to_string().contains("--run-telemetry-jsonl"));
    }

    #[test]
    fn absent_metal_telemetry_is_not_reported_as_an_empty_recording() {
        assert!(!run_telemetry_compiled());
        let mut args = Args::try_parse_from(["lfm2-decode-profile", "--cpu"]).unwrap();
        let mut written = 17;
        flush_run_telemetry(&args, &mut written).unwrap();
        assert_eq!(written, 17);
        args.run_telemetry_jsonl = Some(PathBuf::from("unavailable-telemetry.jsonl"));
        let error = flush_run_telemetry(&args, &mut written).unwrap_err();
        assert!(error.to_string().contains("metal-run-telemetry"));
        assert_eq!(written, 17);
    }

    #[test]
    fn cpu_profile_shim_keeps_gpu_profiling_disabled() {
        assert!(!profile::enabled());
        let snapshot = profile::snapshot();
        assert_eq!(
            (
                snapshot.fence_probes,
                snapshot.fence_hits,
                snapshot.fence_waits
            ),
            (0, 0, 0)
        );
    }
}
