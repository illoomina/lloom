//! What every served family shares: the text side (tokenizer, template, the
//! ids requests are shaped by), the cells' timing records, the memory gauges,
//! the profile snapshot, the device, the HF-cache lookups. Moved out of the
//! LFM2 engine (`lfm2.rs`, the former `lloom-serve/engine.rs`) unchanged so a
//! second family (`qwen4exp.rs`) reuses them rather than restating them.

use crate::ReadbackArm;
use anyhow::Result;
use hf_chat_template::ChatTemplate;
use lloom_tensor::Device;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;

/// The EOS id set that never fires: every arm's EOS mechanism still runs and
/// simply never matches, so a run produces exactly `max_tokens` tokens.
pub const NEVER_EOS: [u32; 1] = [u32::MAX];

/// Text: the tokenizer, the checkpoint's own chat template, and the ids the
/// requests are shaped by.
pub struct Text {
    pub tokenizer: Tokenizer,
    pub template: Option<ChatTemplate>,
    pub bos_token: String,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    /// The benchmark prompt seed, encoded without special tokens and tiled to
    /// the requested prefill length (the MLX sidecar's `make_tokens`).
    pub seed_tokens: Vec<u32>,
}

/// What one throughput cell measured, in the sidecar's units.
pub struct PrefillTiming {
    pub prompt_tokens: usize,
    pub batch: usize,
    pub prefill_ms: f64,
    /// Forwards the batch's prompts took.
    pub forwards: usize,
    pub mem: MemReport,
}

pub struct DecodeTiming {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub batch: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub readback: ReadbackArm,
    /// What `decode_ms` brackets.
    pub bracket: &'static str,
    /// Per row, the tokens decoded inside the bracket.
    pub row_tokens: Vec<usize>,
    /// Per row, the speculative rounds run and the mean accepted per round
    /// (empty on the K=1 arms).
    pub row_rounds: Vec<usize>,
    pub row_accepted: Vec<f64>,
    /// Per row, the host time its rounds spent in each phase — draft,
    /// verify (enqueue + wait), observe — summed over the row's rounds, in
    /// ms (`RoundTiming`; empty on the K=1 arms). On the batched loop a
    /// round's phases are the batch's, so the rows differ only by how many
    /// rounds each ran.
    pub row_draft_ms: Vec<f64>,
    pub row_verify_ms: Vec<f64>,
    pub row_observe_ms: Vec<f64>,
    /// The Metal profile counters over the timed bracket, when
    /// `CANDLE_METAL_PROFILE=1` (a census run: the recorder costs a lock and a
    /// hash per dispatch, so its timing is not the arm's; `None` otherwise).
    pub profile: Option<ProfileReport>,
    pub mem: MemReport,
}

/// What the dispatch recorder saw over one timed bracket: the counts, the
/// GPU-busy union over the bracket's command buffers, and the per-kernel
/// dispatch census (`lloom_tensor::kernels::metal::profile::Snapshot`).
pub struct ProfileReport {
    pub dispatches: u64,
    pub command_buffers: u64,
    pub encoders: u64,
    pub gpu_busy_ms: f64,
    pub gpu_span_ms: f64,
    pub by_label: Vec<(String, u64)>,
    /// GPU time by kernel from each command buffer's LAST dispatch label —
    /// `(label, buffers, total_ms, p50_ms)`; exact only at
    /// `CANDLE_METAL_COMPUTE_PER_BUFFER=1` (one dispatch per buffer), else a
    /// buffer is charged to its last kernel (`lfm2-chat`'s reading, #727).
    pub gpu_by_label: Vec<(String, u64, f64, f64)>,
}

impl ProfileReport {
    pub fn json(&self) -> Value {
        json!({
            "dispatches": self.dispatches,
            "command_buffers": self.command_buffers,
            "encoders": self.encoders,
            "gpu_busy_ms": self.gpu_busy_ms,
            "gpu_span_ms": self.gpu_span_ms,
            "by_label": self.by_label.iter().map(|(k, n)| json!([k, n])).collect::<Vec<_>>(),
            "gpu_by_label": self
                .gpu_by_label
                .iter()
                .map(|(k, b, t, p)| json!([k, b, t, p]))
                .collect::<Vec<_>>(),
        })
    }
}

/// Zero the dispatch recorder before a timed bracket (a no-op unless
/// `CANDLE_METAL_PROFILE=1`).
pub fn profile_reset() {
    #[cfg(feature = "metal")]
    if lloom_tensor::kernels::metal::profile::enabled() {
        lloom_tensor::kernels::metal::profile::reset();
    }
}

/// The recorder's snapshot after a bracket the caller has synchronised
/// (every completion handler has run), or `None` when it is off.
pub fn profile_take() -> Option<ProfileReport> {
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::profile;
        if profile::enabled() {
            let s = profile::snapshot();
            return Some(ProfileReport {
                dispatches: s.dispatches,
                command_buffers: s.command_buffers,
                encoders: s.encoders,
                gpu_busy_ms: 1e3 * s.gpu_busy_union_s,
                gpu_span_ms: 1e3 * s.gpu_span_s,
                by_label: s.by_label,
                gpu_by_label: profile::gpu_time_by_label_detailed()
                    .into_iter()
                    .map(|k| (k.label, k.buffers, 1e3 * k.total_s, 1e3 * k.p50_s))
                    .collect(),
            });
        }
    }
    None
}

/// One reading of the process's memory: the device's allocated bytes, the
/// pool's live / free / pending (CPU-released, awaiting the GPU — §9.5o's
/// class) bytes, and the process's `phys_footprint`, against Metal's
/// `recommendedMaxWorkingSetSize`. Every field is 0 off Metal.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemSnapshot {
    pub device_allocated: u64,
    pub pool_live: u64,
    pub pool_free: u64,
    pub pool_pending: u64,
    pub phys_footprint: u64,
    pub working_set: u64,
}

impl MemSnapshot {
    pub fn read(device: &Device) -> Self {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            let (shared, private) = m.pool_occupancy();
            return Self {
                device_allocated: m.device().current_allocated_size() as u64,
                pool_live: (shared.live_bytes + private.live_bytes) as u64,
                pool_free: (shared.free_bytes + private.free_bytes) as u64,
                pool_pending: (shared.pending_bytes + private.pending_bytes) as u64,
                phys_footprint: proc_footprint().map(|(p, _)| p).unwrap_or(0),
                working_set: m.device().recommended_max_working_set_size() as u64,
            };
        }
        let _ = device;
        Self::default()
    }

    pub fn json(&self) -> Value {
        json!({
            "device_allocated_bytes": self.device_allocated,
            "pool_live_bytes": self.pool_live,
            "pool_free_bytes": self.pool_free,
            "pool_pending_bytes": self.pool_pending,
            "phys_footprint_bytes": self.phys_footprint,
            "recommended_max_working_set_bytes": self.working_set,
        })
    }
}

/// What a cell's memory gauge reports: the readings before and after, and —
/// under `--mem-sample` — the peaks over the cell.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemReport {
    pub before: MemSnapshot,
    pub after: MemSnapshot,
    /// Peaks over the cell; `None` when the sampler was off.
    pub peak: Option<MemSnapshot>,
    pub samples: u64,
}

impl MemReport {
    pub fn json(&self) -> Value {
        let mut v = json!({
            "before": self.before.json(),
            "after": self.after.json(),
            "sampled": self.peak.is_some(),
            "samples": self.samples,
        });
        if let Some(p) = &self.peak {
            v["peak"] = p.json();
        }
        v
    }

    /// One line for the log: the peak (or the after) device bytes and the
    /// footprint, and what the cell left behind.
    pub fn line(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        match &self.peak {
            Some(p) => format!(
                "mem: device {:.2} -> peak {:.2} -> {:.2} GB, pending peak {:.2} GB, phys peak {:.2} GB ({} samples)",
                gb(self.before.device_allocated),
                gb(p.device_allocated),
                gb(self.after.device_allocated),
                gb(p.pool_pending),
                gb(p.phys_footprint),
                self.samples
            ),
            None => format!(
                "mem: device {:.2} -> {:.2} GB, phys {:.2} -> {:.2} GB",
                gb(self.before.device_allocated),
                gb(self.after.device_allocated),
                gb(self.before.phys_footprint),
                gb(self.after.phys_footprint)
            ),
        }
    }
}

/// The 10 ms sampler's stop flag and thread, joined for `(peak, samples)`.
type PeakSampler = (Arc<AtomicBool>, std::thread::JoinHandle<(MemSnapshot, u64)>);

/// The gauge around one cell: a reading before, a reading after, and under
/// `--mem-sample` a 10 ms thread holding the peaks in between.
pub struct MemGauge {
    before: MemSnapshot,
    sampler: Option<PeakSampler>,
}

impl MemGauge {
    pub fn start(device: &Device, sample: bool) -> Self {
        let before = MemSnapshot::read(device);
        let sampler = if sample && !device.is_cpu() {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let device = device.clone();
            let handle = std::thread::spawn(move || {
                let mut peak = MemSnapshot::default();
                let mut samples = 0u64;
                loop {
                    let s = MemSnapshot::read(&device);
                    peak.device_allocated = peak.device_allocated.max(s.device_allocated);
                    peak.pool_live = peak.pool_live.max(s.pool_live);
                    peak.pool_free = peak.pool_free.max(s.pool_free);
                    peak.pool_pending = peak.pool_pending.max(s.pool_pending);
                    peak.phys_footprint = peak.phys_footprint.max(s.phys_footprint);
                    peak.working_set = s.working_set;
                    samples += 1;
                    if flag.load(Ordering::Acquire) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                (peak, samples)
            });
            Some((stop, handle))
        } else {
            None
        };
        Self { before, sampler }
    }

    pub fn finish(self, device: &Device) -> MemReport {
        let (peak, samples) = match self.sampler {
            Some((stop, handle)) => {
                stop.store(true, Ordering::Release);
                match handle.join() {
                    Ok((p, n)) => (Some(p), n),
                    Err(_) => (None, 0),
                }
            }
            None => (None, 0),
        };
        MemReport {
            before: self.before,
            after: MemSnapshot::read(device),
            peak,
            samples,
        }
    }
}

/// The process's `phys_footprint` and resident size, from `proc_pid_rusage`
/// (`RUSAGE_INFO_V6`; the two words at their struct offsets).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn proc_footprint() -> Option<(u64, u64)> {
    const RUSAGE_INFO_V6: i32 = 6;
    const W_RESIDENT: usize = 8;
    const W_PHYS: usize = 9;
    const WORDS: usize = 64;
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    let mut buf = [0u64; WORDS];
    // SAFETY: `proc_pid_rusage` writes at most `sizeof(rusage_info_v6)` bytes
    // (~472) into the buffer, and `WORDS * 8 = 512` is larger, so the write is
    // in bounds; the buffer outlives the call.
    let rc = unsafe {
        proc_pid_rusage(
            std::process::id() as i32,
            RUSAGE_INFO_V6,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    if rc != 0 {
        return None;
    }
    Some((buf[W_PHYS], buf[W_RESIDENT]))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[allow(dead_code)]
fn proc_footprint() -> Option<(u64, u64)> {
    None
}

pub fn open_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return Ok(d);
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}

/// The newest snapshot of `repo` in the HF cache, if any.
pub fn newest_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// `models--Org--Name/snapshots/<hash>` → `Org/Name`; a plain directory → its
/// name.
pub fn model_name_from_dir(dir: &Path) -> String {
    let parts: Vec<&str> = dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    for p in parts.iter().rev() {
        if let Some(rest) = p.strip_prefix("models--") {
            return rest.replacen("--", "/", 1);
        }
    }
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("lloom")
        .to_string()
}

pub fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
