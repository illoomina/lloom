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
use std::sync::Mutex;
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
    /// buffer is charged to its last kernel.
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

pub use lloom::nn::memory::{MemGauge, MemReport, MemSnapshot};

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

/// What the EOS gate did on a run, for a report line: `fired` (an EOS was
/// seen and the forwards queued past it wrote nothing), `armed` (it rode the
/// run, no EOS), `off` (the run was not adaptive, or the gate is off).
pub fn eos_gate_name(report: Option<lloom_tensor::EosGateReport>) -> &'static str {
    match report {
        Some(g) if g.fired => "fired",
        Some(_) => "armed",
        None => "off",
    }
}

/// A run's host fill for a report line: `fills/forwards`, or `off` when the
/// run took none (`DecodeStats::host_feed`).
pub fn host_fill_name(report: Option<&lloom::nn::HostFeedReport>) -> String {
    match report {
        Some(f) => format!("{}/{}", f.fills, f.forwards),
        None => "off".to_string(),
    }
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Install the executor arms on `device` (Metal; nothing elsewhere) before
/// the model loads: the packed param style and the constants pool, the ICB
/// replay executor, the arena's recording, the pass-plan cache.
/// `speculative` refuses `replay` (a round's dispatch structure moves with
/// the accepted length, §11.3x) and skips the arena (served by ordinal).
/// Returns what was armed, for the engagement reports (`report.rs`).
pub fn install_executor(
    device: &Device,
    cfg: &crate::ExecutorConfig,
    speculative: bool,
) -> Result<super::report::Installed> {
    if cfg.replay && speculative {
        anyhow::bail!(
            "executor.replay cannot be combined with a speculative body: the round's dispatch \
             structure is not stable across rounds, so the executor would re-record every few \
             rounds and replay nothing (DESIGN.md §11.3x)"
        );
    }
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::ExecutorSlot;
        use lloom_tensor::kernels::{
            set_constants_pool_enabled, set_default_param_style, BarrierScope, IcbExecutor,
            ParamStyle, ReplayBarriers,
        };
        use lloom_tensor::metal_backend::ArenaLayout;
        let Device::Metal(dev) = device else {
            return Ok(super::report::Installed::default());
        };
        let mut installed = super::report::Installed {
            pass_widths: "lru",
            ..Default::default()
        };
        if cfg.packed || cfg.replay {
            set_default_param_style(ParamStyle::Packed);
            set_constants_pool_enabled(true);
            installed.packed = true;
        }
        if cfg.replay {
            lloom_tensor::kernels::set_pipelines_support_icb(true)
                .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
        }
        let layout = match cfg.arena_layout.as_str() {
            "non-aliasing" => ArenaLayout::NonAliasing,
            _ => ArenaLayout::Packed,
        };
        // One step skipped (the first token after the prefill allocates a
        // different set), two recorded, installed at the end of the second.
        const ARENA_SKIP: usize = 1;
        const ARENA_RECORD_STEPS: usize = 2;
        let arena = (cfg.arena || cfg.replay) && !speculative;
        if arena {
            dev.arm_arena_recording(layout, ARENA_SKIP, ARENA_RECORD_STEPS);
            installed.arena = true;
        }
        if cfg.replay {
            const ICB_RECORD_STEPS: usize = 3;
            let scope = match cfg.replay_barrier_scope.as_str() {
                "since-barrier" => BarrierScope::SinceBarrier,
                _ => BarrierScope::RunStart,
            };
            let barriers = match cfg.replay_barriers.as_str() {
                "always" => ReplayBarriers::Always,
                _ => ReplayBarriers::SkipReplayed,
            };
            let exec = IcbExecutor::configured(ICB_RECORD_STEPS, scope, barriers);
            if arena {
                exec.set_warm_in(1 + ARENA_SKIP + ARENA_RECORD_STEPS + 1);
                exec.set_rerecord_warm_in(ARENA_RECORD_STEPS + 1);
            }
            dev.set_executor(std::sync::Arc::new(ExecutorSlot::Custom(exec.clone())));
            installed.replay = Some(exec);
        }
        if cfg.pass_plans {
            dev.enable_pass_plans(ArenaLayout::Packed);
            installed.pass_plans = true;
        }
        Ok(installed)
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (device, cfg);
        Ok(super::report::Installed::default())
    }
}
