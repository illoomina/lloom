//! What every served family shares: the text side (tokenizer, template, the
//! ids requests are shaped by), the cells' timing records, the memory gauges,
//! the profile snapshot, the device, the HF-cache lookups. Moved out of the
//! LFM2 engine (`lfm2.rs`, the former `lloom-serve/engine.rs`) unchanged so a
//! second family (`qwen4exp.rs`) reuses them rather than restating them.

use super::template::{Template, ThinkTags};
use crate::ReadbackArm;
use anyhow::Result;
use lloom_tensor::Device;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tokenizers::Tokenizer;

/// The EOS id set that never fires: every arm's EOS mechanism still runs and
/// simply never matches, so a run produces exactly `max_tokens` tokens.
pub const NEVER_EOS: [u32; 1] = [u32::MAX];

/// Check all context positions before constructing prompts, caches or work.
/// `extra` is the separately counted first sample in throughput cells; callers
/// must not add it to generation before entering this checked boundary.
pub fn check_context_capacity(
    capacity: usize,
    prompt: usize,
    generation: usize,
    extra: usize,
) -> Result<()> {
    let generated = generation.checked_add(extra).ok_or_else(|| {
        anyhow::anyhow!("generation {generation} + extra {extra} positions overflows usize")
    })?;
    let need = prompt.checked_add(generated).ok_or_else(|| {
        anyhow::anyhow!(
            "prompt {prompt} + generation {generation} + extra {extra} positions overflows usize"
        )
    })?;
    if need > capacity {
        anyhow::bail!(
            "prompt {prompt} + generation {generation} + extra {extra} = {need} positions exceeds \
             the KV capacity {capacity} (start with --kv-capacity >= {need})"
        );
    }
    Ok(())
}

#[cfg(test)]
mod capacity_tests {
    use super::check_context_capacity;

    #[test]
    fn context_capacity_checks_each_sum_without_allocating() {
        for (capacity, prompt, generation, extra) in [
            (0, 0, 0, 0),
            (8, 5, 3, 0),
            (8, 5, 2, 1),
            (usize::MAX, usize::MAX - 2, 1, 1),
        ] {
            check_context_capacity(capacity, prompt, generation, extra).unwrap();
        }
        for (capacity, prompt, generation, extra, expected) in [
            (8, 5, 3, 1, "exceeds the KV capacity 8"),
            (0, 1, 0, 0, "exceeds the KV capacity 0"),
            (usize::MAX, usize::MAX, 1, 0, "overflows usize"),
            (usize::MAX, 0, usize::MAX, 1, "overflows usize"),
            (usize::MAX, 1, usize::MAX - 1, 1, "overflows usize"),
        ] {
            let error = check_context_capacity(capacity, prompt, generation, extra).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}

/// Text: the tokenizer, the checkpoint's own chat template, and the ids the
/// requests are shaped by.
pub struct Text {
    pub tokenizer: Tokenizer,
    pub template: Option<Template>,
    pub bos_token: String,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    /// The benchmark prompt seed, encoded without special tokens and tiled to
    /// the requested prefill length (the MLX sidecar's `make_tokens`).
    pub seed_tokens: Vec<u32>,
    /// The template's reasoning tags, read on first use ([`Self::think_tags`]).
    pub think: std::sync::OnceLock<Option<ThinkTags>>,
    /// The generation prompt's ids per `enable_thinking` argument × the
    /// engine's thinking switch (each unset, on, off), read at load
    /// ([`Self::prime`], [`Self::generation_prompt`]).
    pub gen_prompts: std::sync::OnceLock<[[Option<Vec<u32>>; 3]; 3]>,
}

impl Text {
    /// Read what the template's load-time analysis gives — the reasoning
    /// tags and the generation prompts — so no request renders a probe.
    pub fn prime(&self) {
        let _ = self.think_tags();
        let _ = self.generation_prompt(None, None);
    }

    /// The ids a prompt ends with when rendered with the template argument
    /// `enable_thinking` = `kwarg` and passed through the engine's thinking
    /// switch `want` (`thinking::apply`, which closes a block the template
    /// left open): the template's generation prompt, with the end marker
    /// the engine appends. `None` without a template, or where the renders
    /// do not nest.
    pub fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<&[u32]> {
        let all = self
            .gen_prompts
            .get_or_init(|| self.probe().generation_prompts(self.think_tags()));
        all[thinking_mode(kwarg)][thinking_mode(want)].as_deref()
    }

    fn probe(&self) -> TextProbe<'_> {
        TextProbe {
            tokenizer: &self.tokenizer,
            template: self.template.as_ref(),
            bos_token: &self.bos_token,
        }
    }

    /// The family's reasoning tags, read through the same borrowed probe used
    /// by load-once owners. No tokenizer or template is copied.
    pub fn think_tags(&self) -> Option<&ThinkTags> {
        self.think
            .get_or_init(|| self.probe().think_tags())
            .as_ref()
    }
}

/// Shared load-time analysis over the caller's one tokenizer and template.
/// Incumbent Text keeps its OnceLocks; admitted owners prepare these values
/// inside their load producer, before materializing model tensors.
pub(super) struct TextProbe<'a> {
    pub tokenizer: &'a Tokenizer,
    pub template: Option<&'a Template>,
    pub bos_token: &'a str,
}

pub(super) fn thinking_mode(mode: Option<bool>) -> usize {
    match mode {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    }
}

impl TextProbe<'_> {
    pub fn generation_prompts(&self, tags: Option<&ThinkTags>) -> [[Option<Vec<u32>>; 3]; 3] {
        let modes = [None, Some(true), Some(false)];
        modes.map(|k| modes.map(|w| self.read_generation_prompt(k, w, tags)))
    }

    fn read_generation_prompt(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
        tags: Option<&ThinkTags>,
    ) -> Option<Vec<u32>> {
        let template = self.template?;
        let mut kwargs = serde_json::Map::new();
        if let Some(k) = kwarg {
            kwargs.insert(super::template::ENABLE_THINKING.into(), Value::Bool(k));
        }
        let (mut with, without) = template.generation_prompt(self.bos_token, &kwargs, |s| {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("encoding a probe render: {e}"))
        })?;
        super::thinking::apply(want, tags, &mut with).ok()?;
        Some(with[without.len()..].to_vec())
    }

    pub fn think_tags(&self) -> Option<ThinkTags> {
        let added = self.tokenizer.get_added_tokens_decoder();
        self.template?.think_tags(
            self.bos_token,
            |s| {
                self.tokenizer
                    .encode(s, false)
                    .map(|e| e.get_ids().to_vec())
                    .map_err(|e| anyhow::anyhow!("encoding a probe render: {e}"))
            },
            |id| added.contains_key(&id),
        )
    }
}

#[cfg(test)]
mod text_probe_tests {
    use super::*;
    use lloom::testing::tiny::TinyTokenizer;
    use serde_json::Map;

    fn tokenizer(tiny: &TinyTokenizer) -> Tokenizer {
        let json = lloom::testing::tiny::tokenizer::tokenizer_json(&tiny.added);
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    // Pure bounded token/template controls: three existing byte vocabularies,
    // nine modes each; no tensors, model weights, files, native or admission.
    #[test]
    fn shared_probes_preserve_family_tags_and_all_nine_cached_prompt_modes() {
        for tiny in [
            TinyTokenizer::qwen3_5(),
            TinyTokenizer::qwen4exp(),
            TinyTokenizer::lfm2(),
        ] {
            let text = Text {
                tokenizer: tokenizer(&tiny),
                template: Some(Template::compile(tiny.chat_template).unwrap()),
                bos_token: String::new(),
                bos_id: None,
                eos_ids: Vec::new(),
                seed_tokens: Vec::new(),
                think: std::sync::OnceLock::new(),
                gen_prompts: std::sync::OnceLock::new(),
            };
            text.prime();
            let tags = text
                .think_tags()
                .expect("existing tiny family writes thinking tags");
            assert_eq!(tags.start, tiny.id("<think>").unwrap());
            assert_eq!(tags.end, tiny.id("</think>").unwrap());
            assert_eq!(
                tags.tool_call,
                tiny.id("<tool_call>")
                    .or_else(|| tiny.id("<|tool_call_start|>"))
            );
            let template = text.template.as_ref().unwrap();
            let encode = |s: &str| text.tokenizer.encode(s, false).unwrap().get_ids().to_vec();
            for kwarg in [None, Some(true), Some(false)] {
                let mut kwargs = Map::new();
                if let Some(value) = kwarg {
                    kwargs.insert(
                        super::super::template::ENABLE_THINKING.into(),
                        Value::Bool(value),
                    );
                }
                // Independent direct template render, not TextProbe's table.
                let messages = json!([{ "role": "user", "content": "bounded parity user" }]);
                let without = encode(&template.render_with(&messages, "", &kwargs, false).unwrap());
                let with = encode(&template.render_with(&messages, "", &kwargs, true).unwrap());
                assert!(with.starts_with(&without));
                for want in [None, Some(true), Some(false)] {
                    let mut expected = with.clone();
                    let expected = super::super::thinking::apply(want, Some(tags), &mut expected)
                        .ok()
                        .map(|_| expected[without.len()..].to_vec());
                    assert_eq!(
                        text.generation_prompt(kwarg, want),
                        expected.as_deref(),
                        "kwarg={kwarg:?} want={want:?}"
                    );
                }
            }
            let rendered = template.renders();
            for _ in 0..2 {
                let _ = text.think_tags();
                for kwarg in [None, Some(true), Some(false)] {
                    for want in [None, Some(true), Some(false)] {
                        let _ = text.generation_prompt(kwarg, want);
                    }
                }
            }
            assert_eq!(
                template.renders(),
                rendered,
                "cached metadata rendered request-time probes"
            );
        }
        println!("SHARED_TEXT_PROBE_FAMILIES=3 MODES_PER_FAMILY=9 CACHE_RENDERS=0");
    }

    #[test]
    fn missing_or_plain_template_does_not_fabricate_thinking_metadata() {
        let tiny = TinyTokenizer::qwen3_5();
        let tokenizer = tokenizer(&tiny);
        let absent = TextProbe {
            tokenizer: &tokenizer,
            template: None,
            bos_token: "",
        };
        assert_eq!(absent.think_tags(), None);
        assert!(absent
            .generation_prompts(None)
            .iter()
            .flatten()
            .all(Option::is_none));
        let plain = Template::compile("{%- for m in messages -%}{{ m.role }}:{{ m.content }};{%- endfor -%}{%- if add_generation_prompt -%}assistant:{%- endif -%}").unwrap();
        let probe = TextProbe {
            tokenizer: &tokenizer,
            template: Some(&plain),
            bos_token: "",
        };
        assert_eq!(probe.think_tags(), None);
        let all = probe.generation_prompts(None);
        let expected = tokenizer
            .encode("assistant:", false)
            .unwrap()
            .get_ids()
            .to_vec();
        for row in all {
            assert_eq!(row[0].as_ref(), Some(&expected));
            assert_eq!(row[1], None, "plain template cannot open thinking");
            assert_eq!(row[2].as_ref(), Some(&expected));
        }
        println!("SHARED_TEXT_PROBE_ABSENT_NONE=9 PLAIN_REFUSED_ON=3");
    }
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
/// Pre-release source migration: heavy fields are Option; None is unavailable,
/// while Some(0) and Some(empty) are valid observations. JSON uses null for None.
pub struct ProfileReport {
    pub dispatches: u64,
    pub command_buffers: u64,
    pub encoders: u64,
    pub gpu_busy_ms: Option<f64>,
    pub gpu_span_ms: Option<f64>,
    pub by_label: Option<Vec<(String, u64)>>,
    /// GPU time by kernel from each command buffer's LAST dispatch label —
    /// `(label, buffers, total_ms, p50_ms)`; exact only at
    /// `CANDLE_METAL_COMPUTE_PER_BUFFER=1` (one dispatch per buffer), else a
    /// buffer is charged to its last kernel.
    pub gpu_by_label: Option<Vec<(String, u64, f64, f64)>>,
    /// Detailed recorder availability; cheap scalar counters remain usable.
    pub recorder_available: bool,
}

impl ProfileReport {
    /// Cold human report. Missing detail is not a measured zero or empty census.
    pub fn summary(&self, steps: f64) -> String {
        let steps = steps.max(1.0);
        let kernels = self
            .by_label
            .as_ref()
            .map(|labels| {
                labels
                    .iter()
                    .take(12)
                    .map(|(k, n)| format!("{k}={n}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "unavailable".into());
        format!("gpu_busy {} ms over the bracket ({} ms per step/round of {steps:.0}), span {} ms, {} dispatches ({:.1} per step), {} command buffers, {} encoders; top kernels: {}",
            profile_metric(self.gpu_busy_ms), profile_metric(self.gpu_busy_ms.map(|v| v / steps)),
            profile_metric(self.gpu_span_ms), self.dispatches, self.dispatches as f64 / steps,
            self.command_buffers, self.encoders, kernels)
    }

    pub fn json(&self) -> Value {
        json!({
            "dispatches": self.dispatches,
            "command_buffers": self.command_buffers,
            "encoders": self.encoders,
            "gpu_busy_ms": self.gpu_busy_ms,
            "gpu_span_ms": self.gpu_span_ms,
            "recorder_available": self.recorder_available,
            "by_label": self.by_label,
            "gpu_by_label": self.gpu_by_label,
        })
    }
}

/// Render only real available metrics; shared by cold profile report consumers.
pub(super) fn profile_metric(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "unavailable".into())
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
            let recorder_available = profile::recorder_enabled();
            return Some(ProfileReport {
                recorder_available,
                dispatches: s.dispatches,
                command_buffers: s.command_buffers,
                encoders: s.encoders,
                gpu_busy_ms: recorder_available.then_some(1e3 * s.gpu_busy_union_s),
                gpu_span_ms: recorder_available.then_some(1e3 * s.gpu_span_s),
                by_label: recorder_available.then_some(s.by_label),
                gpu_by_label: recorder_available.then(|| {
                    profile::gpu_time_by_label_detailed()
                        .into_iter()
                        .map(|k| (k.label, k.buffers, 1e3 * k.total_s, 1e3 * k.p50_s))
                        .collect()
                }),
            });
        }
    }
    None
}

/// The profiler's running counts and its commit attribution for `/health`,
/// cumulative since the process started (a caller differences two reads
/// around a request); `null` unless `CANDLE_METAL_PROFILE=1` and
/// `diagnostics.trace.cut_sites` are both on.
pub fn cut_sites_json() -> Value {
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::profile;
        if profile::cut_sites_enabled() {
            let s = profile::snapshot();
            let sites: Vec<Value> = profile::cut_sites()
                .into_iter()
                .map(|c| json!({ "thread": c.thread, "commits": c.commits, "frames": c.frames }))
                .collect();
            return json!({
                "command_buffers": s.command_buffers,
                "dispatches": s.dispatches,
                "encoders": s.encoders,
                "commits": sites.iter().map(|c| c["commits"].as_u64().unwrap_or(0)).sum::<u64>(),
                "sites": sites,
                "by_label": s
                    .by_label
                    .into_iter()
                    .map(|(k, n)| (k, json!(n)))
                    .collect::<serde_json::Map<String, Value>>(),
            });
        }
    }
    Value::Null
}

pub use lloom::nn::memory::{MemGauge, MemReport, MemSnapshot};

/// The label a load's machine reservation is listed under: the family and
/// the checkpoint, so `pid … qwen3_5 Qwen3.6-35B-A3B-UD-Q4_K_M` says which
/// process holds what.
pub fn machine_label(family: &str, checkpoint: &Path) -> String {
    let name = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = match checkpoint.parent() {
        // An HF-cache snapshot directory names nothing; its repo does.
        Some(_) if name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit()) => checkpoint
            .ancestors()
            .find_map(|a| {
                a.file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.strip_prefix("models--"))
                    .map(|n| n.replace("--", "/"))
            })
            .unwrap_or(name),
        _ => name,
    };
    format!("lloom {family} {name}")
}

/// Checked configured setup, before the native/fallback body. Kept separate
/// so host controls exercise ordering without constructing any native device.
fn with_checked_device_setup(body: impl FnOnce() -> Result<Device>) -> Result<Device> {
    lloom_tensor::tuning::try_tuning()?;
    body()
}

pub fn open_device() -> Result<Device> {
    with_checked_device_setup(|| {
        #[cfg(not(feature = "metal"))]
        if lloom_tensor::tuning::tuning().metal.command_api == "metal4" {
            anyhow::bail!("--metal-api metal4 requires a build with the metal feature; select metal3/auto for normal device fallback");
        }
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => return Ok(d),
                Err(error) if lloom_tensor::tuning::tuning().metal.command_api == "metal4" => {
                    // An explicit native API request must not silently become a
                    // CPU run when capability or diagnostics reject that API.
                    return Err(error.into());
                }
                Err(_) => {}
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(d) = Device::new_cuda(0) {
                return Ok(d);
            }
        }
        Ok(Device::Cpu)
    })
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
        let dev = dev.execution_device()?;
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
        if cfg.pass_plans_armed() {
            // `--arena-layout` selects the plans' layout too: `non-aliasing`
            // gives every value its own slot, so no slot is reused inside a pass.
            dev.enable_pass_plans(layout);
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

#[cfg(test)]
mod configured_device_setup_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn invalid_diagnostic_setup_never_enters_device_or_fallback_body() {
        lloom_tensor::isolation::claim("configured device setup test");
        let value = if lloom_tensor::tuning::PROFILE_INSTRUMENTS {
            "not-a-bool"
        } else {
            "1"
        };
        std::env::set_var("CANDLE_METAL_TRACE", value);
        let entered = Cell::new(false);
        let error = with_checked_device_setup(|| {
            entered.set(true);
            Ok(Device::Cpu)
        })
        .unwrap_err();
        assert!(
            !entered.get(),
            "a refused request must not enter even the fallback body"
        );
        assert!(
            !lloom_tensor::tuning::installed(),
            "failed setup must not freeze tuning"
        );
        let message = error.to_string();
        let expected = if lloom_tensor::tuning::PROFILE_INSTRUMENTS {
            "CANDLE_METAL_TRACE"
        } else {
            "diagnostics.trace.dispatch"
        };
        assert!(message.contains(expected), "{message}");
    }

    #[test]
    fn default_setup_retains_the_fallback_body() {
        lloom_tensor::tuning::install(lloom_tensor::tuning::Tuning::default()).unwrap();
        let entered = Cell::new(0);
        let device = with_checked_device_setup(|| {
            entered.set(entered.get() + 1);
            Ok(Device::Cpu)
        })
        .unwrap();
        assert!(device.is_cpu());
        assert_eq!(entered.get(), 1);
    }

    #[test]
    fn disabled_diagnostic_setup_retains_the_fallback_body() {
        lloom_tensor::isolation::claim("configured device setup test");
        std::env::set_var("CANDLE_METAL_TRACE", "0");
        let entered = Cell::new(0);
        let device = with_checked_device_setup(|| {
            entered.set(entered.get() + 1);
            Ok(Device::Cpu)
        })
        .unwrap();
        assert!(device.is_cpu());
        assert_eq!(entered.get(), 1);
        assert!(!lloom_tensor::tuning::tuning().instrument.trace);
    }
}

#[cfg(test)]
mod profile_report_tests {
    use super::*;
    fn report(detail: bool, busy: f64) -> ProfileReport {
        ProfileReport {
            dispatches: 7,
            command_buffers: 3,
            encoders: 4,
            recorder_available: detail,
            gpu_busy_ms: detail.then_some(busy),
            gpu_span_ms: detail.then_some(0.0),
            by_label: detail.then(Vec::new),
            gpu_by_label: detail.then(Vec::new),
        }
    }
    #[test]
    fn public_round_probe_validates_capability_without_native_work() {
        let actual = lloom_tensor::tuning::PROFILE_INSTRUMENTS;
        // Test-only expectation pins controlled build graphs without rejecting
        // legitimate workspace fuzzing/profile feature unification in other runs.
        if let Ok(expected) = std::env::var("LLOOM_TEST_EXPECT_PROFILE_CAPABILITY") {
            let expected = match expected.as_str() {
                "available" => true,
                "absent" => false,
                other => panic!("unknown test capability expectation: {other}"),
            };
            assert_eq!(actual, expected, "unexpected diagnostic feature union");
            println!("profile capability actual={actual} expected={expected}");
        }
        lloom::nn::RoundProbe::default().validate().unwrap();
        lloom::nn::RoundProbe {
            sync: Some(&Device::Cpu),
            census: false,
        }
        .validate()
        .unwrap();
        let error = lloom::nn::RoundProbe {
            sync: Some(&Device::Cpu),
            census: true,
        }
        .validate()
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("diagnostics.gpu_profile.enabled"));
        if !actual {
            assert!(message.contains("profile instrumentation is not compiled"));
        } else {
            assert!(message.contains("requires an active detailed Metal sync probe"));
        }
    }
    #[cfg(feature = "metal")]
    #[test]
    fn actual_host_snapshot_retains_counts_and_reports_build_availability() {
        use lloom_tensor::kernels::metal::profile;
        let mut tuning = lloom_tensor::tuning::Tuning::default();
        tuning.instrument.profile = true;
        lloom_tensor::tuning::install(tuning).unwrap();
        profile::reset();
        let label = std::sync::Arc::<str>::from("host-report-positive");
        profile::record_dispatch(Some(&label), 1);
        profile::record_encoder();
        profile::record_command_buffer(1.0, 3.0, 0.5, 1);
        let p = profile_take().expect("cheap scalar profiling remains enabled");
        assert_eq!((p.dispatches, p.encoders, p.command_buffers), (1, 1, 1));
        let available = profile::recorder_enabled();
        assert_eq!(p.recorder_available, available);
        if available {
            assert_eq!(p.gpu_busy_ms, Some(2000.0));
            assert_eq!(p.by_label, Some(vec![("host-report-positive".into(), 1)]));
        } else {
            assert!(p.gpu_busy_ms.is_none());
            assert!(p.by_label.is_none());
            assert!(p.json()["gpu_busy_ms"].is_null());
        }
    }
    #[test]
    fn unavailable_detail_is_null_and_text_not_zero_with_counts_retained() {
        let p = report(false, 0.0);
        let j = p.json();
        for key in ["gpu_busy_ms", "gpu_span_ms", "by_label", "gpu_by_label"] {
            assert!(j[key].is_null(), "{key}: {j}");
        }
        assert_eq!(j["recorder_available"], false);
        assert_eq!(j["dispatches"], 7);
        let text = p.summary(2.0);
        assert!(text.contains("gpu_busy unavailable"));
        assert!(text.contains("top kernels: unavailable"));
        assert!(text.contains("7 dispatches"));
    }
    #[test]
    fn available_zero_and_empty_are_real_values_not_missing() {
        let p = report(true, 0.0);
        let j = p.json();
        assert_eq!(j["gpu_busy_ms"], 0.0);
        assert_eq!(j["by_label"], json!([]));
        assert_eq!(j["gpu_by_label"], json!([]));
        assert_eq!(j["recorder_available"], true);
        assert!(!p.summary(2.0).contains("unavailable"));
    }
    #[test]
    fn available_nonzero_detail_survives_both_production_renderers() {
        let mut p = report(true, 12.5);
        p.by_label = Some(vec![("kernel".into(), 7)]);
        p.gpu_by_label = Some(vec![("kernel".into(), 3, 12.5, 4.0)]);
        let j = p.json();
        assert_eq!(j["gpu_busy_ms"], 12.5);
        assert_eq!(j["gpu_by_label"][0][2], 12.5);
        let text = p.summary(2.0);
        assert!(text.contains("12.5000"));
        assert!(text.contains("6.2500"));
        assert!(text.contains("kernel=7"));
    }
}
