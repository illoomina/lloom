//! The family layer: what a served model family can do, what it refuses, and
//! how it loads — the seam a third family implements (`decode-driver-design.md`
//! Stage 7, the shape proposed there).
//!
//! `detect(path) → the family → capabilities() → resolve(&Requested) →
//! load(Requested)`. The refusals live in `resolve`, once per family, and name
//! the reason (a family that cannot serve a shape refuses it, never serves it
//! wrong — `served.rs`). `Served` stays the `dyn`-per-request seam; nothing
//! here is on the token path.

use super::{lfm2, qwen3_5, qwen4exp, LoadConfig, Served, Shape};
#[allow(unused_imports)]
use anyhow::Context as _;
use anyhow::{Context, Result};
use lloom::nn::ops::{KvPrecision, KvQuant};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// What a family can do, as constants a caller and `/health` read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub family: &'static str,
    /// Token-generation support (distinct from non-generating decisions).
    pub generation: bool,
    /// An installed decision executor, not merely an HTTP route.
    pub decisions: bool,
    /// The profiles the family ships, as a bound over the eight
    /// (`lloom::profile`, design §3: "a bound, not a list"; #892 B1). The
    /// boundary (§7.4) matches `(--profile, rewind, rows)` against it.
    pub profiles: lloom::profile::Bound,
    /// Forward passes the host may enqueue past an unlearned EOS.
    pub lookahead_max: usize,
    /// The ring arms: `full` (naming any arm on which the input still reaches
    /// the host), or `by-name` (honoured, but the fed id is read back per
    /// step) with the reason.
    pub ring: &'static str,
    /// `--batch N` (the continuous scheduler).
    pub batch: bool,
    /// Prefix sharing between rows (`--share-prefix`).
    pub prefix_share: bool,
    /// The speculation methods (`speculative.method`).
    pub speculation: &'static [&'static str],
    /// The `model` group's axes are this family's `Config`'s.
    pub model_axes: bool,
    /// How far a decode step can be undone.
    pub rewind: &'static str,
    /// The executor arms (`arena`, `packed`, `replay`) are measured on this family.
    pub executor_arms: bool,
    /// The KV-cache precision arms (`model.axes.kv-quant`) this family
    /// serves. Read whatever `model_axes` says: KV precision is a mechanism
    /// of every family, not one family's `Config` field
    /// (`sequence-state.md` §10.7).
    pub kv_quant: &'static [KvQuant],
    /// What `KvQuant::Off` stores on this family.
    pub kv_dtype: &'static str,
    /// The dense stores this family builds, `kv_dtype` among them: a dtype
    /// name (`f16`) selects one of these, never a store it does not name.
    pub kv_dense: &'static [&'static str],
    /// Why the arms missing from `kv_quant` are refused.
    pub kv_quant_missing: &'static str,
    /// Image input (OpenAI `image_url` parts, `lloom chat --image`). A
    /// checkpoint of the family can still lack a tower; the engine refuses
    /// that one by its own reason.
    pub vision: bool,
    /// Why a family without vision refuses an image.
    pub vision_missing: &'static str,
    /// The EOS gate (`decode.readback.eos_gate`) holds: the family's decode
    /// kernels carry the gated variant, so a forward queued past an EOS
    /// writes nothing. Where they do not, the gate is clamped off at load.
    pub eos_gate: bool,
    /// Why a family without the gated kernels runs ungated.
    pub eos_gate_missing: &'static str,
}

impl Capabilities {
    pub fn json(&self) -> Value {
        json!({
            "family": self.family,
            "generation": self.generation,
            "decisions": self.decisions,
            "lookahead_max": self.lookahead_max,
            "ring": self.ring,
            "batch": self.batch,
            "prefix_share": self.prefix_share,
            "speculation": self.speculation,
            "model_axes": self.model_axes,
            "rewind": self.rewind,
            "executor_arms": self.executor_arms,
            "kv_quant": self.kv_quant_names(),
            "kv_dtype": self.kv_dtype,
            "vision": self.vision,
            "eos_gate": self.eos_gate,
        })
    }

    pub fn line(&self) -> String {
        format!(
            "generation={} decisions={} lookahead_max={} ring={} batch={} prefix_share={} speculation=[{}] model_axes={} kv_quant=[{}] kv_dtype={} vision={} eos_gate={} rewind={:?}",
            self.generation,
            self.decisions,
            self.lookahead_max,
            self.ring,
            self.batch,
            self.prefix_share,
            self.speculation.join(","),
            self.model_axes,
            self.kv_quant_names().join(","),
            self.kv_dtype,
            self.vision,
            self.eos_gate,
            self.rewind,
        )
    }

    /// The KV arms by the names `--kv-quant` takes: `off` (the `kv_dtype`
    /// store), each other dense store by its dtype, then the quantized arms.
    fn kv_quant_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for q in self.kv_quant {
            names.push(q.name().to_ascii_lowercase());
            if *q == KvQuant::Off {
                names.extend(
                    self.kv_dense
                        .iter()
                        .filter(|d| **d != self.kv_dtype)
                        .map(|d| d.to_string()),
                );
            }
        }
        names
    }
}

/// The `model.axes` key of the KV precision axis (`lfm2::Config`'s `#[axis]`
/// flag, which `ModelConfig::set` keys the map by).
const KV_QUANT_AXIS: &str = "kv-quant";

/// What a run asked for: the family's own load flags and the option groups.
pub struct Requested<'a> {
    pub model: &'a LoadConfig,
    pub shape: Shape,
}

/// A served model family.
pub trait Family: Send + Sync {
    fn name(&self) -> &'static str;
    /// Whether `path` (a file or a directory) is this family's checkpoint, and
    /// the path the loader takes (the first GGUF shard, the directory).
    fn detect(&self, path: &Path) -> Option<PathBuf>;
    fn capabilities(&self) -> Capabilities;
    /// The refusals, with their reasons: every shape this family cannot serve.
    fn resolve(&self, req: &Requested<'_>) -> Result<()>;
    /// Load the engine for a resolved request.
    fn load(&self, checkpoint: PathBuf, req: Requested<'_>) -> Result<Box<dyn Served>>;
}

pub struct Lfm2Family;
pub struct Qwen4ExpFamily;
pub struct Qwen3_5Family;

pub const LFM2_CAPABILITIES: Capabilities = Capabilities {
    family: "lfm2",
    generation: true,
    decisions: false, // Transitional until retained-trunk execution is installed.
    profiles: lloom::profile::LFM2,
    lookahead_max: 1,
    ring: "full",
    batch: true,
    prefix_share: true,
    speculation: &["dspark"],
    model_axes: true,
    rewind: "the rotating ring's K on the rewindable arms; none on the crown (Shuffle)",
    executor_arms: true,
    kv_quant: &[KvQuant::Off, KvQuant::Int8, KvQuant::Int4],
    kv_dtype: "f16",
    kv_dense: &["f16"],
    kv_quant_missing: "",
    vision: true,
    vision_missing: "",
    eos_gate: true,
    eos_gate_missing: "",
};

pub const QWEN4EXP_CAPABILITIES: Capabilities = Capabilities {
    family: "qwen4exp",
    generation: true,
    decisions: false, // Transitional until retained-trunk execution is installed.
    profiles: lloom::profile::QWEN4EXP,
    lookahead_max: 1,
    ring: "full: the PLE fed by the driver's host fill where the arm stamps each token (Metal); the fed id read back on a deferred read at the fixed cadence",
    batch: true,
    prefix_share: false,
    speculation: &["mtp", "dflash"],
    model_axes: false,
    rewind: "one decode step (the standing window)",
    executor_arms: false,
    kv_quant: &[KvQuant::Off, KvQuant::Int8, KvQuant::Int4],
    kv_dtype: "f32",
    kv_dense: &["f32", "f16"],
    kv_quant_missing: "",
    vision: false,
    vision_missing: "no vision tower is built for Qwen4-exp (its image input is #765)",
    eos_gate: false,
    eos_gate_missing: "its decode kernels (the quantized cores, qwen4exp.metal) carry no gated \
        variant, so a forward queued past an EOS still runs most of its pass, and a request \
        served after gated runs came back wrong (expert-streaming.md §20.13c)",
};

pub const QWEN3_5_CAPABILITIES: Capabilities = Capabilities {
    family: "qwen3_5",
    generation: true,
    decisions: false, // Transitional until retained-trunk execution is installed.
    profiles: lloom::profile::QWEN3_5,
    lookahead_max: 1,
    ring: "by-name: the decoder keeps no standing window, so a forward past an unlearned EOS is not undone",
    batch: false,
    prefix_share: false,
    speculation: &["mtp"],
    model_axes: false,
    rewind: "a verify window over every layer's state (speculative.method = mtp, the Ring shape); no standing window; turn checkpoints (a request re-rendering the previous turn restores where its prefill left one)",
    executor_arms: false,
    kv_quant: &[KvQuant::Off],
    kv_dtype: "f32",
    kv_dense: &["f32"],
    kv_quant_missing: "the reduced stores are built on the attention block but their gates were not run on this family",
    vision: false,
    vision_missing: "no vision tower is built for Qwen3.5 (the text path only; its tower is #765's lineage)",
    eos_gate: false,
    eos_gate_missing: "its decode kernels (the GDN and attention kernels, the quantized cores) carry no gated variant",
};

/// Every family, in detection order: the GGUF families read the header's
/// `general.architecture`, so their order does not matter; LFM2 is a
/// directory with a `config.json`.
pub static FAMILIES: [&dyn Family; 3] = [&Lfm2Family, &Qwen3_5Family, &Qwen4ExpFamily];

/// The GGUF checkpoint `path` names — a shard, or a directory of shards —
/// as the first shard, with its header's `general.architecture`.
fn gguf_checkpoint(path: &Path) -> Option<(PathBuf, String)> {
    let first = if path.is_file() {
        (path.extension().and_then(|e| e.to_str()) == Some("gguf")).then(|| path.to_path_buf())?
    } else {
        if !path.is_dir() || path.join("config.json").is_file() {
            return None;
        }
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(path)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
            .collect();
        ggufs.sort();
        // The first shard of a sharded checkpoint, else the only file.
        ggufs
            .iter()
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("-00001-of-"))
                    .unwrap_or(false)
            })
            .or_else(|| (ggufs.len() == 1).then(|| &ggufs[0]))
            .cloned()?
    };
    // A file whose header does not read (an empty placeholder in a test)
    // names no architecture; the load reports what is wrong with it.
    let arch = std::fs::File::open(&first)
        .ok()
        .and_then(|mut f| lloom_tensor::quantized::gguf_file::Content::read(&mut f).ok())
        .and_then(|c| {
            c.metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .cloned()
        })
        .unwrap_or_default();
    Some((first, arch))
}

/// The family for `path`, and the checkpoint path its loader takes.
pub fn detect(path: &Path) -> Result<(&'static dyn Family, PathBuf)> {
    for f in FAMILIES.iter() {
        if let Some(p) = f.detect(path) {
            return Ok((*f, p));
        }
    }
    if !path.exists() {
        anyhow::bail!("{}: no such model", path.display());
    }
    anyhow::bail!(
        "{}: neither a checkpoint directory (config.json — LFM2) nor a GGUF shard or a directory of shards (Qwen4-exp, Qwen3.5)",
        path.display()
    )
}

/// The KV precision arm requested, refused unless `cap` serves it. `f16`
/// names the f16 dense store, served where the family builds one (`Off`
/// itself on LFM2, a second dense store on Qwen4-exp) and refused where it
/// does not, so a name never selects a store it does not describe.
fn resolve_kv_quant(
    cap: &Capabilities,
    axes: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    let Some(value) = axes.get(KV_QUANT_AXIS) else {
        return Ok(());
    };
    let arm =
        KvQuant::parse(value).map_err(|e| anyhow::anyhow!("model.axes.{KV_QUANT_AXIS}: {e}"))?;
    let arms = cap.kv_quant_names().join(", ");
    if !cap.kv_quant.contains(&arm) {
        anyhow::bail!(
            "kv-quant = {value} is not {}'s (its KV arms: [{arms}]){}{}",
            cap.family,
            if cap.kv_quant_missing.is_empty() {
                ""
            } else {
                ": "
            },
            cap.kv_quant_missing
        );
    }
    if value == "f16" && !cap.kv_dense.contains(&"f16") {
        anyhow::bail!(
            "kv-quant = f16 names an f16 KV store; {} builds none (its `off` stores {}; its KV arms: [{arms}])",
            cap.family,
            cap.kv_dtype,
        );
    }
    Ok(())
}

/// Qwen4-exp's KV store for the requested `kv-quant`, `None` when the axis
/// is not set (the checkpoint's f32). The value is the family's own parse
/// (`qwen4exp::config::parse_kv_precision`): `off` is f32, `f16` the f16
/// dense store, `int8` / `int4` the codes.
fn qwen_kv_precision(
    axes: &std::collections::BTreeMap<String, String>,
) -> Result<Option<KvPrecision>> {
    axes.get(KV_QUANT_AXIS)
        .map(|v| {
            lloom::models::qwen4exp::config::parse_kv_precision(v)
                .map_err(|e| anyhow::anyhow!("model.axes.{KV_QUANT_AXIS}: {e}"))
        })
        .transpose()
}

/// The `fusion` axis written by the merged key `kernels.fusion.policy`
/// (which sets every family's policy at once): its value is also the
/// process's Qwen4-exp policy. The LFM2-only spellings set the axis alone.
fn merged_fusion(axis: &str, value: &str) -> bool {
    axis == "fusion" && lloom::tuning::tuning().qwen4exp.fusion == value
}

/// The first of Qwen4-exp's own keys (`models.qwen4exp.*`) that `model`
/// sets off its default, for a family that has no such arm to refuse by
/// name (decode-driver-design.md Stage 7: a family refuses another's key).
fn qwen4exp_key_set(model: &LoadConfig) -> Option<&'static str> {
    let d = LoadConfig::default();
    [
        (model.ple != d.ple, "models.qwen4exp.ple.enabled"),
        (
            model.ple_cache_rows != d.ple_cache_rows,
            "models.qwen4exp.ple.cache_rows",
        ),
        (
            model.moe_act != d.moe_act,
            "models.qwen4exp.moe.activation_dtype",
        ),
        (
            model.prefill_mode != d.prefill_mode,
            "models.qwen4exp.prefill_mode",
        ),
    ]
    .into_iter()
    .find_map(|(set, key)| set.then_some(key))
}

/// The refusals every family shares.
fn resolve_common(cap: &Capabilities, req: &Requested<'_>) -> Result<()> {
    let s = &req.shape;
    if !cap.speculation.contains(&s.speculative.method.as_str())
        && !matches!(s.speculative.method.as_str(), "none" | "auto")
    {
        anyhow::bail!(
            "speculative.method = {} is not {}'s (its methods: [{}])",
            s.speculative.method,
            cap.family,
            cap.speculation.join(", ")
        );
    }
    resolve_kv_quant(cap, &s.model.axes)?;
    let config_axes: Vec<&str> = s
        .model
        .axes
        .iter()
        .filter(|(k, v)| *k != KV_QUANT_AXIS && !merged_fusion(k, v))
        .map(|(k, _)| k.as_str())
        .collect();
    if !cap.model_axes && (!config_axes.is_empty() || s.model.cache != "auto") {
        anyhow::bail!(
            "the model axes ({}{}) are LFM2's `Config`'s; {}'s arms are its load flags and the `tuning.{}.*` knobs",
            if s.model.cache != "auto" { format!("cache={} ", s.model.cache) } else { String::new() },
            config_axes.join(", "),
            cap.family,
            cap.family,
        );
    }
    if !cap.batch && s.batch > 1 {
        anyhow::bail!("{} does not serve --batch {}", cap.family, s.batch);
    }
    if !cap.prefix_share && s.share_prefix {
        anyhow::bail!(
            "{} has no prefix sharing between rows (its recurrent channels have no page to share)",
            cap.family
        );
    }
    if !cap.executor_arms && (s.executor.arena || s.executor.packed || s.executor.replay) {
        anyhow::bail!(
            "executor.{{arena, packed, replay}} are measured on LFM2 only; on {} the executor arm reachable is pass_plans",
            cap.family
        );
    }
    Ok(())
}

impl Family for Lfm2Family {
    fn name(&self) -> &'static str {
        "lfm2"
    }
    fn detect(&self, path: &Path) -> Option<PathBuf> {
        (path.is_dir() && path.join("config.json").is_file()).then(|| path.to_path_buf())
    }
    fn capabilities(&self) -> Capabilities {
        LFM2_CAPABILITIES
    }
    fn resolve(&self, req: &Requested<'_>) -> Result<()> {
        resolve_common(&LFM2_CAPABILITIES, req)?;
        let s = &req.shape;
        lfm2::RowsArm::parse(&s.rows, s.batch > 1)?;
        if s.share_prefix && !(s.batch > 1 && lfm2::RowsArm::parse(&s.rows, true)?.pooled()) {
            anyhow::bail!("--share-prefix needs `--batch N --rows pool` (the rows share one pool)");
        }
        Ok(())
    }
    fn load(&self, dir: PathBuf, req: Requested<'_>) -> Result<Box<dyn Served>> {
        let (model, shape) = (req.model, req.shape);
        let rows = lfm2::RowsArm::parse(&shape.rows, shape.batch > 1)?;
        let spec = &shape.speculative;
        lfm2::load(lfm2::LoadOptions {
            model_dir: dir,
            cpu: model.cpu(),
            kv_capacity: shape.memory.kv_capacity,
            readback: shape.decode.readback.arm(),
            decode: shape.decode.clone(),
            model: shape.model.clone(),
            reuse_prefix: shape.reuse_prefix,
            warmup: model.warmup,
            seed_text: shape.seed_text.clone(),
            model_name: model.model_name.clone(),
            batch: shape.batch,
            dspark: (spec.method == "dspark").then(|| spec.drafter.clone().unwrap_or_default()),
            prefill_tile: shape.decode.prefill.tile,
            prefill_chunk: shape.decode.prefill.chunk,
            memory_fraction: shape.memory.memory_fraction,
            mem_sample: shape.mem_sample,
            rows,
            share_prefix: shape.share_prefix,
            dspark_k: spec.k,
            executor: shape.executor.clone(),
            memory: shape.memory.clone(),
            flash_k: model.flash_k,
        })
    }
}

impl Family for Qwen4ExpFamily {
    fn name(&self) -> &'static str {
        "qwen4exp"
    }
    fn detect(&self, path: &Path) -> Option<PathBuf> {
        let (first, arch) = gguf_checkpoint(path)?;
        // A GGUF whose header names another family is theirs; one with no
        // header (an empty placeholder in a test) is Qwen4-exp's, as before.
        (arch.is_empty() || arch == "qwen4exp").then_some(first)
    }
    fn capabilities(&self) -> Capabilities {
        QWEN4EXP_CAPABILITIES
    }
    fn resolve(&self, req: &Requested<'_>) -> Result<()> {
        resolve_common(&QWEN4EXP_CAPABILITIES, req)?;
        let s = &req.shape;
        if s.speculative.method == "mtp" {
            if s.speculative.drafter.is_none() {
                anyhow::bail!("speculative.method = mtp needs the MTP sidecar GGUF (`--mtp <path>` / `speculative.drafter`)");
            }
            if s.batch > 1 {
                anyhow::bail!(
                    "--batch {} with --mtp on Qwen4-exp: the batched loop is the K = 1 driver; the MTP head speculates at --batch 1",
                    s.batch
                );
            }
        }
        if s.speculative.method == "dflash" && s.batch > 1 {
            anyhow::bail!(
                "--batch {} with speculative.method = dflash on Qwen4-exp: the batched loop is the K = 1 driver; the DFlash drafter speculates at --batch 1",
                s.batch
            );
        }
        if req.model.flash_k.is_some() {
            anyhow::bail!("--flash-k sizes LFM2's FlashDecoding chunk; qwen4exp has no such arm");
        }
        match req.model.experts.as_str() {
            "stream" | "resident" => {}
            other => anyhow::bail!("--experts {other}: stream | resident"),
        }
        match req.model.prefill_mode.as_str() {
            "chunk" | "replay" => {}
            other => anyhow::bail!("--prefill-mode {other}: chunk | replay"),
        }
        if let Some(p) = qwen_kv_precision(&s.model.axes)? {
            if p != lloom::models::qwen4exp::config::DEFAULT_KV_PRECISION && s.batch > 1 {
                anyhow::bail!(
                    "kv-quant = {p:?} with --batch {}: the rows' paged KV pool is f32 (its read is \
                     the f32 SdpaVectorPaged); the reduced stores are the single sequence's",
                    s.batch
                );
            }
            if p.quant().is_on() && req.model.cpu() {
                anyhow::bail!(
                    "kv-quant = {p:?} is Metal-only (the quantized append and reads are Metal kernels)"
                );
            }
        }
        Ok(())
    }
    fn load(&self, first: PathBuf, req: Requested<'_>) -> Result<Box<dyn Served>> {
        let (model, shape) = (req.model, req.shape);
        let spec = &shape.speculative;
        qwen4exp::load(qwen4exp::LoadOptions {
            first_shard: first,
            tokenizer: model.tokenizer.clone(),
            cpu: model.cpu(),
            kv_capacity: shape.memory.kv_capacity,
            readback: shape.decode.readback.arm(),
            decode: shape.decode.clone(),
            reuse_prefix: shape.reuse_prefix,
            warmup: model.warmup,
            seed_text: shape.seed_text.clone(),
            model_name: model.model_name.clone(),
            prefill_tile: shape.decode.prefill.tile,
            memory_fraction: shape.memory.memory_fraction,
            mem_sample: shape.mem_sample,
            experts: model.experts.clone(),
            slots: shape.memory.slots.clone(),
            loader: model.loader.clone(),
            ple_cache_rows: model.ple_cache_rows,
            ple: model.ple,
            moe_act: model.moe_act.clone(),
            mtp: (spec.method == "mtp")
                .then(|| spec.drafter.clone())
                .flatten(),
            dflash: (spec.method == "dflash").then(|| spec.drafter.clone().unwrap_or_default()),
            spec_k: spec.k,
            hotlist: model.hotlist.clone(),
            batch: shape.batch,
            mtp_k: spec.k.unwrap_or(1),
            prefetch: model.prefetch,
            io_hints: model.io_hints,
            prefill_mode: model.prefill_mode.clone(),
            pin: model.pin,
            kv_reserve_tokens: shape.memory.kv_reserve_tokens,
            executor: shape.executor.clone(),
            memory: shape.memory.clone(),
            kv_precision: qwen_kv_precision(&shape.model.axes)?,
        })
    }
}

impl Family for Qwen3_5Family {
    fn name(&self) -> &'static str {
        "qwen3_5"
    }
    fn detect(&self, path: &Path) -> Option<PathBuf> {
        let (first, arch) = gguf_checkpoint(path)?;
        (arch == lloom::models::qwen3_5::ARCH_DENSE
            || arch == lloom::models::qwen3_5::ARCH_MOE
            || arch == lloom::models::qwen3_5::ARCH_CLEF)
            .then_some(first)
    }
    fn capabilities(&self) -> Capabilities {
        QWEN3_5_CAPABILITIES
    }
    fn resolve(&self, req: &Requested<'_>) -> Result<()> {
        resolve_common(&QWEN3_5_CAPABILITIES, req)?;
        let s = &req.shape;
        if s.speculative.method == "mtp" {
            if let Some(d) = &s.speculative.drafter {
                if !d.is_file() {
                    anyhow::bail!(
                        "speculative.model {}: the MTP sidecar GGUF is not a file",
                        d.display()
                    );
                }
            }
        }
        if req.model.flash_k.is_some() {
            anyhow::bail!("--flash-k sizes LFM2's FlashDecoding chunk; qwen3_5 has no such arm");
        }
        if let Some(key) = qwen4exp_key_set(req.model) {
            anyhow::bail!("`{key}` is Qwen4-exp's; qwen3_5 has no such arm");
        }
        match req.model.experts.as_str() {
            "stream" | "resident" => {}
            other => anyhow::bail!("--experts {other}: stream | resident"),
        }
        Ok(())
    }
    fn load(&self, first: PathBuf, req: Requested<'_>) -> Result<Box<dyn Served>> {
        let (model, shape) = (req.model, req.shape);
        qwen3_5::load(qwen3_5::LoadOptions {
            first_shard: first,
            tokenizer: model.tokenizer.clone(),
            cpu: model.cpu(),
            kv_capacity: shape.memory.kv_capacity,
            readback: shape.decode.readback.arm(),
            decode: shape.decode.clone(),
            reuse_prefix: shape.reuse_prefix,
            warmup: model.warmup,
            seed_text: shape.seed_text.clone(),
            model_name: model.model_name.clone(),
            prefill_tile: shape.decode.prefill.tile,
            memory_fraction: shape.memory.memory_fraction,
            mem_sample: shape.mem_sample,
            executor: shape.executor.clone(),
            memory: shape.memory.clone(),
            kv_precision: None,
            experts: model.experts.clone(),
            mtp: shape.speculative.method == "mtp",
            mtp_sidecar: (shape.speculative.method == "mtp")
                .then(|| shape.speculative.drafter.clone())
                .flatten(),
            mtp_k: shape.speculative.k.unwrap_or(1),
        })
    }
}

/// Turn the EOS gate off on a family whose decode kernels carry no gated
/// variant (`Capabilities::eos_gate`). The gate defaults on, so this is a
/// clamp, not a refusal; the note is returned only where the configured arm
/// would have carried the gate (a ring under the adaptive commit).
fn clamp_ungated_eos_gate(cap: &Capabilities, shape: &mut Shape) -> Option<String> {
    let readback = &mut shape.decode.readback;
    if cap.eos_gate || !readback.eos_gate {
        return None;
    }
    readback.eos_gate = false;
    let rides = readback.kind == crate::ReadbackKind::Ring && readback.ring_arm().adaptive;
    rides.then(|| {
        format!(
            "eos_gate is off on {}: {}",
            cap.family, cap.eos_gate_missing
        )
    })
}

/// `speculative.method = auto` resolved for `family`: LFM2's DSpark when
/// `speculative.drafter` names a checkpoint or the HF cache holds one;
/// Qwen4-exp's MTP when it names the head's sidecar or
/// `~/.cache/lloom/qwen4exp-mtp-q8.gguf` exists, else its DFlash drafter
/// when it names a checkpoint directory or the HF cache holds the PixelML
/// one, at `--batch 1` (either drafter speculates at one row); else
/// `none`. The note names the choice, so a run records which arm it took.
fn resolve_auto_speculation(family: &str, shape: &mut Shape) -> Option<String> {
    if shape.speculative.method != "auto" {
        return None;
    }
    let batch = shape.batch;
    let spec = &mut shape.speculative;
    let chosen = match family {
        "lfm2" => (spec.drafter.is_some()
            || super::common::newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark").is_some())
        .then_some("dspark"),
        // The block in the main file is not known here (the header is read
        // at load): `auto` takes MTP when a sidecar is named.
        "qwen3_5" => spec
            .drafter
            .as_ref()
            .is_some_and(|d| d.is_file())
            .then_some("mtp"),
        "qwen4exp" => {
            if spec.drafter.is_none() {
                spec.drafter = std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".cache/lloom/qwen4exp-mtp-q8.gguf"))
                    .filter(|p| p.is_file());
            }
            let dflash = match &spec.drafter {
                Some(d) => d.is_dir(),
                None => super::common::newest_snapshot(super::qwen4exp::DFLASH_SNAPSHOT).is_some(),
            };
            match (batch <= 1, dflash, spec.drafter.is_some()) {
                (false, ..) => None,
                (true, true, _) => Some("dflash"),
                (true, false, true) => Some("mtp"),
                (true, false, false) => None,
            }
        }
        _ => None,
    };
    spec.method = chosen.unwrap_or("none").into();
    Some(match (&spec.drafter, chosen) {
        (Some(d), Some(m)) => format!("auto -> {m} (drafter {})", d.display()),
        (_, Some(m)) => format!("auto -> {m} (the HF cache's drafter)"),
        _ => format!("auto -> none (no {family} drafter at hand, or --batch {batch})"),
    })
}

/// Detect, resolve, load — the one entry `serve` and `chat` call.
pub fn load(model: &LoadConfig, shape: Shape) -> Result<(Box<dyn Served>, PathBuf)> {
    let path = match &model.model {
        Some(p) => p.clone(),
        None => lfm2::default_model_dir()
            .context("no --model given and the LFM2.5-2.6B snapshot was not in the HF cache")?,
    };
    let (family, checkpoint) = detect(&path)?;
    let mut shape = shape;
    if let Some(note) = resolve_auto_speculation(family.name(), &mut shape) {
        eprintln!("[speculate: {note}]");
        crate::couplings::refuse_speculative(&shape.speculative.method, &shape.sampling)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let clamp = clamp_ungated_eos_gate(&family.capabilities(), &mut shape);
    if let Some(note) = &clamp {
        eprintln!("[readback: {note}]");
    }
    let req = Requested { model, shape };
    family
        .resolve(&req)
        .map_err(|e| anyhow::anyhow!("{} cannot serve this shape: {e}", family.name()))?;
    let mut engine = family.load(checkpoint.clone(), req)?;
    engine.note_load_warnings(
        clamp
            .into_iter()
            .map(|n| {
                lloom::load_warning::LoadWarning::new(
                    lloom::load_warning::LoadWarningKind::Readback,
                    n,
                )
            })
            .collect(),
    );
    Ok((engine, checkpoint))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn model_args(argv: &[&str]) -> LoadConfig {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            m: crate::options::LoadConfigArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        let mut m = LoadConfig::default();
        T::parse_from(full).m.apply(&mut m).unwrap();
        m
    }

    fn shape() -> Shape {
        Shape {
            decode: Default::default(),
            model: Default::default(),
            speculative: Default::default(),
            executor: Default::default(),
            memory: Default::default(),
            sampling: Default::default(),
            reuse_prefix: false,
            seed_text: String::new(),
            batch: 1,
            mem_sample: false,
            rows: "auto".into(),
            share_prefix: false,
        }
    }

    #[test]
    fn each_family_refuses_the_others_shapes_with_the_reason() {
        let m = model_args(&[]);
        let mut s = shape();
        s.speculative.method = "mtp".into();
        let err = Lfm2Family
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("not lfm2's"), "{err}");
        let mut s = shape();
        s.speculative.method = "dspark".into();
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("not qwen4exp's"), "{err}");
        let mut s = shape();
        s.model.axes.insert("attn".into(), "flash".into());
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("LFM2's `Config`'s"), "{err}");
        let mut s = shape();
        s.executor.replay = true;
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("pass_plans"), "{err}");
        let mut s = shape();
        s.speculative.method = "mtp".into();
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs the MTP sidecar"), "{err}");
        let mut s = shape();
        s.share_prefix = true;
        let err = Lfm2Family
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("--rows pool"), "{err}");
        // The shapes each family serves pass.
        let mut s = shape();
        s.speculative.method = "dspark".into();
        s.executor.replay = false;
        Lfm2Family
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap();
        let mut s = shape();
        s.speculative.method = "mtp".into();
        s.speculative.drafter = Some("/x.gguf".into());
        Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap();
    }

    /// `auto` resolves to the family's own drafter when one is named, to
    /// `none` at `--batch` > 1 on Qwen4-exp and for a family with none, and
    /// passes `resolve` as a method of every family.
    #[test]
    fn auto_speculation_takes_the_familys_drafter_when_one_is_at_hand() {
        let m = model_args(&[]);
        let drafter = std::env::temp_dir().join(format!("lloom-auto-{}.gguf", std::process::id()));
        std::fs::write(&drafter, b"x").unwrap();
        let auto = |batch: usize, drafter: Option<&Path>| {
            let mut s = shape();
            s.batch = batch;
            s.speculative.method = "auto".into();
            s.speculative.drafter = drafter.map(Path::to_path_buf);
            s
        };
        let mut s = auto(1, Some(&drafter));
        let note = resolve_auto_speculation("qwen4exp", &mut s).expect("a note");
        assert_eq!(s.speculative.method, "mtp", "{note}");
        let mut s = auto(2, Some(&drafter));
        resolve_auto_speculation("qwen4exp", &mut s);
        assert_eq!(
            s.speculative.method, "none",
            "the MTP head speculates at one row"
        );
        // A directory names a DFlash checkpoint on Qwen4-exp, at one row.
        let dir = std::env::temp_dir().join(format!("lloom-auto-dflash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut s = auto(1, Some(&dir));
        let note = resolve_auto_speculation("qwen4exp", &mut s).expect("a note");
        assert_eq!(s.speculative.method, "dflash", "{note}");
        Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap();
        let mut s = auto(2, Some(&dir));
        resolve_auto_speculation("qwen4exp", &mut s);
        assert_eq!(
            s.speculative.method, "none",
            "the DFlash drafter speculates at one row"
        );
        let mut s = shape();
        s.batch = 2;
        s.speculative.method = "dflash".into();
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("speculates at --batch 1"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
        let mut s = auto(1, Some(&drafter));
        resolve_auto_speculation("lfm2", &mut s);
        assert_eq!(s.speculative.method, "dspark");
        let mut s = auto(1, Some(&drafter));
        resolve_auto_speculation("gemma", &mut s);
        assert_eq!(s.speculative.method, "none");
        // A named method is left as it is.
        let mut s = shape();
        s.speculative.method = "mtp".into();
        assert!(resolve_auto_speculation("qwen4exp", &mut s).is_none());
        assert_eq!(s.speculative.method, "mtp");
        // Unresolved, `auto` is a method every family's `resolve` takes.
        Lfm2Family
            .resolve(&Requested {
                model: &m,
                shape: auto(1, None),
            })
            .unwrap();
        Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: auto(1, None),
            })
            .unwrap();
        std::fs::remove_file(&drafter).ok();
    }

    fn with_kv_quant(value: &str) -> Shape {
        let mut s = shape();
        s.model.set(KV_QUANT_AXIS, value).unwrap();
        s
    }

    #[test]
    fn each_family_serves_its_own_kv_arms_and_refuses_the_rest_by_name() {
        let m = model_args(&[]);
        let serves = |f: &'static dyn Family, s: Shape| {
            f.resolve(&Requested {
                model: &m,
                shape: s,
            })
        };
        let (qwen, lfm2): (&'static dyn Family, &'static dyn Family) =
            (&Qwen4ExpFamily, &Lfm2Family);
        for v in ["off", "f16", "int8", "8", "int4", "4"] {
            serves(qwen, with_kv_quant(v))
                .unwrap_or_else(|e| panic!("qwen4exp serves kv-quant = {v}: {e}"));
            serves(lfm2, with_kv_quant(v))
                .unwrap_or_else(|e| panic!("lfm2 serves kv-quant = {v}: {e}"));
        }
        // The rows' paged pool has only the f32 store: the reduced stores
        // are refused at --batch N by name, the shipped one is not.
        for v in ["f16", "int8", "int4"] {
            let mut s = with_kv_quant(v);
            s.batch = 2;
            let err = serves(qwen, s).unwrap_err().to_string();
            assert!(err.contains("paged KV pool is f32"), "{v}: {err}");
        }
        let mut s = with_kv_quant("off");
        s.batch = 2;
        serves(qwen, s).expect("off is the pool's own store");
        // The codes are Metal kernels; the f16 dense store is not.
        let cpu = model_args(&["--cpu"]);
        for v in ["int8", "int4"] {
            let err = Qwen4ExpFamily
                .resolve(&Requested {
                    model: &cpu,
                    shape: with_kv_quant(v),
                })
                .unwrap_err()
                .to_string();
            assert!(err.contains("Metal-only"), "{v}: {err}");
        }
        Qwen4ExpFamily
            .resolve(&Requested {
                model: &cpu,
                shape: with_kv_quant("f16"),
            })
            .expect("the f16 dense store needs no Metal kernel");
        // A family without an f16 store refuses the name rather than
        // serving another store under it.
        let no_f16 = Capabilities {
            kv_dense: &["f32"],
            ..QWEN4EXP_CAPABILITIES
        };
        let err = resolve_kv_quant(&no_f16, &with_kv_quant("f16").model.axes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("builds none (its `off` stores f32"), "{err}");
        let int_only = Capabilities {
            kv_quant: &[KvQuant::Off],
            ..QWEN4EXP_CAPABILITIES
        };
        let err = resolve_kv_quant(&int_only, &with_kv_quant("int8").model.axes)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("kv-quant = int8 is not qwen4exp's (its KV arms: [off, f16])"),
            "{err}"
        );
        // The other axes are still refused, and kv-quant is not blamed for them.
        let mut s = with_kv_quant("off");
        s.model.set("attn", "sdpa").unwrap();
        let err = serves(qwen, s).unwrap_err().to_string();
        assert!(err.contains("the model axes (attn)"), "{err}");
        assert!(LFM2_CAPABILITIES
            .line()
            .contains("kv_quant=[off,int8,int4] kv_dtype=f16"));
        assert!(QWEN4EXP_CAPABILITIES
            .line()
            .contains("kv_quant=[off,f16,int8,int4] kv_dtype=f32"));
    }

    /// What the load is handed for each spelling: the store the name
    /// describes, never the f32 store under another label.
    #[test]
    fn the_qwen_load_takes_the_store_the_name_describes() {
        use lloom_tensor::DType;
        let arm = |v: &str| qwen_kv_precision(&with_kv_quant(v).model.axes).unwrap();
        assert_eq!(qwen_kv_precision(&shape().model.axes).unwrap(), None);
        assert_eq!(arm("off"), Some(KvPrecision::dense(DType::F32)));
        assert_eq!(arm("f16"), Some(KvPrecision::dense(DType::F16)));
        assert_eq!(
            arm("int8"),
            Some(KvPrecision::quantized(KvQuant::Int8).unwrap())
        );
        assert_eq!(
            arm("8"),
            Some(KvPrecision::quantized(KvQuant::Int8).unwrap())
        );
        assert_eq!(
            arm("int4"),
            Some(KvPrecision::quantized(KvQuant::Int4).unwrap())
        );
        assert_eq!(
            arm("4"),
            Some(KvPrecision::quantized(KvQuant::Int4).unwrap())
        );
    }

    #[test]
    fn detection_is_by_checkpoint_shape() {
        let dir = std::env::temp_dir().join(format!("lloom-family-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("lfm2")).unwrap();
        std::fs::write(dir.join("lfm2/config.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.join("qwen")).unwrap();
        std::fs::write(dir.join("qwen/m-00002-of-00002.gguf"), "").unwrap();
        std::fs::write(dir.join("qwen/m-00001-of-00002.gguf"), "").unwrap();
        let (f, p) = detect(&dir.join("lfm2")).unwrap();
        assert_eq!((f.name(), p), ("lfm2", dir.join("lfm2")));
        let (f, p) = detect(&dir.join("qwen")).unwrap();
        assert_eq!(
            (f.name(), p),
            ("qwen4exp", dir.join("qwen/m-00001-of-00002.gguf"))
        );
        let (f, _) = detect(&dir.join("qwen/m-00002-of-00002.gguf")).unwrap();
        assert_eq!(f.name(), "qwen4exp");
        let err = detect(&dir.join("nope"))
            .err()
            .map(|e| e.to_string())
            .unwrap();
        assert!(err.contains("no such model"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(LFM2_CAPABILITIES.speculation, &["dspark"]);
        // One forward past an unlearned EOS on both: Qwen4-exp's standing
        // window undoes it (`rewind` above), and its PLE is fed from the host.
        assert!(QWEN4EXP_CAPABILITIES.line().contains("lookahead_max=1"));
        assert!(QWEN4EXP_CAPABILITIES.ring.starts_with("full"));
    }

    /// The gate is clamped off on a family whose kernels carry no gated
    /// variant, noted where the arm would have carried it, silent where it
    /// would not (the blocking arm, the fixed cadence); a family with the
    /// kernels keeps it.
    #[test]
    fn the_eos_gate_is_clamped_where_the_kernels_carry_no_gated_variant() {
        use crate::ReadbackConfig;
        assert_eq!(
            (LFM2_CAPABILITIES.eos_gate, QWEN4EXP_CAPABILITIES.eos_gate),
            (true, false)
        );
        assert!(QWEN4EXP_CAPABILITIES.line().contains("eos_gate=false"));
        let with = |spelling: &str| {
            let mut s = shape();
            ReadbackConfig::apply_spelling(&mut s.decode.readback, spelling).unwrap();
            assert!(s.decode.readback.eos_gate, "the gate defaults on");
            s
        };

        let mut s = with("ring-nonblocking");
        let note = clamp_ungated_eos_gate(&QWEN4EXP_CAPABILITIES, &mut s).expect("noted");
        assert!(!s.decode.readback.eos_gate);
        assert!(
            note.contains("eos_gate is off on qwen4exp") && note.contains("no gated"),
            "{note}"
        );
        for spelling in ["auto", "blocking", "ring-fixed"] {
            let mut s = with(spelling);
            assert_eq!(
                clamp_ungated_eos_gate(&QWEN4EXP_CAPABILITIES, &mut s),
                None,
                "{spelling}"
            );
            assert!(!s.decode.readback.eos_gate, "{spelling}: clamped, silently");
        }
        let mut s = with("ring-nonblocking");
        assert_eq!(clamp_ungated_eos_gate(&LFM2_CAPABILITIES, &mut s), None);
        assert!(s.decode.readback.eos_gate, "LFM2 keeps its gate");
    }

    /// `--flash-k` is LFM2's FlashDecoding chunk: Qwen4-exp refuses it by
    /// name, LFM2 resolves it (the load refuses it off `--attn flash`).
    /// qwen3_5 serves `speculative.method = mtp` (the block in the file, or
    /// the sidecar `speculative.model` names, which must be a file), and
    /// `auto` takes it only when a sidecar is named.
    #[test]
    fn qwen3_5_serves_mtp_and_auto_takes_a_named_sidecar() {
        let m = model_args(&[]);
        let mut s = shape();
        s.speculative.method = "mtp".into();
        Qwen3_5Family
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap();
        let mut s = shape();
        s.speculative.method = "mtp".into();
        s.speculative.drafter = Some("/no/such/mtp.gguf".into());
        let err = Qwen3_5Family
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a file"), "{err}");
        let mut s = shape();
        s.speculative.method = "auto".into();
        resolve_auto_speculation("qwen3_5", &mut s);
        assert_eq!(s.speculative.method, "none");
        let sidecar =
            std::env::temp_dir().join(format!("lloom-q35-mtp-{}.gguf", std::process::id()));
        std::fs::write(&sidecar, b"x").unwrap();
        let mut s = shape();
        s.speculative.method = "auto".into();
        s.speculative.drafter = Some(sidecar.clone());
        resolve_auto_speculation("qwen3_5", &mut s);
        assert_eq!(s.speculative.method, "mtp");
        let _ = std::fs::remove_file(&sidecar);
    }

    /// Qwen4-exp's own keys are refused on qwen3_5 by name, and its
    /// defaults pass (`LoadConfig::default` carries them for every family).
    #[test]
    fn qwen4exps_keys_are_refused_on_qwen3_5_by_name() {
        for (argv, key) in [
            (
                &["--moe-act", "f16"][..],
                "models.qwen4exp.moe.activation_dtype",
            ),
            (&["--no-ple"][..], "models.qwen4exp.ple.enabled"),
            (
                &["--ple-cache-rows", "8"][..],
                "models.qwen4exp.ple.cache_rows",
            ),
            (
                &["--prefill-mode", "replay"][..],
                "models.qwen4exp.prefill_mode",
            ),
        ] {
            let m = model_args(argv);
            let err = Qwen3_5Family
                .resolve(&Requested {
                    model: &m,
                    shape: shape(),
                })
                .unwrap_err()
                .to_string();
            assert!(err.contains(key), "{argv:?}: {err}");
        }
        let m = model_args(&[]);
        Qwen3_5Family
            .resolve(&Requested {
                model: &m,
                shape: shape(),
            })
            .unwrap();
    }

    #[test]
    fn flash_k_is_lfm2s() {
        let m = model_args(&["--flash-k", "2"]);
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: shape(),
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("--flash-k"), "{err}");
        Lfm2Family
            .resolve(&Requested {
                model: &m,
                shape: shape(),
            })
            .unwrap();
    }

    #[test]
    fn vision_is_a_capability_and_its_absence_has_a_reason() {
        assert!(LFM2_CAPABILITIES.vision);
        assert!(!QWEN4EXP_CAPABILITIES.vision);
        assert!(QWEN4EXP_CAPABILITIES
            .vision_missing
            .contains("no vision tower"));
        assert!(LFM2_CAPABILITIES.line().contains("vision=true"));
        assert_eq!(
            QWEN4EXP_CAPABILITIES.json()["vision"],
            serde_json::json!(false)
        );
    }
}
