//! The family layer: what a served model family can do, what it refuses, and
//! how it loads — the seam a third family implements (`decode-driver-design.md`
//! Stage 7, the shape proposed there).
//!
//! `detect(path) → the family → capabilities() → resolve(&Requested) →
//! load(Requested)`. The refusals live in `resolve`, once per family, and name
//! the reason (a family that cannot serve a shape refuses it, never serves it
//! wrong — `served.rs`). `Served` stays the `dyn`-per-request seam; nothing
//! here is on the token path.

use super::{lfm2, qwen4exp, ModelArgs, Served, Shape};
use anyhow::{Context, Result};
#[allow(unused_imports)]
use anyhow::Context as _;
use lloom::nn::ops::KvQuant;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// What a family can do, as constants a caller and `/health` read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub family: &'static str,
    /// Forward passes the host may enqueue past an unlearned EOS.
    pub lookahead_max: usize,
    /// The ring arms: `full`, or `by-name` (honoured, but the fed id is read
    /// back per step, so no faster than blocking) with the reason.
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
    /// Why the arms missing from `kv_quant` are refused.
    pub kv_quant_missing: &'static str,
    /// Image input (OpenAI `image_url` parts, `lloom chat --image`). A
    /// checkpoint of the family can still lack a tower; the engine refuses
    /// that one by its own reason.
    pub vision: bool,
    /// Why a family without vision refuses an image.
    pub vision_missing: &'static str,
}

impl Capabilities {
    pub fn json(&self) -> Value {
        json!({
            "family": self.family,
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
        })
    }

    pub fn line(&self) -> String {
        format!(
            "lookahead_max={} ring={} batch={} prefix_share={} speculation=[{}] model_axes={} kv_quant=[{}] kv_dtype={} vision={} rewind={:?}",
            self.lookahead_max,
            self.ring,
            self.batch,
            self.prefix_share,
            self.speculation.join(","),
            self.model_axes,
            self.kv_quant_names().join(","),
            self.kv_dtype,
            self.vision,
            self.rewind,
        )
    }

    fn kv_quant_names(&self) -> Vec<String> {
        self.kv_quant
            .iter()
            .map(|q| q.name().to_ascii_lowercase())
            .collect()
    }
}

/// The `model.axes` key of the KV precision axis (`lfm2::Config`'s `#[axis]`
/// flag, which `ModelConfig::set` keys the map by).
const KV_QUANT_AXIS: &str = "kv-quant";

/// What a run asked for: the family's own load flags and the option groups.
pub struct Requested<'a> {
    pub model: &'a ModelArgs,
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

pub const LFM2_CAPABILITIES: Capabilities = Capabilities {
    family: "lfm2",
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
    kv_quant_missing: "",
    vision: true,
    vision_missing: "",
};

pub const QWEN4EXP_CAPABILITIES: Capabilities = Capabilities {
    family: "qwen4exp",
    lookahead_max: 0,
    ring: "by-name: the fed id is read back per step (the PLE hashes it on the host), no faster than blocking",
    batch: true,
    prefix_share: false,
    speculation: &["mtp"],
    model_axes: false,
    rewind: "one decode step (the standing window)",
    executor_arms: false,
    kv_quant: &[KvQuant::Off],
    kv_dtype: "f32",
    kv_quant_missing: "its attention KV is stored at f32 and no quantized arm is built for it \
                       (head_dim 256 and QSA's selected-block read; sequence-state.md §10.7)",
    vision: false,
    vision_missing: "no vision tower is built for Qwen4-exp (its image input is #765)",
};

/// Every family, in detection order.
pub static FAMILIES: [&dyn Family; 2] = [&Lfm2Family, &Qwen4ExpFamily];

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
        "{}: neither a checkpoint directory (config.json — LFM2) nor a GGUF shard or a directory of shards (Qwen4-exp)",
        path.display()
    )
}

/// The KV precision arm requested, refused unless `cap` serves it. `f16` is
/// `Off`'s spelling where `Off` is an f16 store, and refused where it is not,
/// so a name never selects a store it does not describe.
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
            "kv-quant = {value} is not {}'s (its KV arms: [{arms}]): {}",
            cap.family,
            cap.kv_quant_missing
        );
    }
    if value == "f16" && cap.kv_dtype != "f16" {
        anyhow::bail!(
            "kv-quant = f16 names an f16 KV store; {}'s `off` stores {} (its KV arms: [{arms}]): {}",
            cap.family,
            cap.kv_dtype,
            cap.kv_quant_missing
        );
    }
    Ok(())
}

/// The refusals every family shares.
fn resolve_common(cap: &Capabilities, req: &Requested<'_>) -> Result<()> {
    let s = &req.shape;
    if !cap.speculation.contains(&s.speculative.method.as_str()) && s.speculative.method != "none" {
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
        .keys()
        .map(String::as_str)
        .filter(|k| *k != KV_QUANT_AXIS)
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
        anyhow::bail!("{} has no prefix sharing between rows (its recurrent channels have no page to share)", cap.family);
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
        let engine = lfm2::Engine::load(lfm2::LoadOptions {
            model_dir: dir,
            cpu: model.cpu,
            kv_capacity: shape.memory.kv_capacity,
            readback: shape.decode.readback.arm(),
            decode: shape.decode.clone(),
            model: shape.model.clone(),
            reuse_prefix: shape.reuse_prefix,
            warmup: !model.no_warmup,
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
        })?;
        Ok(Box::new(engine))
    }
}

impl Family for Qwen4ExpFamily {
    fn name(&self) -> &'static str {
        "qwen4exp"
    }
    fn detect(&self, path: &Path) -> Option<PathBuf> {
        if path.is_file() {
            return (path.extension().and_then(|e| e.to_str()) == Some("gguf")).then(|| path.to_path_buf());
        }
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
            .cloned()
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
        match req.model.experts.as_str() {
            "stream" | "resident" => {}
            other => anyhow::bail!("--experts {other}: stream | resident"),
        }
        match req.model.prefill_mode.as_str() {
            "chunk" | "replay" => {}
            other => anyhow::bail!("--prefill-mode {other}: chunk | replay"),
        }
        Ok(())
    }
    fn load(&self, first: PathBuf, req: Requested<'_>) -> Result<Box<dyn Served>> {
        let (model, shape) = (req.model, req.shape);
        let spec = &shape.speculative;
        qwen4exp::Engine::load(qwen4exp::LoadOptions {
            first_shard: first,
            tokenizer: model.tokenizer.clone(),
            cpu: model.cpu,
            kv_capacity: shape.memory.kv_capacity,
            readback: shape.decode.readback.arm(),
            decode: shape.decode.clone(),
            reuse_prefix: shape.reuse_prefix,
            warmup: !model.no_warmup,
            seed_text: shape.seed_text.clone(),
            model_name: model.model_name.clone(),
            prefill_tile: shape.decode.prefill.tile,
            memory_fraction: shape.memory.memory_fraction,
            mem_sample: shape.mem_sample,
            experts: model.experts.clone(),
            slots: shape.memory.slots.clone(),
            loader: model.loader.clone(),
            ple_cache_rows: model.ple_cache_rows,
            ple: !model.no_ple,
            moe_act: model.moe_act.clone(),
            mtp: (spec.method == "mtp").then(|| spec.drafter.clone()).flatten(),
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
        })
    }
}

/// Detect, resolve, load — the one entry `serve` and `chat` call.
pub fn load(model: &ModelArgs, shape: Shape) -> Result<(Box<dyn Served>, PathBuf)> {
    let path = match &model.model {
        Some(p) => p.clone(),
        None => lfm2::default_model_dir()
            .context("no --model given and the LFM2.5-2.6B snapshot was not in the HF cache")?,
    };
    let (family, checkpoint) = detect(&path)?;
    let req = Requested { model, shape };
    family
        .resolve(&req)
        .map_err(|e| anyhow::anyhow!("{} cannot serve this shape: {e}", family.name()))?;
    let engine = family.load(checkpoint.clone(), req)?;
    Ok((engine, checkpoint))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn model_args(argv: &[&str]) -> ModelArgs {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            m: ModelArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        T::parse_from(full).m
    }

    fn shape() -> Shape {
        Shape {
            decode: Default::default(),
            model: Default::default(),
            speculative: Default::default(),
            executor: Default::default(),
            memory: Default::default(),
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
        let err = Lfm2Family.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("not lfm2's"), "{err}");
        let mut s = shape();
        s.speculative.method = "dspark".into();
        let err = Qwen4ExpFamily.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("not qwen4exp's"), "{err}");
        let mut s = shape();
        s.model.axes.insert("attn".into(), "flash".into());
        let err = Qwen4ExpFamily.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("LFM2's `Config`'s"), "{err}");
        let mut s = shape();
        s.executor.replay = true;
        let err = Qwen4ExpFamily.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("pass_plans"), "{err}");
        let mut s = shape();
        s.speculative.method = "mtp".into();
        let err = Qwen4ExpFamily.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("needs the MTP sidecar"), "{err}");
        let mut s = shape();
        s.share_prefix = true;
        let err = Lfm2Family.resolve(&Requested { model: &m, shape: s }).unwrap_err().to_string();
        assert!(err.contains("--rows pool"), "{err}");
        // The shapes each family serves pass.
        let mut s = shape();
        s.speculative.method = "dspark".into();
        s.executor.replay = false;
        Lfm2Family.resolve(&Requested { model: &m, shape: s }).unwrap();
        let mut s = shape();
        s.speculative.method = "mtp".into();
        s.speculative.drafter = Some("/x.gguf".into());
        Qwen4ExpFamily.resolve(&Requested { model: &m, shape: s }).unwrap();
    }

    fn with_kv_quant(value: &str) -> Shape {
        let mut s = shape();
        s.model.set(KV_QUANT_AXIS, value).unwrap();
        s
    }

    #[test]
    fn each_family_serves_its_own_kv_arms_and_refuses_the_rest_by_name() {
        let m = model_args(&[]);
        for v in ["int8", "8", "int4", "4"] {
            let err = Qwen4ExpFamily
                .resolve(&Requested {
                    model: &m,
                    shape: with_kv_quant(v),
                })
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("kv-quant = {v} is not qwen4exp's")),
                "{err}"
            );
            assert!(err.contains("its KV arms: [off]"), "{err}");
            assert!(err.contains("stored at f32"), "{err}");
        }
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: with_kv_quant("f16"),
            })
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`off` stores f32"),
            "f16 must not name Qwen4-exp's f32 store: {err}"
        );
        Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: with_kv_quant("off"),
            })
            .expect("off is Qwen4-exp's own KV arm");
        // The other axes are still refused, and kv-quant is not blamed for them.
        let mut s = with_kv_quant("off");
        s.model.set("attn", "sdpa").unwrap();
        let err = Qwen4ExpFamily
            .resolve(&Requested {
                model: &m,
                shape: s,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("the model axes (attn)"), "{err}");
        for v in ["off", "f16", "int8", "8", "int4", "4"] {
            Lfm2Family
                .resolve(&Requested {
                    model: &m,
                    shape: with_kv_quant(v),
                })
                .unwrap_or_else(|e| panic!("lfm2 serves kv-quant = {v}: {e}"));
        }
        assert!(LFM2_CAPABILITIES
            .line()
            .contains("kv_quant=[off,int8,int4] kv_dtype=f16"));
        assert!(QWEN4EXP_CAPABILITIES
            .line()
            .contains("kv_quant=[off] kv_dtype=f32"));
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
        assert_eq!((f.name(), p), ("qwen4exp", dir.join("qwen/m-00001-of-00002.gguf")));
        let (f, _) = detect(&dir.join("qwen/m-00002-of-00002.gguf")).unwrap();
        assert_eq!(f.name(), "qwen4exp");
        let err = detect(&dir.join("nope")).err().map(|e| e.to_string()).unwrap();
        assert!(err.contains("no such model"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(LFM2_CAPABILITIES.speculation, &["dspark"]);
        assert!(QWEN4EXP_CAPABILITIES.line().contains("lookahead_max=0"));
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
