//! `lloom` — the one binary over every served family: `serve` (the OpenAI
//! endpoint + the pipette sidecar contract, the former `lloom-serve`) and
//! `chat` (a streaming REPL on the same engine). The model is named once
//! (`model.path`, [`LoadConfig`]) and its family is read from the path: a safetensors
//! checkpoint directory is LFM2 (`lfm2.rs`, every arm the serving binary
//! had), a GGUF is Qwen4-exp (`qwen4exp.rs`). `lloom-serve` stays as a
//! binary name for the pipette runner and is `lloom serve` verbatim.

pub mod budgets;
pub mod causal_loaded;
pub mod chat;
pub mod common;
pub mod decision_endpoint;
mod decisions;
pub mod family;
pub mod front;
pub mod lfm2;
pub mod lfm2_gguf;
pub mod modes;
pub mod omni;
pub mod qwen3_5;
pub mod qwen4exp;
pub mod report;
mod request_slot;
pub mod serve;
pub mod served;
pub mod sessions;
pub mod template;
pub mod thinking;
pub mod vision;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use served::Served;
use std::path::{Path, PathBuf};

/// What `--model` and the load-time options name: the option tree's
/// `model`, `backend.device`, `weights.experts` and `models.qwen4exp` keys.
pub use crate::options::LoadConfig;

/// The serving shape the family builds for: `serve`'s width and cells, or
/// `chat`'s one sequence with prefix reuse.
pub struct Shape {
    /// The decode loop's options (`--readback`, the cadence, the prefill).
    pub decode: crate::DecodeConfig,
    /// The model's axes: the cache preset and the per-axis overrides.
    pub model: crate::ModelConfig,
    /// The speculative body.
    pub speculative: crate::SpeculativeConfig,
    /// The executor arms.
    pub executor: crate::ExecutorConfig,
    /// The memory budgets.
    pub memory: crate::MemoryConfig,
    /// The sampling defaults, which a resolved drafter is checked against
    /// (C11).
    pub sampling: crate::SamplingConfig,
    pub reuse_prefix: bool,
    pub seed_text: String,
    pub batch: usize,
    /// The most requests one intake tick of the batched scheduler takes
    /// off its queue while rows are live (`scheduler.intake_per_tick`).
    pub intake_per_tick: usize,
    pub mem_sample: bool,
    pub rows: String,
    pub share_prefix: bool,
    /// `scheduler.decision_ragged`; taken only under the `fast` contract.
    pub decision_ragged: bool,
    /// The load keeps a frame of the checkpoint ring for a session's pinned
    /// head (`session.keep = prefix`, `docs/sessions.md` §8): `serve`, one
    /// sequence, prefix reuse on.
    pub pin_prefix: bool,
}

impl Shape {
    /// The shape `o` names for `frontend`, whose own prefix reuse applies
    /// where `context.prefix.reuse` is unset (`crate::frontend`).
    pub fn of(
        o: &crate::EngineOptions,
        frontend: crate::frontend::Frontend,
        seed_text: String,
    ) -> Shape {
        Shape {
            decode: o.decode.clone(),
            model: o.model.clone(),
            speculative: o.speculative.clone(),
            executor: o.executor.clone(),
            memory: o.memory.clone(),
            sampling: o.sampling.clone(),
            reuse_prefix: o.prefix.reuse.unwrap_or_else(|| frontend.reuse_prefix()),
            seed_text,
            batch: o.scheduler.max_sequences,
            intake_per_tick: o.scheduler.intake_per_tick,
            mem_sample: o.report.memory_sample,
            rows: o.scheduler.rows.clone(),
            share_prefix: o.prefix.share,
            decision_ragged: o.scheduler.decision_ragged,
            pin_prefix: frontend == crate::frontend::Frontend::Serve
                && o.session.keep == "prefix"
                && o.prefix.reuse.unwrap_or_else(|| frontend.reuse_prefix())
                && o.scheduler.max_sequences <= 1,
        }
    }
}

/// What `--model` names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Family {
    /// A safetensors checkpoint directory.
    Lfm2(PathBuf),
    /// A GGUF's first shard (`general.architecture = qwen4exp`).
    Qwen4Exp(PathBuf),
    /// A GGUF's first shard (`qwen35` / `qwen35moe`).
    Qwen3_5(PathBuf),
    /// An LFM2 GGUF (`general.architecture = lfm2`, d1-3B's official files).
    Lfm2Gguf(PathBuf),
}

/// The family from the path: a `.gguf` file, or a directory holding GGUF
/// shards, is Qwen4-exp; a directory with a `config.json` is LFM2 (its
/// `model_type` is checked at load). No path: LFM2's default snapshot.
pub fn detect_family(model: Option<&Path>) -> Result<Family> {
    let path = match model {
        Some(p) => p.to_path_buf(),
        None => lfm2::default_model_dir()
            .context("no --model given and the LFM2.5-2.6B snapshot was not in the HF cache")?,
    };
    let (f, checkpoint) = family::detect(&path)?;
    Ok(match f.name() {
        "lfm2" | "lfm2_moe" => Family::Lfm2(checkpoint),
        "qwen3_5" => Family::Qwen3_5(checkpoint),
        "lfm2-gguf" => Family::Lfm2Gguf(checkpoint),
        _ => Family::Qwen4Exp(checkpoint),
    })
}

/// Load the engine `load` names for `shape`. Returns it with the path it
/// was loaded from. An adaptive commit the readback arm cannot carry, and
/// an EOS gate the executor arms cannot run, are named here with their
/// reasons, beside the arm the loaded line renders.
pub fn load_engine(load: &LoadConfig, mut shape: Shape) -> Result<(Box<dyn Served>, PathBuf)> {
    lloom_tensor::memory_admission::begin_model_load();
    use lloom::load_warning::{LoadWarning, LoadWarningKind};
    let mut notes = Vec::new();
    if let Some(note) = shape.decode.readback.adaptive_clamp() {
        eprintln!("[readback: {note}]");
        notes.push(LoadWarning::new(LoadWarningKind::Readback, note));
    }
    if let Some(note) = clamp_eos_gate(&mut shape) {
        eprintln!("[readback: {note}]");
        notes.push(LoadWarning::new(LoadWarningKind::Readback, note));
    }
    let (mut engine, path) = family::load(load, shape)?;
    engine.note_load_warnings_checked(notes)?;
    Ok((engine, path))
}

/// Turn the EOS gate off where the executor arms refuse it
/// (`ExecutorConfig::refuses_eos_gate`), and say why. The gate defaults on,
/// so a refusal here would meet every `--replay` run that did not ask for it.
fn clamp_eos_gate(shape: &mut Shape) -> Option<String> {
    if !shape.decode.readback.eos_gate {
        return None;
    }
    let why = shape.executor.refuses_eos_gate()?;
    shape.decode.readback.eos_gate = false;
    Some(format!("eos_gate is off: {why}"))
}

/// The help every subcommand ends with: where the options without a flag
/// are.
pub const OPTIONS_HELP: &str =
    "Every option is a key in one tree, and the flags above are the stable \
surface. `lloom options [GROUP]` lists every key (with --flow, --facet, --status, --internal, \
--schema); set any of them with --set KEY=VALUE, --<group>-config JSON, --config FILE, or the env \
(LLOOM_<KEY>, `__` between levels). Every older flag and env name still works.";

const LAYERS: &str = "Config layers";

/// The option layers every subcommand takes: a config file, the JSON
/// groups, the dotted overrides, and the flags generated from the groups —
/// applied in that order onto the defaults and the env.
#[derive(Args, Clone, Debug, Default)]
pub struct CommonArgs {
    /// A JSON file of options in the tree's shape (what `lloom config` prints), `$schema` allowed; any older dotted spelling is read too. Applied over the defaults and the env.
    #[arg(long, value_name = "PATH", help_heading = LAYERS)]
    pub config: Option<PathBuf>,

    /// One option by key, repeatable (`--set decode.readback.eos=gpu-batched`); any older dotted spelling is read too.
    #[arg(long, value_name = "KEY=VALUE", action = clap::ArgAction::Append, help_heading = LAYERS)]
    pub set: Vec<String>,

    /// The decode group as JSON (`'{"readback": "ring-hostdrain", "in_flight_forwards": 2}'`).
    #[arg(long, value_name = "JSON", hide = true)]
    pub decode_config: Option<String>,
    /// The storage tree's tuning group as JSON (`'{"qmv": {"q6k_nr0": 4}}'`).
    #[arg(long, value_name = "JSON", hide = true)]
    pub tuning_config: Option<String>,
    /// The model group as JSON (`'{"path": "…", "served_name": "x"}'`; the legacy `cache` and `axes` too).
    #[arg(long, value_name = "JSON", hide = true)]
    pub model_config: Option<String>,
    /// The speculative group as JSON (`'{"method": "mtp", "num_speculative_tokens": 2, "model": "…gguf"}'`).
    #[arg(long, value_name = "JSON", hide = true)]
    pub speculative_config: Option<String>,
    /// The executor group as JSON (`'{"replay": {"enabled": true}}'`).
    #[arg(long, value_name = "JSON", hide = true)]
    pub executor_config: Option<String>,
    /// The memory group as JSON (`'{"fraction": 0.75}'`).
    #[arg(long, value_name = "JSON", hide = true)]
    pub memory_config: Option<String>,
    /// The generation group as JSON (`'{"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}'`; vLLM's name).
    #[arg(long, value_name = "JSON", hide = true)]
    pub generation_config: Option<String>,
    /// The weights group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub weights_config: Option<String>,
    /// The context group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub context_config: Option<String>,
    /// The scheduler group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub scheduler_config: Option<String>,
    /// The prefill group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub prefill_config: Option<String>,
    /// The reasoning group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub reasoning_config: Option<String>,
    /// The prompt group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub prompt_config: Option<String>,
    /// The session group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub session_config: Option<String>,
    /// The kernels group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub kernels_config: Option<String>,
    /// The backend group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub backend_config: Option<String>,
    /// The models group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub models_config: Option<String>,
    /// The diagnostics group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub diagnostics_config: Option<String>,
    /// The server group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub server_config: Option<String>,
    /// The bench group as JSON.
    #[arg(long, value_name = "JSON", hide = true)]
    pub bench_config: Option<String>,

    /// LFM2: the DSpark speculative body (`speculative.method = dspark`).
    #[arg(long, hide = true)]
    pub dspark: bool,

    /// Qwen4-exp: the MTP head's sidecar GGUF (`speculative.method = mtp`, `speculative.model`).
    #[arg(long, value_name = "GGUF", hide = true)]
    pub mtp: Option<PathBuf>,

    #[command(flatten)]
    pub options: crate::options::EngineOptionsArgs,
}

impl CommonArgs {
    /// The `--<group>-config` flags given, in the order they apply: the
    /// group, its JSON, and the storage group its older keys fall back to.
    fn group_configs(&self) -> Vec<(&'static str, &String, Option<&'static str>)> {
        [
            ("decode", &self.decode_config, None),
            ("tuning", &self.tuning_config, None),
            ("model", &self.model_config, None),
            ("speculative", &self.speculative_config, None),
            ("executor", &self.executor_config, None),
            ("memory", &self.memory_config, None),
            ("generation", &self.generation_config, Some("sampling")),
            ("weights", &self.weights_config, None),
            ("context", &self.context_config, None),
            ("scheduler", &self.scheduler_config, None),
            ("prefill", &self.prefill_config, None),
            ("reasoning", &self.reasoning_config, None),
            ("prompt", &self.prompt_config, None),
            ("session", &self.session_config, None),
            ("kernels", &self.kernels_config, None),
            ("backend", &self.backend_config, None),
            ("models", &self.models_config, None),
            ("diagnostics", &self.diagnostics_config, None),
            ("server", &self.server_config, None),
            ("bench", &self.bench_config, None),
        ]
        .into_iter()
        .filter_map(|(g, j, f)| j.as_ref().map(|j| (g, j, f)))
        .collect()
    }

    /// Whether `--set` names `key` by any of its spellings.
    pub fn sets(&self, key: &str) -> bool {
        let idx = crate::option_tree::index();
        self.set.iter().any(|kv| {
            lloom_tensor::options::split_assignment(kv)
                .ok()
                .and_then(|(k, _)| idx.key_of(k))
                .is_some_and(|k| k.path == key)
        })
    }
}

/// A subcommand's own flags (the groups only it flattens), applied as the
/// top layer beside the common flags.
pub type FrontFlags<'a> = &'a dyn Fn(&mut crate::EngineOptions) -> std::result::Result<(), String>;

/// No subcommand flags.
pub fn no_front(_: &mut crate::EngineOptions) -> std::result::Result<(), String> {
    Ok(())
}

/// The process environment, for the env layer.
pub fn process_env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// Resolve the option tree from `common` for `frontend`: defaults < env
/// (the legacy names, then `LLOOM_<KEY>`) < `--config` < `--<group>-config`
/// < `--set` < flags, then the front-end's own default of every key no
/// layer named; every refusal, and every capability the build lacks,
/// checked once at the end. The tuning is installed into both library
/// crates here — before the device or the model exists, so every
/// `OnceLock` read site takes it.
pub fn resolve_options(
    common: &CommonArgs,
    front: FrontFlags<'_>,
    frontend: crate::frontend::Frontend,
) -> Result<crate::EngineOptions> {
    let o = resolve_layers(common, front, &process_env, false, Some(frontend))?.0;
    o.tuning.install().map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!("[contract: {}]", lloom_tensor::tuning::contract().report());
    Ok(o)
}

/// The option layers without a front-end's defaults and without installing
/// the tuning (`lloom config`, `lloom options`, the tests — nothing that
/// goes on to load a model).
pub fn resolve_options_uninstalled(
    common: &CommonArgs,
    front: FrontFlags<'_>,
) -> Result<crate::EngineOptions> {
    Ok(resolve_layers(common, front, &process_env, false, None)?.0)
}

/// One layer's effect: its label and the keys it moved, with their values.
pub type Layer = (
    String,
    std::collections::BTreeMap<String, serde_json::Value>,
);

/// The layers, applied in order, then `frontend`'s defaults where no layer
/// named the key; with `trace` each one's moves are kept (`lloom config
/// --explain`).
pub fn resolve_layers(
    common: &CommonArgs,
    front: FrontFlags<'_>,
    env: &dyn Fn(&str) -> Option<String>,
    trace: bool,
    frontend: Option<crate::frontend::Frontend>,
) -> Result<(crate::EngineOptions, Vec<Layer>)> {
    let idx = crate::option_tree::index();
    // The keys a layer named, by storage path: what a front-end's default
    // leaves alone, whatever value the layer gave.
    let mut named: std::collections::BTreeSet<String> = Default::default();
    let mut o = crate::EngineOptions::default();
    let mut layers: Vec<Layer> = Vec::new();
    let mut last = if trace {
        idx.flat(&o)
    } else {
        Default::default()
    };
    let mut mark = |label: &str, o: &crate::EngineOptions| {
        if !trace {
            return;
        }
        let now = idx.flat(o);
        let moved = now
            .iter()
            .filter(|(k, v)| last.get(*k) != Some(*v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if !moved.is_empty() {
            layers.push((label.to_string(), moved));
        }
        last = now;
    };
    for note in crate::option_grammar::legacy_notes(env) {
        eprintln!("{note}");
    }
    o.from_env_with(env)
        .map_err(|e| anyhow::anyhow!("env: {e}"))?;
    for key in idx.keys.values() {
        if env(&key.env()).is_some() || key.legacy_env().iter().any(|e| env(e).is_some()) {
            named.extend(key.targets.iter().map(|l| l.storage.clone()));
        }
    }
    mark("env", &o);
    if let Some(path) = &common.config {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading --config {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("--config {}: not JSON", path.display()))?;
        named.extend(
            idx.merge_written(&mut o, &v)
                .map_err(|e| anyhow::anyhow!("--config {}: {e}", path.display()))?,
        );
        mark(&format!("--config {}", path.display()), &o);
    }
    for (group, json, fallback) in common.group_configs() {
        let v: serde_json::Value =
            serde_json::from_str(json).with_context(|| format!("--{group}-config: not JSON"))?;
        named.extend(
            idx.merge_group_written(&mut o, group, &v, fallback)
                .map_err(|e| anyhow::anyhow!("--{group}-config: {e}"))?,
        );
        mark(&format!("--{group}-config"), &o);
    }
    for kv in &common.set {
        let (k, v) = lloom_tensor::options::split_assignment(kv)
            .map_err(|e| anyhow::anyhow!("--set {e}"))?;
        idx.set(&mut o, k, v)
            .map_err(|e| anyhow::anyhow!("--set {kv}: {e}"))?;
        match idx.storage_of(k) {
            Some(storage) => {
                named.insert(storage.to_string());
            }
            None => named.extend(
                idx.key_of(k)
                    .into_iter()
                    .flat_map(|key| key.targets.iter().map(|l| l.storage.clone())),
            ),
        }
        mark(&format!("--set {k}"), &o);
    }
    common
        .options
        .apply(&mut o)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    common.options.given("", &mut named);
    // A subcommand's own flags: the keys they moved (none of them carries
    // a front-end default other than an optional one, which is unset
    // until a value is given).
    let before_front = idx.flat(&o);
    front(&mut o).map_err(|e| anyhow::anyhow!("{e}"))?;
    for (k, v) in idx.flat(&o) {
        if before_front.get(&k) != Some(&v) {
            if let Some(key) = idx.keys.get(&k) {
                named.extend(key.targets.iter().map(|l| l.storage.clone()));
            }
        }
    }
    // The speculation shorthands (the harnesses' spellings), over the group.
    if common.dspark && common.mtp.is_some() {
        anyhow::bail!(
            "--dspark is LFM2's drafter and --mtp Qwen4-exp's; one speculative body per run"
        );
    }
    if common.dspark {
        o.speculative.method = "dspark".into();
    }
    if let Some(path) = &common.mtp {
        o.speculative.method = "mtp".into();
        o.speculative.drafter = Some(path.clone());
    }
    mark("flags", &o);
    if let Some(f) = frontend {
        let named = named_keys(idx, &named);
        let preset = preset_keys(idx, &o, &named);
        crate::frontend::apply(&mut o, f, &named, &preset).map_err(|e| anyhow::anyhow!("{e}"))?;
        mark(&format!("the {} default", f.name()), &o);
    }
    o.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    for note in crate::compiled::check(&o, &crate::compiled::this_build())
        .map_err(|e| anyhow::anyhow!("{e}"))?
    {
        eprintln!("{note}");
    }
    Ok((o, layers))
}

/// The keys a layer named, from the fields it named (`named`, by storage
/// path): every key with a named field, and every member of a compound group
/// a layer named by the group's own shorthand or by a member —
/// `--readback ring-fixed` writes `decode.readback`'s members as one
/// spelling, and the `decode.readback.eos` row must not re-set the EOS of a
/// readback the caller named.
pub(crate) fn named_keys(
    idx: &crate::option_tree::Index,
    named: &std::collections::BTreeSet<String>,
) -> std::collections::BTreeSet<String> {
    let under = |storage: &str, group: &str| {
        storage.len() > group.len()
            && storage.starts_with(group)
            && storage.as_bytes()[group.len()] == b'.'
    };
    let groups: Vec<String> = idx
        .keys
        .values()
        .filter_map(|k| k.compound.clone())
        .filter(|g| named.contains(g) || named.iter().any(|s| under(s, g)))
        .collect();
    idx.keys
        .values()
        .filter(|k| {
            k.targets
                .iter()
                .chain(&k.legacy)
                .any(|l| named.contains(&l.storage) || groups.iter().any(|g| under(&l.storage, g)))
        })
        .map(|k| k.path.clone())
        .collect()
}

/// The keys of the axes a named LFM2 preset sets (`lfm2_presets::AXES`),
/// given the keys a layer named (`named`): a preset is explicit for its axes
/// as a key is, so `--cache crown` keeps the crown's KV and attention. Such
/// an axis is unset in the tree whatever its spelling — unset is the
/// preset's. `auto` names no arm: it is the engine's rule over the shape,
/// which the front-end's rows refine.
pub(crate) fn preset_keys(
    idx: &crate::option_tree::Index,
    o: &crate::EngineOptions,
    named: &std::collections::BTreeSet<String>,
) -> std::collections::BTreeSet<String> {
    if !named.contains(crate::options::ModelConfig::PRESET_KEY) || o.model.cache == "auto" {
        return Default::default();
    }
    lloom::models::lfm2::Config::AXES_META
        .iter()
        .filter(|a| lloom::models::lfm2_presets::AXES.contains(&a.key))
        .flat_map(|axis| {
            idx.keys
                .values()
                .filter(move |k| {
                    k.targets
                        .iter()
                        .chain(&k.legacy)
                        .any(|l| l.meta.key == axis.key && l.meta.path == axis.path)
                })
                .map(|k| k.path.clone())
        })
        .collect()
}

/// `lloom config`: the resolved option tree as JSON — every key with the
/// value this invocation would run, in the shape `--config` reads back.
#[derive(Parser)]
#[command(name = "lloom config", about = "Print the resolved option tree as JSON", after_help = OPTIONS_HELP)]
pub struct ConfigCmdArgs {
    /// Print, beside the tree, where each value that is not its default
    /// came from, and what the couplings resolve.
    #[arg(long)]
    pub explain: bool,
    /// Print the storage tree (the groups' own shape) instead of the keys.
    #[arg(long, hide = true)]
    pub storage: bool,
    /// Resolve as this front-end would — `chat`, `serve` or `bench` — its own
    /// defaults applied where no layer named the key; without it, the
    /// library's tree.
    #[arg(long, value_name = "FRONTEND", value_parser = crate::frontend::Frontend::parse)]
    pub frontend: Option<crate::frontend::Frontend>,
    #[command(flatten)]
    pub common: CommonArgs,
    #[command(flatten)]
    pub front: FrontArgs,
}

/// The groups the front-ends flatten, all at once (`lloom config`).
#[derive(Args, Clone, Debug, Default)]
pub struct FrontArgs {
    #[command(flatten)]
    pub scheduler: crate::options::SchedulerConfigArgs,
    #[command(flatten)]
    pub prefix: crate::options::PrefixConfigArgs,
    #[command(flatten)]
    pub reasoning: crate::options::ReasoningConfigArgs,
    #[command(flatten)]
    pub prompt: crate::options::PromptConfigArgs,
    #[command(flatten)]
    pub session: crate::options::SessionConfigArgs,
    #[command(flatten)]
    pub report: crate::options::ReportConfigArgs,
    #[command(flatten)]
    pub server: crate::options::ServerConfigArgs,
    #[command(flatten)]
    pub bench: crate::options::BenchConfigArgs,
}

impl FrontArgs {
    pub fn apply(&self, o: &mut crate::EngineOptions) -> std::result::Result<(), String> {
        self.scheduler.apply(&mut o.scheduler)?;
        self.prefix.apply(&mut o.prefix)?;
        self.reasoning.apply(&mut o.reasoning)?;
        self.prompt.apply(&mut o.prompt)?;
        self.session.apply(&mut o.session)?;
        self.report.apply(&mut o.report)?;
        self.server.apply(&mut o.server)?;
        self.bench.apply(&mut o.bench)
    }
}

/// `lloom config`.
pub fn print_config(args: &ConfigCmdArgs) -> Result<()> {
    let front = |o: &mut crate::EngineOptions| args.front.apply(o);
    let (o, layers) = resolve_layers(
        &args.common,
        &front,
        &process_env,
        args.explain,
        args.frontend,
    )?;
    let idx = crate::option_tree::index();
    if args.storage {
        println!("{}", serde_json::to_string_pretty(&o)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&idx.render(&o))?);
    }
    if args.explain {
        let defaults = idx.flat(&crate::EngineOptions::default());
        let mut from: std::collections::BTreeMap<String, (String, serde_json::Value)> =
            Default::default();
        for (label, moved) in &layers {
            for (k, v) in moved {
                from.insert(k.clone(), (label.clone(), v.clone()));
            }
        }
        eprintln!("[explain: the values that are not their default, and the layer that set them]");
        for (k, (label, v)) in &from {
            if defaults.get(k) != Some(v) {
                eprintln!("{k} = {}  <- {label}", lloom_tensor::options::render(v));
            }
        }
        eprintln!("[explain: what the couplings resolve]");
        for line in crate::couplings::explain(&o) {
            eprintln!("{line}");
        }
    }
    Ok(())
}

/// `lloom options` (alias `knobs`): every key of the tree, or one group's.
#[derive(Parser)]
#[command(name = "lloom options", about = "List the option tree's keys", after_help = OPTIONS_HELP)]
pub struct OptionsCmdArgs {
    /// One group (`kernels`, `decode.readback`, …); every key when omitted.
    pub group: Option<String>,
    /// The keys acting on one data flow: token-out, host-feed-in, weights-in, commands-out, prompt-in, memory-return.
    #[arg(long)]
    pub flow: Option<String>,
    /// The keys of one facet: source, produce, buffer, transport, signal, wait, lead, cadence, locality, consume, qos, liveness.
    #[arg(long)]
    pub facet: Option<String>,
    /// The keys of one status: stable, advanced, arm, reference, instrument.
    #[arg(long)]
    pub status: Option<String>,
    /// Each key's internal notes (its references), its legacy spellings and its couplings.
    #[arg(long)]
    pub internal: bool,
    /// Print the JSON Schema of the tree instead, generated from the same declarations.
    #[arg(long)]
    pub schema: bool,
    /// Print this binary's compiled capabilities instead (`profile`, the backends), which the keys that need one name.
    #[arg(long)]
    pub compiled: bool,
}

/// `lloom options`.
pub fn print_options(args: &OptionsCmdArgs) -> Result<()> {
    if args.schema {
        println!(
            "{}",
            serde_json::to_string_pretty(&crate::option_schema::schema())?
        );
        return Ok(());
    }
    if args.compiled {
        for line in crate::compiled::table_lines(&crate::compiled::this_build()) {
            println!("{line}");
        }
        return Ok(());
    }
    if let Some(s) = &args.status {
        if lloom_tensor::options::Status::parse(s).is_none() {
            anyhow::bail!("--status {s}: stable | advanced | arm | reference | instrument");
        }
    }
    let idx = crate::option_tree::index();
    let defaults = idx.flat(&crate::EngineOptions::default());
    let group = args.group.as_deref().unwrap_or("");
    if !group.is_empty() && idx.keys_under(group).next().is_none() {
        anyhow::bail!("no key under `{group}`");
    }
    let mut shown = 0usize;
    for k in idx.keys_under(group) {
        let m = k.meta();
        if let Some(f) = &args.flow {
            if !m.flows.contains(&f.as_str()) {
                continue;
            }
        }
        if let Some(f) = &args.facet {
            if !m.facets.contains(&f.as_str()) {
                continue;
            }
        }
        if let Some(s) = &args.status {
            if m.status.name() != s {
                continue;
            }
        }
        shown += 1;
        let flag = match k.cli() {
            Some(c) => format!("--{c}"),
            None => "-".to_string(),
        };
        let value = if k.compound.is_some() {
            "(shorthand)".to_string()
        } else {
            lloom_tensor::options::render(defaults.get(&k.path).unwrap_or(&serde_json::Value::Null))
        };
        let arms = if m.forms.is_empty() {
            m.variants.join("|")
        } else {
            format!("{} of {}", m.forms.join("|"), m.schema)
        };
        println!(
            "{key}  [{status}]  {flag}  default={value}{arms}",
            key = k.path,
            status = m.status.name(),
            arms = if arms.is_empty() {
                String::new()
            } else {
                format!("  ({arms})")
            },
        );
        println!("    {}", m.doc);
        if let Some(d) = crate::frontend::describe(&k.path) {
            println!("    unset: {d}");
        }
        if !m.needs.is_empty() {
            match m.needs_part {
                "" => println!("    needs: the `{}` build", m.needs),
                part => println!("    needs: the `{}` build for {part}", m.needs),
            }
        }
        if args.internal {
            if m.internal != m.doc && !m.internal.is_empty() {
                println!("    internal: {}", m.internal);
            }
            let legacy: Vec<String> = k
                .flags()
                .into_iter()
                .chain(k.legacy_env().into_iter().map(|e| format!("${e}")))
                .chain(k.legacy_keys())
                .collect();
            println!("    env: {}  spellings: {}", k.env(), legacy.join(" "));
            if !m.facets.is_empty() || !m.flows.is_empty() {
                println!(
                    "    facets: {}  flows: {}",
                    m.facets.join(","),
                    m.flows.join(",")
                );
            }
            let c = crate::couplings::of(&k.path);
            if !c.is_empty() {
                println!("    couplings: {}", c.join(", "));
            }
        }
    }
    eprintln!("[{shown} keys]");
    Ok(())
}

#[derive(Parser)]
#[command(
    name = "lloom",
    about = "lloom: serve or chat with an LFM2 or Qwen4-exp checkpoint on the same engine"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// The OpenAI-compatible endpoint and the pipette benchmark sidecar
    /// contract on a loopback port.
    Serve(serve::ServeArgs),
    /// A streaming chat REPL on stdin/stdout.
    Chat(chat::ChatArgs),
    /// The benchmark cells (the pipette sidecar's) in-process, both families.
    Bench(serve::BenchArgs),
    /// Print the resolved option tree (the layers applied) as JSON.
    Config(ConfigCmdArgs),
    /// List the option tree: key, status, flag, default, description.
    #[command(alias = "knobs")]
    Options(OptionsCmdArgs),
}

/// How long a dropped device's retirement may take at exit before the
/// process leaves it to the OS.
const DEVICE_RETIREMENT_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// The command line, each front-end's help naming its own default of every
/// key it resolves itself (`crate::frontend`), where the flag's help ends.
pub fn command() -> clap::Command {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    for f in crate::frontend::Frontend::ALL {
        cmd = cmd.mut_subcommand(f.name(), |mut sub| {
            let ids: Vec<(String, String)> = sub
                .get_arguments()
                .filter_map(|a| {
                    let help = a.get_help()?.to_string();
                    let key = help.split("[key: ").nth(1)?.split(']').next()?.to_string();
                    let parts: Vec<String> = crate::frontend::rows_for(&key, f)
                        .map(|r| {
                            let when = match r.when.describe() {
                                "" => String::new(),
                                w => format!(" when {w}"),
                            };
                            let change = match crate::frontend::exception(r) {
                                Some(e) if !e.landed => {
                                    format!("; decision {}: {} once landed", e.decision, e.after)
                                }
                                _ => String::new(),
                            };
                            format!("{}{when}{change}", crate::frontend::value(r))
                        })
                        .collect();
                    if parts.is_empty() {
                        return None;
                    }
                    Some((
                        a.get_id().to_string(),
                        format!("{help} [unset here: {}]", parts.join("; ")),
                    ))
                })
                .collect();
            for (id, help) in ids {
                sub = sub.mut_arg(id, |a| a.help(help));
            }
            sub
        });
    }
    cmd
}

/// The `lloom` binary's entry. The HTTP retirement service owns whatever a
/// dropped server left running, so the process joins it before it exits; a
/// dropped device retires on the host retirement service, so the process
/// joins that retirement too, after the HTTP one.
pub fn main(argv: Vec<String>) -> Result<()> {
    use clap::FromArgMatches;
    let cli = Cli::from_arg_matches(&command().get_matches_from(argv)).unwrap_or_else(|e| e.exit());
    let result = match cli.command {
        Command::Serve(args) => serve::run(args),
        Command::Chat(args) => chat::run(args),
        Command::Bench(args) => serve::bench(args),
        Command::Config(args) => print_config(&args),
        Command::Options(args) => print_options(&args),
    };
    serve::join_http_retirement();
    let joined = join_device_retirement();
    result.and(joined)
}

/// Joins the retirement of every dropped device, says so when one is still
/// retiring at the budget, and judges the release witness's count.
fn join_device_retirement() -> Result<()> {
    let joined =
        lloom_tensor::join_device_retirement(std::time::Instant::now() + DEVICE_RETIREMENT_BUDGET);
    if !joined.completed {
        eprintln!(
            "[device retirement: {} set(s) with {} record(s) still retiring at the {}s budget]",
            joined.sets_retiring,
            joined.records_pending,
            DEVICE_RETIREMENT_BUDGET.as_secs(),
        );
    }
    retirement_verdict(&joined, lloom_tensor::tuning::contract().is_fast())
}

/// An allocation still listed when its record released it is an ordering
/// defect the witness turned into a kept allocation (`tuning.md` §6.3m). It
/// is printed whenever it happened; under `deterministic` it fails the run,
/// under `fast` the line is the report.
fn retirement_verdict(joined: &lloom_tensor::DeviceRetirementReport, fast: bool) -> Result<()> {
    if joined.listed_at_release == 0 {
        return Ok(());
    }
    eprintln!(
        "[device retirement: {} allocation(s) still in the residency set when their record released them; kept, never freed]",
        joined.listed_at_release
    );
    if fast {
        Ok(())
    } else {
        anyhow::bail!(
            "{} allocation(s) were released while the residency set still listed them (the deterministic contract fails on it)",
            joined.listed_at_release
        )
    }
}

/// The `lloom-serve` binary's entry: `lloom serve` with the same flags, so
/// the pipette runner's spawn line (`lloom-serve --model <dir> --port <p>`)
/// keeps working.
pub fn main_serve_compat(mut argv: Vec<String>) -> Result<()> {
    argv.insert(1.min(argv.len()), "serve".to_string());
    main(argv)
}
#[cfg(test)]
mod frontend_tests;
#[cfg(test)]
mod option_acceptance_tests;
#[cfg(test)]
mod spellings_tests;
#[cfg(test)]
mod test_alloc;

#[cfg(test)]
mod retirement_verdict_tests {
    use super::retirement_verdict;
    use lloom_tensor::DeviceRetirementReport;

    fn report(listed_at_release: u64) -> DeviceRetirementReport {
        DeviceRetirementReport {
            completed: true,
            worker_stopped: true,
            listed_at_release,
            ..Default::default()
        }
    }

    /// A kept allocation fails a deterministic run and is only reported
    /// under fast; none passes both.
    #[test]
    fn a_listed_release_fails_deterministic_and_is_reported_under_fast() {
        assert!(retirement_verdict(&report(0), false).is_ok());
        assert!(retirement_verdict(&report(0), true).is_ok());
        let refused = retirement_verdict(&report(3), false).expect_err("deterministic fails");
        assert!(refused.to_string().contains("3 allocation(s)"), "{refused}");
        assert!(retirement_verdict(&report(3), true).is_ok());
    }
}

#[cfg(test)]
mod option_layer_tests {
    use super::*;
    use crate::{EosKind, ReadbackKind};

    /// Every flag of every group flattened into `serve` and `chat` is
    /// distinct (clap's own check; a duplicate id or long name is a runtime
    /// parse failure otherwise).
    #[test]
    fn the_command_line_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    fn common(argv: &[&str]) -> CommonArgs {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            common: CommonArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        T::parse_from(full).common
    }

    #[test]
    fn the_layers_apply_in_order() {
        let dir = std::env::temp_dir().join(format!("lloom-options-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("c.json");
        std::fs::write(
            &file,
            r#"{"decode": {"readback": {"eos": "host-drain", "lookahead": 0}, "in_flight": 4}}"#,
        )
        .unwrap();
        let path = file.to_str().unwrap();
        // file < --decode-config < --set < flags
        let o = resolve_options_uninstalled(
            &common(&[
                "--config",
                path,
                "--decode-config",
                r#"{"in_flight": 5, "readback": {"wake_cadence": 2}}"#,
                "--set",
                "decode.in_flight=6",
                "--set",
                "decode.readback.kind=ring",
                "--in-flight",
                "7",
            ]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.decode.readback.eos, EosKind::HostDrain, "from the file");
        assert_eq!(o.decode.readback.lookahead, 0, "from the file");
        assert_eq!(o.decode.readback.wake_cadence, 2, "from --decode-config");
        assert_eq!(o.decode.readback.kind, ReadbackKind::Ring, "from --set");
        assert_eq!(
            o.decode.in_flight, 7,
            "the flag wins over --set over the json over the file"
        );
        // The dump (`lloom config`'s tree) is a valid --config that
        // reproduces the options.
        let dumped = dir.join("dump.json");
        std::fs::write(
            &dumped,
            serde_json::to_string_pretty(&crate::option_tree::index().render(&o)).unwrap(),
        )
        .unwrap();
        let back = resolve_options_uninstalled(
            &common(&["--config", dumped.to_str().unwrap()]),
            &no_front,
        )
        .unwrap();
        assert_eq!(back, o);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `lloom` ships `fast` and `--profile deterministic` is the opt-in; the
    /// tensor crate's half carries what the flag resolved; the profiler's
    /// switch moved to `--gpu-profile` and still takes a bool.
    #[test]
    fn the_contract_defaults_to_fast_and_the_flag_pins_deterministic() {
        if std::env::var_os("LLOOM_PROFILE").is_some() {
            eprintln!("LLOOM_PROFILE is set; the default is not observable here");
            return;
        }
        let o = resolve_options_uninstalled(&common(&[]), &no_front).unwrap();
        assert_eq!(o.tuning.contract.profile, "fast");
        assert_eq!(o.tuning.tensor().contract.profile, "fast");
        let o = resolve_options_uninstalled(&common(&["--profile", "deterministic"]), &no_front)
            .unwrap();
        assert_eq!(o.tuning.tensor().contract.profile, "deterministic");
        assert!(resolve_options_uninstalled(
            &common(&["--set", "tuning.contract.profile=relaxed"]),
            &no_front
        )
        .is_err());
        let o = resolve_options_uninstalled(&common(&["--gpu-profile"]), &no_front).unwrap();
        assert!(o.tuning.instrument.profile);
        assert_eq!(o.tuning.contract.profile, "fast");
    }

    #[test]
    fn a_refused_combination_names_itself() {
        // The blocking arm ignores the cadence knobs rather than refusing them.
        let o = resolve_options_uninstalled(
            &common(&["--readback", "blocking", "--adaptive-commit"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.decode.readback.arm(), crate::ReadbackArm::Blocking);
        // The EOS gate defaults on and rides the adaptive commit: every
        // fixed-cadence route serves without it rather than being refused.
        for argv in [
            &["--adaptive-commit=false"][..],
            &["--eos-gate", "--adaptive-commit=false"],
            &["--readback", "ring-la0"],
            &["--readback", "ring-fixed"],
            &["--lookahead", "0"],
        ] {
            let o = resolve_options_uninstalled(&common(argv), &no_front).unwrap();
            assert!(o.decode.readback.eos_gate, "{argv:?}");
            assert!(!o.decode.readback.ring_arm().adaptive, "{argv:?}");
        }
        // `ring` (the flag EOS read) resolves to the fixed cadence, and the
        // clamp note names the EOS gate that goes off with the commit.
        for argv in [
            &["--readback", "ring"][..],
            &["--readback", "ring", "--eos-gate"],
        ] {
            let o = resolve_options_uninstalled(&common(argv), &no_front).unwrap();
            assert!(
                o.decode
                    .readback
                    .adaptive_clamp()
                    .is_some_and(|n| n.contains("EOS gate")),
                "{argv:?}"
            );
            assert!(matches!(o.decode.readback.arm(), crate::ReadbackArm::Ring(r) if !r.adaptive));
        }
        // The stream reader streams every ring arm, the flag read included.
        let o = resolve_options_uninstalled(
            &common(&["--readback", "ring", "--stream-reader"]),
            &no_front,
        )
        .unwrap();
        assert!(o.decode.readback.stream_reader);
        let err = resolve_options_uninstalled(&common(&["--set", "decode.nope=1"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key `decode.nope`"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--set", "decode.readback"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected key=value"), "{err}");
    }

    /// The replay executor refuses the gate (`Device::eos_gate_begin`), so
    /// load turns it off and says why. No other executor arm does, and a
    /// run already ungated prints nothing.
    #[test]
    fn the_replay_executor_takes_the_eos_gate_off_at_load() {
        let shape = |argv: &[&str]| {
            let o = resolve_options_uninstalled(&common(argv), &no_front).unwrap();
            Shape {
                decode: o.decode,
                model: o.model,
                speculative: o.speculative,
                executor: o.executor,
                memory: o.memory,
                sampling: o.sampling,
                reuse_prefix: false,
                seed_text: String::new(),
                batch: 1,
                intake_per_tick: crate::options::DEFAULT_INTAKE_PER_TICK,
                mem_sample: false,
                rows: "auto".into(),
                share_prefix: false,
                decision_ragged: true,
                pin_prefix: false,
            }
        };
        let mut s = shape(&["--replay"]);
        let note = clamp_eos_gate(&mut s).expect("the clamp is named");
        assert!(
            note.contains("eos_gate is off") && note.contains("replay"),
            "{note}"
        );
        assert!(!s.decode.readback.eos_gate);
        for argv in [
            &[][..],
            &["--arena"],
            &["--packed"],
            &["--pass-plans"],
            &["--replay", "--eos-gate=false"],
        ] {
            let mut s = shape(argv);
            let before = s.decode.readback.eos_gate;
            assert_eq!(clamp_eos_gate(&mut s), None, "{argv:?}");
            assert_eq!(s.decode.readback.eos_gate, before, "{argv:?}");
        }
    }

    #[test]
    fn the_shorthands_are_lfm2_chats() {
        let o = resolve_options_uninstalled(
            &common(&["--readback", "ring-gpubatch-nonblocking"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.decode.readback.kind, ReadbackKind::Ring);
        assert_eq!(o.decode.readback.eos, EosKind::GpuBatchedNonblocking);
        let o =
            resolve_options_uninstalled(&common(&["--readback", "ring-la0"]), &no_front).unwrap();
        assert_eq!(o.decode.readback.lookahead, 0);
        assert!(!o.decode.readback.adaptive_commit);
        // The compound and its keys compose in either order: the spelling
        // sets kind + eos, the key flags override field by field.
        let o = resolve_options_uninstalled(
            &common(&["--readback", "ring-hostdrain", "--lookahead", "0"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.decode.readback.eos, EosKind::HostDrain);
        assert_eq!(o.decode.readback.lookahead, 0);
    }
}

#[cfg(test)]
mod model_group_tests {
    use super::*;
    use lloom::models::lfm2::{AttnImpl, Config, KvAppend};

    fn common(argv: &[&str]) -> CommonArgs {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            common: CommonArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        T::parse_from(full).common
    }

    #[test]
    fn explicit_mmproj_uses_existing_option_layers_and_roundtrips() {
        assert!(crate::LoadConfig::default().mmproj.is_none());
        for args in [
            vec!["--mmproj", "/one/mmproj.gguf"],
            vec!["--set", "model.mmproj=/one/mmproj.gguf"],
            vec!["--model-config", r#"{"mmproj":"/one/mmproj.gguf"}"#],
        ] {
            let options = resolve_options_uninstalled(&common(&args), &no_front).unwrap();
            assert_eq!(
                options.load.mmproj.as_deref(),
                Some(std::path::Path::new("/one/mmproj.gguf"))
            );
            let dump = serde_json::to_value(&options).unwrap();
            let mut restored = crate::EngineOptions::default();
            restored.merge_json(&dump).unwrap();
            assert_eq!(restored, options);
        }
    }

    #[test]
    fn the_axes_reach_the_config_by_flag_key_and_json() {
        let o = resolve_options_uninstalled(
            &common(&[
                "--cache",
                "crown",
                "--attn",
                "flash",
                "--gemv-silu=false",
                "--set",
                "model.kv-append=paged",
                "--model-config",
                r#"{"axes": {"qkv-layout": "fused"}}"#,
            ]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.model.cache, "crown");
        assert_eq!(o.model.axes.get("attn").map(String::as_str), Some("flash"));
        assert_eq!(
            o.model.axes.get("gemv-silu").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            o.model.axes.get("kv-append").map(String::as_str),
            Some("paged")
        );
        assert_eq!(
            o.model.axes.get("qkv-layout").map(String::as_str),
            Some("fused")
        );
        let mut cfg = Config::default();
        cfg.gemv_silu = true;
        o.model.apply_axes(&mut cfg).unwrap();
        assert_eq!(cfg.attn_impl, AttnImpl::FlashDecoding);
        assert_eq!(cfg.kv_append, KvAppend::Paged);
        assert!(
            !cfg.gemv_silu,
            "`--gemv-silu=false` forces the preset's arm off"
        );
        assert!(
            cfg.axes_line().contains("AttnImpl=FlashDecoding"),
            "{}",
            cfg.axes_line()
        );
        // The dump round-trips.
        let v = serde_json::to_value(&o).unwrap();
        let mut back = crate::EngineOptions::default();
        back.merge_json(&v).unwrap();
        assert_eq!(back, o);
    }

    /// `--kv-quant` is the model group's, from `lfm2::Config`'s own derive:
    /// every front-end reaches the field the cache is built from, and the
    /// loaded line renders the arm from that field.
    #[test]
    fn kv_quant_reaches_the_config_from_every_front_end() {
        use lloom::nn::ops::KvQuant;
        for (argv, want) in [
            (vec!["--kv-quant", "int8"], KvQuant::Int8),
            (vec!["--set", "model.kv-quant=int4"], KvQuant::Int4),
            (
                vec!["--model-config", r#"{"axes": {"kv-quant": "8"}}"#],
                KvQuant::Int8,
            ),
            (vec!["--kv-quant", "off"], KvQuant::Off),
        ] {
            let o = resolve_options_uninstalled(&common(&argv), &no_front).unwrap();
            let mut cfg = Config::default();
            o.model.apply_axes(&mut cfg).unwrap();
            assert_eq!(cfg.kv_quant, want, "{argv:?}");
            assert!(
                cfg.axes_line()
                    .contains(&format!("KvQuant={}", want.name())),
                "{argv:?}: {}",
                cfg.axes_line()
            );
        }
        let err = resolve_options_uninstalled(&common(&["--kv-quant", "int9"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--kv-quant"), "{err}");
    }

    #[test]
    fn a_bad_axis_or_preset_is_refused_at_parse() {
        let err = resolve_options_uninstalled(&common(&["--attn", "bogus"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--attn"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--cache", "shiny"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cache:"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--set", "model.nope=1"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key `model.nope`"), "{err}");
    }
}

#[cfg(test)]
mod sampling_and_speculation_tests {
    use super::*;
    use lloom::generation::Sampling;

    fn common(argv: &[&str]) -> CommonArgs {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            common: CommonArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        T::parse_from(full).common
    }

    /// Every layer reaches the processors — flags, `--set`, the JSON — and
    /// every one of them is refused by name out of range, the JSON included
    /// (it deserializes past the flag parsers). The checkpoint's
    /// `repetition_penalty` and `min_p` fill the unset fields under
    /// `--sampling-defaults checkpoint` only.
    #[test]
    fn the_processor_fields_reach_the_params_from_every_layer() {
        let o = resolve_options_uninstalled(
            &common(&[
                "--repetition-penalty",
                "1.1",
                "--presence-penalty",
                "0.5",
                "--frequency-penalty",
                "-0.5",
                "--min-p",
                "0.05",
                "--logit-bias",
                "42:-100, 7:2.5",
            ]),
            &no_front,
        )
        .unwrap();
        let p = o
            .sampling
            .params(&[1], crate::ReadbackArm::Blocking, 0)
            .processors;
        assert_eq!(
            p,
            lloom::generation::Processors {
                repetition_penalty: 1.1,
                presence_penalty: 0.5,
                frequency_penalty: -0.5,
                min_p: 0.05,
                logit_bias: vec![(7, 2.5), (42, -100.0)],
            }
        );
        let o = resolve_options_uninstalled(
            &common(&[
                "--generation-config",
                r#"{"presence_penalty": 1.0, "logit_bias": {"3": -5}}"#,
                "--set",
                "sampling.min_p=0.2",
            ]),
            &no_front,
        )
        .unwrap();
        let p = o.sampling.processors();
        assert_eq!(
            (p.presence_penalty, p.min_p, p.logit_bias.clone()),
            (1.0, 0.2, vec![(3, -5.0)])
        );
        assert!(!resolve_options_uninstalled(&common(&[]), &no_front)
            .unwrap()
            .sampling
            .processors()
            .is_active());
        for (argv, name) in [
            (vec!["--presence-penalty", "2.5"], "presence_penalty"),
            (
                vec!["--set", "sampling.repetition_penalty=0"],
                "repetition_penalty",
            ),
            (vec!["--generation-config", r#"{"min_p": 2}"#], "min_p"),
            (
                vec!["--generation-config", r#"{"logit_bias": {"3": 101}}"#],
                "logit_bias",
            ),
            (vec!["--logit-bias", "x:1"], "not a token id"),
        ] {
            let e = resolve_options_uninstalled(&common(&argv), &no_front)
                .unwrap_err()
                .to_string();
            assert!(e.contains(name), "{argv:?}: {e}");
        }
        let gc = lloom::hub::GenerationConfig {
            repetition_penalty: Some(1.1),
            min_p: Some(0.1),
            ..Default::default()
        };
        let mut s = resolve_options_uninstalled(&common(&["--min-p", "0.3"]), &no_front)
            .unwrap()
            .sampling;
        s.apply_checkpoint(Some(&gc));
        assert_eq!(
            s.repetition_penalty, None,
            "greedy defaults ignore the checkpoint"
        );
        s.defaults = "checkpoint".into();
        s.apply_checkpoint(Some(&gc));
        assert_eq!(
            (s.repetition_penalty, s.min_p),
            (Some(1.1), Some(0.3)),
            "unset fields only"
        );
    }

    #[test]
    fn sampling_fields_cross_real_cli_key_and_json_layers() {
        let cases = [
            (
                vec!["-t", "0.7", "--gumbel", "--top-k", "2", "--top-p", "0.6"],
                Sampling::FilteredGumbel {
                    temperature: 0.7,
                    k: 2,
                    p: 0.6,
                },
            ),
            (
                vec![
                    "--set",
                    "generation.temperature=0.7",
                    "--set",
                    "generation.sampling.method=gumbel",
                    "--set",
                    "generation.top_k=2",
                    "--set",
                    "generation.top_p=0.6",
                ],
                Sampling::FilteredGumbel {
                    temperature: 0.7,
                    k: 2,
                    p: 0.6,
                },
            ),
            (
                vec![
                    "--generation-config",
                    r#"{"temperature":" 0.7 ","top_k":"+2","top_p":"0.6","sampling":{"method":"gumbel"}}"#,
                ],
                Sampling::FilteredGumbel {
                    temperature: 0.7,
                    k: 2,
                    p: 0.6,
                },
            ),
            (
                vec![
                    "--generation-config",
                    r#"{"temperature":0.7,"top_k":0,"top_p":1,"sampling":{"method":"multinomial"}}"#,
                ],
                Sampling::All { temperature: 0.7 },
            ),
            (
                vec!["-t", "0", "--gumbel", "--top-k", "2", "--top-p", "0.6"],
                Sampling::ArgMax,
            ),
            (
                vec!["-t", "0.0000001", "--gumbel", "--top-k", "2"],
                Sampling::FilteredGumbel {
                    temperature: 1e-7,
                    k: 2,
                    p: 1.0,
                },
            ),
        ];
        for (argv, want) in cases {
            let o = resolve_options_uninstalled(&common(&argv), &no_front).unwrap();
            assert_eq!(o.sampling.sampling().unwrap(), want, "{argv:?}");
        }
        for argv in [
            vec!["--generation-config", r#"{"temperature":-1}"#],
            vec!["--generation-config", r#"{"top_p":1.1}"#],
            vec!["--generation-config", r#"{"top_p":"-0.1"}"#],
            vec!["--generation-config", r#"{"top_k":-1}"#],
            vec!["--set", "generation.temperature=nan"],
        ] {
            assert!(
                resolve_options_uninstalled(&common(&argv), &no_front).is_err(),
                "{argv:?}"
            );
        }
        // A ring named by flag or by key with a host-side sampler resolves
        // (C2): the options keep the ring and the sampler, and each request
        // runs on the blocking arm with a note.
        for argv in [
            vec![
                "-t",
                "0.7",
                "--gumbel",
                "--top-k",
                "2",
                "--readback",
                "ring",
            ],
            vec![
                "-t",
                "0.7",
                "--gumbel",
                "--top-k",
                "2",
                "--set",
                "decode.readback.kind=ring",
            ],
        ] {
            let o = resolve_options_uninstalled(&common(&argv), &no_front).unwrap();
            assert_eq!(
                o.sampling.sampling().unwrap(),
                Sampling::FilteredGumbel {
                    temperature: 0.7,
                    k: 2,
                    p: 1.0,
                },
                "{argv:?}"
            );
            assert_eq!(
                o.decode.readback.kind,
                crate::ReadbackKind::Ring,
                "{argv:?}"
            );
            assert!(
                crate::couplings::explain(&o)
                    .iter()
                    .any(|l| l.starts_with("decode.readback.kind = ring -> blocking")),
                "{argv:?}"
            );
        }
    }

    #[test]
    fn the_sampler_arms_follow_the_fields() {
        let o = resolve_options_uninstalled(&common(&[]), &no_front).unwrap();
        assert_eq!(
            o.sampling.sampling().unwrap(),
            Sampling::ArgMax,
            "greedy by default"
        );
        let o = resolve_options_uninstalled(
            &common(&["-t", "0.7", "--top-p", "0.9", "-n", "64", "--seed", "7"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(
            o.sampling.sampling().unwrap(),
            Sampling::TopP {
                p: 0.9,
                temperature: 0.7
            }
        );
        assert_eq!(o.sampling.max_tokens, 64);
        assert_eq!(o.sampling.seed, Some(7));
        let o = resolve_options_uninstalled(
            &common(&["-t", "0.7", "--top-k", "40", "--top-p", "0.9"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(
            o.sampling.sampling().unwrap(),
            Sampling::TopKThenTopP {
                k: 40,
                p: 0.9,
                temperature: 0.7
            }
        );
        let o =
            resolve_options_uninstalled(&common(&["-t", "0.7", "--gumbel"]), &no_front).unwrap();
        assert_eq!(
            o.sampling.sampling().unwrap(),
            Sampling::GumbelSoftmax { temperature: 0.7 }
        );
        let filtered = resolve_options_uninstalled(
            &common(&["-t", "0.7", "--gumbel", "--top-k", "2", "--top-p", "0.4"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(
            filtered.sampling.sampling().unwrap(),
            Sampling::FilteredGumbel {
                temperature: 0.7,
                k: 2,
                p: 0.4,
            }
        );
        assert!(filtered.sampling.sampling().unwrap().requires_host());
        let o = resolve_options_uninstalled(
            &common(&[
                "--generation-config",
                r#"{"temperature": 0.3, "stop": ["<|im_end|>", "\n\n"], "ignore_eos": true}"#,
            ]),
            &no_front,
        )
        .unwrap();
        assert_eq!(
            o.sampling.sampling().unwrap(),
            Sampling::All { temperature: 0.3 }
        );
        assert_eq!(o.sampling.stop, vec!["<|im_end|>", "\n\n"]);
        let p = o.sampling.params(&[7], crate::ReadbackArm::Blocking, 0);
        assert_eq!(
            p.eos_ids,
            vec![u32::MAX],
            "ignore_eos replaces the EOS set with an id no tokenizer emits"
        );
        assert_eq!(p.max_tokens, 512);
        // The checkpoint's defaults fill only unset fields, only when asked.
        let gc = lloom::hub::GenerationConfig {
            temperature: Some(0.6),
            top_p: Some(0.95),
            ..Default::default()
        };
        let mut s = o.sampling.clone();
        s.apply_checkpoint(Some(&gc));
        assert_eq!(
            s.temperature,
            Some(0.3),
            "greedy mode ignores the checkpoint"
        );
        s.defaults = "checkpoint".into();
        s.apply_checkpoint(Some(&gc));
        assert_eq!(s.temperature, Some(0.3), "a set field stays");
        assert_eq!(s.top_p, Some(0.95), "an unset one takes the checkpoint's");
    }

    #[test]
    fn the_speculation_shorthands_fill_the_group() {
        let o = resolve_options_uninstalled(&common(&["--dspark", "--dspark-k", "3"]), &no_front)
            .unwrap();
        assert_eq!(o.speculative.method, "dspark");
        assert_eq!(o.speculative.k, Some(3));
        assert_eq!(o.speculative.drafter, None);
        let o = resolve_options_uninstalled(
            &common(&["--mtp", "/tmp/mtp.gguf", "--mtp-k", "2"]),
            &no_front,
        )
        .unwrap();
        assert_eq!(o.speculative.method, "mtp");
        assert_eq!(o.speculative.k, Some(2));
        assert_eq!(
            o.speculative.drafter.as_deref(),
            Some(std::path::Path::new("/tmp/mtp.gguf"))
        );
        let o = resolve_options_uninstalled(
            &common(&[
                "--speculative-config",
                r#"{"method": "dspark", "k": 4, "drafter": "/d"}"#,
            ]),
            &no_front,
        )
        .unwrap();
        assert_eq!(
            (o.speculative.method.as_str(), o.speculative.k),
            ("dspark", Some(4))
        );
        let err = resolve_options_uninstalled(&common(&["--dspark", "--mtp", "/x"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("one speculative body"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--spec-method", "ngram"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a speculation method"), "{err}");
    }
}

#[cfg(test)]
mod executor_and_memory_tests {
    use super::*;

    fn common(argv: &[&str]) -> CommonArgs {
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            common: CommonArgs,
        }
        let mut full = vec!["t"];
        full.extend_from_slice(argv);
        T::parse_from(full).common
    }

    #[test]
    fn the_groups_take_lfm2_chats_flags() {
        let o = resolve_options_uninstalled(
            &common(&[
                "--replay",
                "--arena-layout",
                "non-aliasing",
                "--replay-barriers",
                "always",
                "--pass-plans",
                "--memory-fraction",
                "0.7",
                "--kv-capacity",
                "4096",
                "--kv-bucket",
                "2048",
                "--kv-grow-fraction",
                "0.5",
                "--free-budget-mb",
                "512",
                "--slots",
                "200",
                "--kv-reserve-tokens",
                "1024",
            ]),
            &no_front,
        )
        .unwrap();
        assert!(o.executor.replay && o.executor.pass_plans && !o.executor.arena);
        assert_eq!(o.executor.arena_layout, "non-aliasing");
        // The replay's arena is the single-key one; the plan cache would
        // bypass its recording, so the replay supersedes it.
        assert!(!o.executor.pass_plans_armed());
        assert_eq!(
            o.executor.line(),
            "arena=non-aliasing packed=true replay=always/run-start pass_plans=false"
        );
        assert_eq!(o.memory.memory_fraction, 0.7);
        assert_eq!(o.memory.kv_capacity, 4096);
        assert_eq!(o.memory.kv_bucket, 2048);
        assert_eq!(o.memory.kv_grow_fraction, Some(0.5));
        assert_eq!(o.memory.free_budget_mb, Some(512));
        assert_eq!(o.memory.slots, "200");
        assert_eq!(o.memory.kv_reserve_tokens, Some(1024));
        let err = resolve_options_uninstalled(&common(&["--arena-layout", "loose"]), &no_front)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not an arena layout"), "{err}");
        let o = resolve_options_uninstalled(&common(&[]), &no_front).unwrap();
        assert_eq!(
            o.executor.line(),
            "arena=off packed=false replay=off pass_plans=true"
        );
        let o =
            resolve_options_uninstalled(&common(&["--pass-plans", "false"]), &no_front).unwrap();
        assert_eq!(
            o.executor.line(),
            "arena=off packed=false replay=off pass_plans=false"
        );
        let o = resolve_options_uninstalled(&common(&["--arena"]), &no_front).unwrap();
        assert_eq!(
            o.executor.line(),
            "arena=packed packed=false replay=off pass_plans=false"
        );
        // The layout selects the plans' layout too, and the line says so.
        let o =
            resolve_options_uninstalled(&common(&["--arena-layout", "non-aliasing"]), &no_front)
                .unwrap();
        assert_eq!(
            o.executor.line(),
            "arena=off packed=false replay=off pass_plans=true/non-aliasing"
        );
        assert_eq!(o.memory.kv_capacity, 8704, "the serving default");
    }

    /// A speculative body does not install the single-key arena, so the load
    /// installs and reports the pass-plan cache instead and names the change;
    /// with the cache off nothing planned serves, and says so. A plain body
    /// keeps the arena, and replay is the installer's refusal, not this.
    #[test]
    fn an_arena_under_a_speculative_body_is_not_reported_as_installed() {
        let o = resolve_options_uninstalled(&common(&["--arena"]), &no_front).unwrap();
        assert_eq!(o.executor.under_body(false), (o.executor.clone(), None));
        let (spec, note) = o.executor.under_body(true);
        assert_eq!(
            spec.line(),
            "arena=off packed=false replay=off pass_plans=true"
        );
        let note = note.expect("the declined arena is named");
        assert!(
            note.starts_with("executor.arena is not installed under a speculative body")
                && note.ends_with("the pass-plan cache serves instead"),
            "{note}"
        );
        let o =
            resolve_options_uninstalled(&common(&["--arena", "--pass-plans", "false"]), &no_front)
                .unwrap();
        let (spec, note) = o.executor.under_body(true);
        assert_eq!(
            spec.line(),
            "arena=off packed=false replay=off pass_plans=false"
        );
        assert!(note.unwrap().contains("no planned executor serves"));
        let o = resolve_options_uninstalled(&common(&["--replay"]), &no_front).unwrap();
        assert_eq!(o.executor.under_body(true), (o.executor.clone(), None));
    }
}
