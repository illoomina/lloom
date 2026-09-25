//! `lloom` — the one binary over every served family: `serve` (the OpenAI
//! endpoint + the pipette sidecar contract, the former `lloom-serve`) and
//! `chat` (a streaming REPL on the same engine). The model is named once
//! ([`ModelArgs`]) and its family is read from the path: a safetensors
//! checkpoint directory is LFM2 (`lfm2.rs`, every arm the serving binary
//! had), a GGUF is Qwen4-exp (`qwen4exp.rs`). `lloom-serve` stays as a
//! binary name for the pipette runner and is `lloom serve` verbatim.

pub mod chat;
pub mod common;
pub mod lfm2;
pub mod qwen4exp;
pub mod serve;
pub mod served;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use served::Served;
use std::path::{Path, PathBuf};

/// Which model, and the family's own knobs. Shared by `serve` and `chat`.
#[derive(Args, Clone, Debug)]
pub struct ModelArgs {
    /// The model: an LFM2 checkpoint directory (config.json, tokenizer.json,
    /// *.safetensors, chat_template.jinja) or a Qwen4-exp GGUF (the first
    /// shard, `-00001-of-0000N`; its siblings are found beside it, or a
    /// directory holding them). Defaults to LFM2_MODEL_DIR, then the
    /// LFM2.5-2.6B snapshot in the HF cache.
    #[arg(long, alias = "model-dir")]
    pub model: Option<PathBuf>,

    /// The `model` id reported by `/v1/models` and accepted by requests.
    /// Defaults to the checkpoint's Hub id when it is a HF-cache snapshot.
    #[arg(long)]
    pub model_name: Option<String>,

    /// KV positions per request (prompt + generation). A request past it is
    /// refused rather than reallocated. The standard pipette ladder tops out at
    /// 8192 + 256.
    #[arg(long, default_value_t = 8704)]
    pub kv_capacity: usize,

    /// Skip the pipeline warmup at load (the first prompt then compiles the
    /// prefill pipelines inside its own TTFT).
    #[arg(long)]
    pub no_warmup: bool,

    /// Force the CPU device.
    #[arg(long)]
    pub cpu: bool,

    /// The fraction of Metal's `recommendedMaxWorkingSetSize` this process
    /// may claim: LFM2's `MemoryBudget` admission (§9.5d — the KV ceiling's
    /// cache, the tile's activations and the weights predicted at load and
    /// refused if they do not fit; a KV grow past it refused at the grow) and
    /// the streamed expert tier's slot budget on Qwen4-exp (`--slots auto`).
    /// `0` turns LFM2's admission off — the unguarded arm.
    #[arg(long, default_value_t = 0.65)]
    pub memory_fraction: f64,

    /// Qwen4-exp: tokenizer.json (with chat_template.jinja beside it).
    /// Default: the HF cache's Qwen/Qwen3.8-Flash-Next snapshot.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Qwen4-exp: how the routed experts are held — `stream` (a slot pool
    /// per layer filled from the checkpoint on demand, the tier that fits
    /// this machine; `expert-streaming.md` §20) or `resident` (every expert
    /// in memory).
    #[arg(long, default_value = "stream")]
    pub experts: String,

    /// Qwen4-exp, `--experts stream`: streamed experts per layer per
    /// projection — a count, or `auto` (sized from the device's working set
    /// at `--memory-fraction`).
    #[arg(long, default_value = "auto")]
    pub slots: String,

    /// Qwen4-exp, `--experts stream`: the loader primitive, `pread` (read +
    /// compute-encoder copy) or `io` (`MTLIOCommandQueue`).
    #[arg(long, default_value = "pread")]
    pub loader: String,

    /// Qwen4-exp: rows of the N-gram PLE hot-row cache (0 = read the memory
    /// map every gather).
    #[arg(long, default_value_t = 0)]
    pub ple_cache_rows: usize,

    /// Qwen4-exp: load without the N-gram PLE (a measurement arm; the
    /// digests belong to the PLE-on model).
    #[arg(long)]
    pub no_ple: bool,

    /// Qwen4-exp: the routed experts' activation dtype, `f32` (the
    /// byte-identical routed arm) or `f16`.
    #[arg(long, default_value = "f32")]
    pub moe_act: String,

    /// Qwen4-exp `--experts stream`: the expert hotlist file — read at load
    /// to warm every layer's pool before the first request (§20.3 R2; a
    /// missing file warms nothing), written at shutdown with the usage
    /// since load merged in.
    #[arg(long)]
    pub hotlist: Option<PathBuf>,

    /// Qwen4-exp `--experts stream`: the expert prefetcher (the streamed
    /// tier's reads off the encoding thread, ahead of the layer; §20.4).
    #[arg(long, num_args = 0..=1, default_missing_value = "true", default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub prefetch: bool,

    /// Qwen4-exp `--experts stream --loader io`: the in-flight residency
    /// hints into free slots (measured not to pay on a pool smaller than the
    /// tile; off).
    #[arg(long, num_args = 0..=1, default_missing_value = "true", default_value_t = false, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub io_hints: bool,

    /// Qwen4-exp: how a prompt is prefilled — `chunk` (the chunked DeltaNet /
    /// QSA passes) or `replay` (the decode recurrence replayed per token; a
    /// reference arm).
    #[arg(long, default_value = "chunk", value_name = "chunk|replay")]
    pub prefill_mode: String,

    /// Qwen4-exp `--experts stream --hotlist`: pin this many experts per
    /// layer from the hotlist (never evicted); 0 = none.
    #[arg(long, default_value_t = 0)]
    pub pin: usize,

    /// Qwen4-exp `--slots auto`: KV tokens the slot budget reserves for
    /// (default: `--kv-capacity`).
    #[arg(long)]
    pub kv_reserve_tokens: Option<usize>,

    /// Qwen4-exp: the pass-keyed plan cache (DESIGN §9.2aa) on the Qwen path.
    #[arg(long)]
    pub pass_plans: bool,
}

/// The serving shape the family builds for: `serve`'s width and cells, or
/// `chat`'s one sequence with prefix reuse.
pub struct Shape {
    /// The decode loop's options (`--readback`, the cadence, the prefill).
    pub decode: crate::DecodeConfig,
    /// The model's axes: the cache preset and the per-axis overrides.
    pub model: crate::ModelConfig,
    /// The speculative body.
    pub speculative: crate::SpeculativeConfig,
    pub reuse_prefix: bool,
    pub seed_text: String,
    pub batch: usize,
    pub mem_sample: bool,
    pub rows: String,
    pub share_prefix: bool,
}

/// What `--model` names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Family {
    /// A safetensors checkpoint directory.
    Lfm2(PathBuf),
    /// A GGUF's first shard.
    Qwen4Exp(PathBuf),
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
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(Family::Qwen4Exp(path));
        }
        anyhow::bail!(
            "{}: a model is a checkpoint directory (LFM2) or a GGUF shard (Qwen4-exp)",
            path.display()
        );
    }
    if !path.is_dir() {
        anyhow::bail!("{}: no such model", path.display());
    }
    if path.join("config.json").is_file() {
        return Ok(Family::Lfm2(path));
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(&path)
        .with_context(|| format!("reading {}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    ggufs.sort();
    // The first shard of a sharded checkpoint, else the only file.
    if let Some(first) = ggufs
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains("-00001-of-"))
                .unwrap_or(false)
        })
        .or_else(|| (ggufs.len() == 1).then(|| &ggufs[0]))
    {
        return Ok(Family::Qwen4Exp(first.clone()));
    }
    anyhow::bail!(
        "{}: neither a checkpoint directory (config.json) nor a GGUF shard directory",
        path.display()
    )
}

/// Load the engine `args` names for `shape`. Returns it with the path it
/// was loaded from.
pub fn load_engine(model: &ModelArgs, shape: Shape) -> Result<(Box<dyn Served>, PathBuf)> {
    let family = detect_family(model.model.as_deref())?;
    let prefill_tile = shape.decode.prefill.tile;
    let prefill_chunk = shape.decode.prefill.chunk;
    let readback = shape.decode.readback.arm();
    match family {
        Family::Lfm2(dir) => {
            let batched = shape.batch > 1;
            let rows = lfm2::RowsArm::parse(&shape.rows, batched)?;
            let spec = &shape.speculative;
            if spec.method == "mtp" {
                anyhow::bail!("speculative.method = mtp (`--mtp`) is Qwen4-exp's; an LFM2 checkpoint has no MTP head — LFM2 speculates with `--dspark`");
            }
            let engine = lfm2::Engine::load(lfm2::LoadOptions {
                model_dir: dir.clone(),
                cpu: model.cpu,
                kv_capacity: model.kv_capacity,
                readback,
                decode: shape.decode.clone(),
                model: shape.model.clone(),
                reuse_prefix: shape.reuse_prefix,
                warmup: !model.no_warmup,
                seed_text: shape.seed_text,
                model_name: model.model_name.clone(),
                batch: shape.batch,
                dspark: (spec.method == "dspark").then(|| spec.drafter.clone().unwrap_or_default()),
                prefill_tile,
                prefill_chunk,
                memory_fraction: model.memory_fraction,
                mem_sample: shape.mem_sample,
                rows,
                share_prefix: shape.share_prefix,
                dspark_k: spec.k,
            })?;
            Ok((Box::new(engine), dir))
        }
        Family::Qwen4Exp(first) => {
            if !shape.model.axes.is_empty() || shape.model.cache != "auto" {
                anyhow::bail!(
                    "the model axes ({}{}) are LFM2's `Config`'s; Qwen4-exp's arms are `--experts`, `--slots`, `--loader`, `--moe-act`, `--prefill-mode`, `--prefetch`, `--io-hints` and the `tuning.qwen4exp.*` knobs",
                    if shape.model.cache != "auto" { format!("cache={} ", shape.model.cache) } else { String::new() },
                    shape.model.axes.keys().cloned().collect::<Vec<_>>().join(", ")
                );
            }
            let spec = &shape.speculative;
            let mtp = match spec.method.as_str() {
                "mtp" => match &spec.drafter {
                    Some(path) => Some(path.clone()),
                    None => anyhow::bail!("speculative.method = mtp needs the MTP sidecar GGUF (`--mtp <path>` / `speculative.drafter`)"),
                },
                "dspark" => anyhow::bail!("speculative.method = dspark (`--dspark`) is LFM2's drafter; Qwen4-exp speculates with `--mtp`"),
                _ => None,
            };
            if shape.batch > 1 && mtp.is_some() {
                anyhow::bail!(
                    "--batch {} with --mtp on Qwen4-exp: the batched loop is the K = 1 driver; \
                     the MTP head speculates at --batch 1",
                    shape.batch
                );
            }
            let engine = qwen4exp::Engine::load(qwen4exp::LoadOptions {
                first_shard: first.clone(),
                tokenizer: model.tokenizer.clone(),
                cpu: model.cpu,
                kv_capacity: model.kv_capacity,
                readback,
                decode: shape.decode.clone(),
                reuse_prefix: shape.reuse_prefix,
                warmup: !model.no_warmup,
                seed_text: shape.seed_text,
                model_name: model.model_name.clone(),
                prefill_tile,
                memory_fraction: model.memory_fraction,
                mem_sample: shape.mem_sample,
                experts: model.experts.clone(),
                slots: model.slots.clone(),
                loader: model.loader.clone(),
                ple_cache_rows: model.ple_cache_rows,
                ple: !model.no_ple,
                moe_act: model.moe_act.clone(),
                mtp,
                hotlist: model.hotlist.clone(),
                batch: shape.batch,
                mtp_k: spec.k.unwrap_or(1),
                prefetch: model.prefetch,
                io_hints: model.io_hints,
                prefill_mode: model.prefill_mode.clone(),
                pin: model.pin,
                kv_reserve_tokens: model.kv_reserve_tokens,
                pass_plans: model.pass_plans,
            })?;
            Ok((engine, first))
        }
    }
}

/// The option layers every subcommand takes (`lloom_engine::options`):
/// a config file, the JSON groups, the dotted overrides, and the shorthand
/// flags generated from the groups — applied in that order onto the
/// defaults and the env aliases.
#[derive(Args, Clone, Debug, Default)]
pub struct CommonArgs {
    /// A JSON file of option groups (`{"decode": {...}}`) — what `lloom config`
    /// prints, read back. Applied over the defaults and the env aliases.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// The decode group as JSON (`'{"readback": "ring-hostdrain", "in_flight": 2}'`).
    #[arg(long, value_name = "JSON")]
    pub decode_config: Option<String>,

    /// The tuning group as JSON (`'{"qmv": {"q6k_nr0": 4}, "metal": {"cut_policy": "node-aligned"}}'`).
    #[arg(long, value_name = "JSON")]
    pub tuning_config: Option<String>,

    /// The model group as JSON (`'{"cache": "crown", "axes": {"attn": "flash", "kv-append": "paged"}}'`).
    #[arg(long, value_name = "JSON")]
    pub model_config: Option<String>,

    /// The speculative group as JSON (`'{"method": "mtp", "k": 2, "drafter": "…gguf"}'`).
    #[arg(long, value_name = "JSON")]
    pub speculative_config: Option<String>,

    /// The sampling group as JSON (`'{"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}'`; vLLM's name).
    #[arg(long, value_name = "JSON")]
    pub generation_config: Option<String>,

    /// LFM2: the speculative body (`speculative.method = dspark`): a DSpark
    /// drafter proposes a block per round, the target verifies it in one
    /// pass. Loads the newest LFM2.5-2.6B-DSpark snapshot in the HF cache,
    /// or `--draft-dir`.
    #[arg(long)]
    pub dspark: bool,

    /// LFM2, under `--dspark`: the drafter checkpoint directory (`speculative.drafter`).
    #[arg(long, value_name = "DIR")]
    pub draft_dir: Option<PathBuf>,

    /// LFM2, under `--dspark`: propose this many positions per round instead
    /// of the drafter's block width (9), clamped to it (`speculative.k`).
    #[arg(long)]
    pub dspark_k: Option<usize>,

    /// Qwen4-exp: the GGUF carrying the MTP head (`blk.{n_layers}.*` +
    /// `nextn.*`, the sidecar `qwen4exp-mtp-quantize write` produces);
    /// `speculative.method = mtp`, `speculative.drafter`.
    #[arg(long, value_name = "GGUF")]
    pub mtp: Option<PathBuf>,

    /// Qwen4-exp `--mtp`: tokens the drafter proposes per round (`speculative.k`).
    #[arg(long)]
    pub mtp_k: Option<usize>,

    /// One knob by dotted key, repeatable (`--set decode.readback.eos=gpu-batched`).
    #[arg(long, value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub set: Vec<String>,

    #[command(flatten)]
    pub options: crate::options::EngineOptionsArgs,
}

/// Resolve the option tree from `common`: defaults < env aliases <
/// `--config` < `--<group>-config` < `--set` < flags; every group's refusals
/// checked once at the end. The tuning knobs are installed into both library
/// crates here — before the device or the model exists, so every `OnceLock`
/// read site takes them.
pub fn resolve_options(common: &CommonArgs) -> Result<crate::EngineOptions> {
    let o = resolve_options_uninstalled(common)?;
    o.tuning.install().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(o)
}

/// [`resolve_options`] without installing the tuning knobs (`lloom config`,
/// `lloom knobs`, the tests — nothing that goes on to load a model).
pub fn resolve_options_uninstalled(common: &CommonArgs) -> Result<crate::EngineOptions> {
    let mut o = crate::EngineOptions::default();
    o.from_env().map_err(|e| anyhow::anyhow!("env: {e}"))?;
    if let Some(path) = &common.config {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading --config {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("--config {}: not JSON", path.display()))?;
        o.merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--config {}: {e}", path.display()))?;
    }
    if let Some(json) = &common.decode_config {
        let v: serde_json::Value =
            serde_json::from_str(json).context("--decode-config: not JSON")?;
        o.decode
            .merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--decode-config: {e}"))?;
    }
    if let Some(json) = &common.tuning_config {
        let v: serde_json::Value =
            serde_json::from_str(json).context("--tuning-config: not JSON")?;
        o.tuning
            .merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--tuning-config: {e}"))?;
    }
    if let Some(json) = &common.model_config {
        let v: serde_json::Value =
            serde_json::from_str(json).context("--model-config: not JSON")?;
        o.model
            .merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--model-config: {e}"))?;
    }
    if let Some(json) = &common.speculative_config {
        let v: serde_json::Value =
            serde_json::from_str(json).context("--speculative-config: not JSON")?;
        o.speculative
            .merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--speculative-config: {e}"))?;
    }
    if let Some(json) = &common.generation_config {
        let v: serde_json::Value =
            serde_json::from_str(json).context("--generation-config: not JSON")?;
        o.sampling
            .merge_json(&v)
            .map_err(|e| anyhow::anyhow!("--generation-config: {e}"))?;
    }
    for kv in &common.set {
        let (k, v) = lloom_tensor::options::split_assignment(kv).map_err(|e| anyhow::anyhow!("--set {e}"))?;
        o.set(k, v).map_err(|e| anyhow::anyhow!("--set {kv}: {e}"))?;
    }
    common
        .options
        .apply(&mut o)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // The speculation shorthands (`lfm2-chat` / `qwen4exp-chat` spellings),
    // over the group.
    if common.dspark && common.mtp.is_some() {
        anyhow::bail!("--dspark is LFM2's drafter and --mtp Qwen4-exp's; one speculative body per run");
    }
    if common.dspark {
        o.speculative.method = "dspark".into();
    }
    if let Some(dir) = &common.draft_dir {
        o.speculative.drafter = Some(dir.clone());
    }
    if let Some(k) = common.dspark_k {
        o.speculative.k = Some(k);
    }
    if let Some(path) = &common.mtp {
        o.speculative.method = "mtp".into();
        o.speculative.drafter = Some(path.clone());
    }
    if let Some(k) = common.mtp_k {
        o.speculative.k = Some(k);
    }
    o.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(o)
}

/// `lloom config`: the resolved option tree as JSON — every knob with the
/// value this invocation would run, in the shape `--config` reads back.
#[derive(Parser)]
#[command(name = "lloom config", about = "Print the resolved option tree as JSON")]
pub struct ConfigArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

/// `lloom knobs`: every knob of every group — key, flag, env alias, type,
/// arms, the default, the doc — one line each.
pub fn print_knobs() -> Result<()> {
    let defaults = crate::EngineOptions::default();
    let value = serde_json::to_value(&defaults)?;
    let mut rows = lloom_tensor::options::rows("", crate::EngineOptions::META, &value);
    // The model axes come from `lfm2::Config`'s own derive: one row each,
    // the default rendered from a default config.
    let probe = lloom::models::lfm2::Config::default();
    let line = probe.axes_line();
    for m in lloom::models::lfm2::Config::AXES_META {
        let value = line
            .split(' ')
            .find_map(|kv| kv.strip_prefix(&format!("{}=", m.ty)))
            .unwrap_or("")
            .to_string();
        rows.push(lloom_tensor::options::KnobRow {
            key: format!("model.axes.{}", m.key),
            flag: format!("--{}", m.flag),
            env: None,
            ty: m.ty,
            variants: m.variants,
            doc: m.doc,
            value,
        });
    }
    for r in rows {
        let env = r.env.map(|e| format!(" env={e}")).unwrap_or_default();
        let arms = if r.variants.is_empty() {
            String::new()
        } else {
            format!(" arms=[{}]", r.variants.join(", "))
        };
        println!(
            "{key}  {flag} ({ty}) default={value}{env}{arms}\n    {doc}",
            key = r.key,
            flag = r.flag,
            ty = r.ty,
            value = r.value,
            doc = r.doc,
        );
    }
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
    /// Print the resolved option tree (the layers applied) as JSON.
    Config(ConfigArgs),
    /// List every knob: key, flag, env alias, default, arms, doc.
    Knobs,
}

/// The `lloom` binary's entry.
pub fn main(argv: Vec<String>) -> Result<()> {
    let cli = Cli::parse_from(argv);
    match cli.command {
        Command::Serve(args) => serve::run(args),
        Command::Chat(args) => chat::run(args),
        Command::Config(args) => {
            let o = resolve_options_uninstalled(&args.common)?;
            println!("{}", serde_json::to_string_pretty(&o)?);
            Ok(())
        }
        Command::Knobs => print_knobs(),
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
        let o = resolve_options_uninstalled(&common(&[
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
        ]))
        .unwrap();
        assert_eq!(o.decode.readback.eos, EosKind::HostDrain, "from the file");
        assert_eq!(o.decode.readback.lookahead, 0, "from the file");
        assert_eq!(o.decode.readback.wake_cadence, 2, "from --decode-config");
        assert_eq!(o.decode.readback.kind, ReadbackKind::Ring, "from --set");
        assert_eq!(o.decode.in_flight, 7, "the flag wins over --set over the json over the file");
        // The dump is a valid --config that reproduces the tree.
        let dumped = dir.join("dump.json");
        std::fs::write(&dumped, serde_json::to_string_pretty(&o).unwrap()).unwrap();
        let back = resolve_options_uninstalled(&common(&["--config", dumped.to_str().unwrap()])).unwrap();
        assert_eq!(back, o);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_refused_combination_names_itself() {
        // The blocking arm ignores the cadence knobs rather than refusing them.
        let o = resolve_options_uninstalled(&common(&["--readback", "blocking", "--adaptive-commit"])).unwrap();
        assert_eq!(o.decode.readback.arm(), crate::ReadbackArm::Blocking);
        let err = resolve_options_uninstalled(&common(&["--eos-gate", "--adaptive-commit=false"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("eos_gate needs adaptive_commit"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--set", "decode.nope=1"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key `nope`"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--set", "decode.readback"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected key=value"), "{err}");
    }

    #[test]
    fn the_shorthands_are_lfm2_chats() {
        let o = resolve_options_uninstalled(&common(&["--readback", "ring-gpubatch-nonblocking"])).unwrap();
        assert_eq!(o.decode.readback.kind, ReadbackKind::Ring);
        assert_eq!(o.decode.readback.eos, EosKind::GpuBatchedNonblocking);
        let o = resolve_options_uninstalled(&common(&["--readback", "ring-la0"])).unwrap();
        assert_eq!(o.decode.readback.lookahead, 0);
        assert!(!o.decode.readback.adaptive_commit);
        // The compound and its keys compose in either order: the spelling
        // sets kind + eos, the key flags override field by field.
        let o = resolve_options_uninstalled(&common(&["--readback", "ring-hostdrain", "--lookahead", "0"])).unwrap();
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
    fn the_axes_reach_the_config_by_flag_key_and_json() {
        let o = resolve_options_uninstalled(&common(&[
            "--cache",
            "crown",
            "--attn",
            "flash",
            "--gemv-silu=false",
            "--set",
            "model.kv-append=paged",
            "--model-config",
            r#"{"axes": {"qkv-layout": "fused"}}"#,
        ]))
        .unwrap();
        assert_eq!(o.model.cache, "crown");
        assert_eq!(o.model.axes.get("attn").map(String::as_str), Some("flash"));
        assert_eq!(o.model.axes.get("gemv-silu").map(String::as_str), Some("false"));
        assert_eq!(o.model.axes.get("kv-append").map(String::as_str), Some("paged"));
        assert_eq!(o.model.axes.get("qkv-layout").map(String::as_str), Some("fused"));
        let mut cfg = Config::default();
        cfg.gemv_silu = true;
        o.model.apply_axes(&mut cfg).unwrap();
        assert_eq!(cfg.attn_impl, AttnImpl::FlashDecoding);
        assert_eq!(cfg.kv_append, KvAppend::Paged);
        assert!(!cfg.gemv_silu, "`--gemv-silu=false` forces the preset's arm off");
        assert!(cfg.axes_line().contains("AttnImpl=FlashDecoding"), "{}", cfg.axes_line());
        // The dump round-trips.
        let v = serde_json::to_value(&o).unwrap();
        let mut back = crate::EngineOptions::default();
        back.merge_json(&v).unwrap();
        assert_eq!(back, o);
    }

    #[test]
    fn a_bad_axis_or_preset_is_refused_at_parse() {
        let err = resolve_options_uninstalled(&common(&["--attn", "bogus"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--attn"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--cache", "shiny"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("cache:"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--set", "model.nope=1"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown axis `nope`"), "{err}");
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

    #[test]
    fn the_sampler_arms_follow_the_fields() {
        let o = resolve_options_uninstalled(&common(&[])).unwrap();
        assert_eq!(o.sampling.sampling(), Sampling::ArgMax, "greedy by default");
        let o = resolve_options_uninstalled(&common(&["-t", "0.7", "--top-p", "0.9", "-n", "64", "--seed", "7"])).unwrap();
        assert_eq!(o.sampling.sampling(), Sampling::TopP { p: 0.9, temperature: 0.7 });
        assert_eq!(o.sampling.max_tokens, 64);
        assert_eq!(o.sampling.seed, Some(7));
        let o = resolve_options_uninstalled(&common(&["-t", "0.7", "--top-k", "40", "--top-p", "0.9"])).unwrap();
        assert_eq!(o.sampling.sampling(), Sampling::TopKThenTopP { k: 40, p: 0.9, temperature: 0.7 });
        let o = resolve_options_uninstalled(&common(&["-t", "0.7", "--gumbel"])).unwrap();
        assert_eq!(o.sampling.sampling(), Sampling::GumbelSoftmax { temperature: 0.7 });
        let o = resolve_options_uninstalled(&common(&["--generation-config", r#"{"temperature": 0.3, "stop": ["<|im_end|>", "\n\n"], "ignore_eos": true}"#])).unwrap();
        assert_eq!(o.sampling.sampling(), Sampling::All { temperature: 0.3 });
        assert_eq!(o.sampling.stop, vec!["<|im_end|>", "\n\n"]);
        let p = o.sampling.params(&[7], crate::ReadbackArm::Blocking, 0);
        assert_eq!(p.eos_ids, vec![u32::MAX], "ignore_eos replaces the EOS set with an id no tokenizer emits");
        assert_eq!(p.max_tokens, 512);
        // The checkpoint's defaults fill only unset fields, only when asked.
        let gc = lloom::hub::GenerationConfig {
            temperature: Some(0.6),
            top_p: Some(0.95),
            ..Default::default()
        };
        let mut s = o.sampling.clone();
        s.apply_checkpoint(Some(&gc));
        assert_eq!(s.temperature, Some(0.3), "greedy mode ignores the checkpoint");
        s.defaults = "checkpoint".into();
        s.apply_checkpoint(Some(&gc));
        assert_eq!(s.temperature, Some(0.3), "a set field stays");
        assert_eq!(s.top_p, Some(0.95), "an unset one takes the checkpoint's");
    }

    #[test]
    fn the_speculation_shorthands_fill_the_group() {
        let o = resolve_options_uninstalled(&common(&["--dspark", "--dspark-k", "3"])).unwrap();
        assert_eq!(o.speculative.method, "dspark");
        assert_eq!(o.speculative.k, Some(3));
        assert_eq!(o.speculative.drafter, None);
        let o = resolve_options_uninstalled(&common(&["--mtp", "/tmp/mtp.gguf", "--mtp-k", "2"])).unwrap();
        assert_eq!(o.speculative.method, "mtp");
        assert_eq!(o.speculative.k, Some(2));
        assert_eq!(o.speculative.drafter.as_deref(), Some(std::path::Path::new("/tmp/mtp.gguf")));
        let o = resolve_options_uninstalled(&common(&["--speculative-config", r#"{"method": "dspark", "k": 4, "drafter": "/d"}"#])).unwrap();
        assert_eq!((o.speculative.method.as_str(), o.speculative.k), ("dspark", Some(4)));
        let err = resolve_options_uninstalled(&common(&["--dspark", "--mtp", "/x"])).unwrap_err().to_string();
        assert!(err.contains("one speculative body"), "{err}");
        let err = resolve_options_uninstalled(&common(&["--spec-method", "ngram"])).unwrap_err().to_string();
        assert!(err.contains("not a speculation method"), "{err}");
    }
}
