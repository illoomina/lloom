//! The Qwen3.5 engine: one sequence over the dense (`qwen35`) or the
//! resident-MoE (`qwen35moe`) model on the generic decoder
//! (`lloom::models::qwen3_5`), driven by the same `DecodeSession` the other
//! families' engines drive. `speculative.method = mtp` puts the checkpoint's
//! MTP block (`nn::mtp`, in the main file or the sidecar `speculative.model`
//! names) on the speculative body, at the `Ring × One` shape: one drafted
//! token a round (`speculative.k` more), verified by the target,
//! byte-identical to greedy. No rows (`--batch N`), no streamed expert
//! tier — each refused by name in `family.rs`.

use super::common::{
    model_name_from_dir, ms, newest_snapshot, open_device, DecodeTiming, MemGauge, MemReport,
    PrefillTiming, Text, NEVER_EOS,
};
use super::request_slot::{AbortState, Lease as RequestLease, State as RequestState};
use super::served::Served;
use super::template::{Rendered, Template};
use crate::{DecodeParams, DecodeSession, Decoded, ReadbackArm, RingPool};
use anyhow::{Context, Result};
use lloom::load_warning::LoadWarning;
use lloom::models::qwen3_5::{arch_of, Config, LoadOptions as ModelLoad, Qwen3_5, Qwen3_5Moe};
use lloom::nn::decoder::{Decoder, Layer, NormHead, StateCapacity, TokenEmbed};
use lloom::nn::model::{CacheReset, CacheRewind, CausalLM};
use lloom::nn::mtp::{NextnDrafter, NextnHead, NextnLanes};
use lloom::nn::speculator::NextnHidden;
use lloom::nn::{Probes, RmsNorm, Sequence};
use lloom::profile::{Profile, ProfileKey, RewindProof, Shaped};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Tensor, D};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

pub struct LoadOptions {
    pub first_shard: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub cpu: bool,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    pub decode: crate::DecodeConfig,
    pub reuse_prefix: bool,
    pub warmup: bool,
    pub seed_text: String,
    pub model_name: Option<String>,
    pub prefill_tile: usize,
    pub memory_fraction: f64,
    pub mem_sample: bool,
    pub executor: crate::ExecutorConfig,
    pub memory: crate::MemoryConfig,
    pub kv_precision: Option<lloom::nn::ops::KvPrecision>,
    /// `--experts` as asked; the MoE loads resident whatever it says, and
    /// says so.
    pub experts: String,
    /// `speculative.method = mtp`: decode through the speculative body on
    /// the checkpoint's MTP block.
    pub mtp: bool,
    /// `speculative.model`: the MTP sidecar GGUF (bartowski's `mtp-*.gguf`)
    /// when the main file does not keep the block.
    pub mtp_sidecar: Option<PathBuf>,
    /// `speculative.k`: tokens drafted a round.
    pub mtp_k: usize,
}

/// A Qwen3.5 model as the engine drives it: a `Decoder` of either alias,
/// with the shared tables its MTP drafter reads.
pub trait Qwen3_5Model:
    CausalLM<Config = Config> + Probes<NextnHidden> + Shaped + Send + Sync + 'static
{
    fn embed_table(&self) -> &QMatMul;
    fn lm_head(&self) -> &QMatMul;
    fn capacity(&self) -> StateCapacity;
    /// The bytes one turn checkpoint of a state takes (`Decoder::checkpoint_frame_bytes`).
    fn checkpoint_frame_bytes(&self) -> usize;
    /// Retain `h_nextn` on every forward of `cache` (the drafter's seeding
    /// reads the prefill's rows).
    fn arm_nextn(cache: &mut Self::Cache);
}

impl<P: Profile, L: Layer + 'static> Qwen3_5Model
    for Decoder<P, Config, TokenEmbed, L, NormHead<RmsNorm>>
{
    fn embed_table(&self) -> &QMatMul {
        self.embed_block().table()
    }
    fn lm_head(&self) -> &QMatMul {
        self.head_block().proj()
    }
    fn capacity(&self) -> StateCapacity {
        Decoder::capacity(self)
    }
    fn checkpoint_frame_bytes(&self) -> usize {
        Decoder::checkpoint_frame_bytes(self)
    }
    fn arm_nextn(cache: &mut Self::Cache) {
        cache.set_want_nextn(true);
    }
}

/// Side state travels with the target sequence under one request checkout.
struct SingleExtra<X: Layer> {
    decode: crate::DecodeConfig,
    lanes: Option<NextnLanes<X::State>>,
}
impl<X: Layer> AbortState for SingleExtra<X> {
    fn abort(&mut self) {
        if let Some(lanes) = &mut self.lanes {
            NextnDrafter::<X>::reset_lanes(lanes);
        }
    }
}

/// Preserve the drafter's complete lane state even if a callback unwinds. The
/// outer request lease marks incomplete target/lane reuse; the next started
/// request resets both before execution, outside callback unwind.
struct DrafterLease<'m, 's, X: Layer> {
    slot: &'s mut Option<NextnLanes<X::State>>,
    drafter: Option<NextnDrafter<'m, X>>,
}
impl<X: Layer> Drop for DrafterLease<'_, '_, X> {
    fn drop(&mut self) {
        if let Some(drafter) = self.drafter.take() {
            *self.slot = Some(drafter.into_lanes());
        }
    }
}

/// The engine over a model `M` of this family (the dense or the MoE
/// alias) and its MTP block's layer type `X`.
pub struct Engine<M: Qwen3_5Model, X: Layer> {
    model: Arc<M>,
    /// The MTP head, when the run decodes through the speculative body.
    mtp: Option<NextnHead<X>>,
    mtp_k: usize,
    device: Device,
    text: Arc<Text>,
    model_name: String,
    kv_capacity: usize,
    readback: ReadbackArm,
    executor: crate::ExecutorConfig,
    installed: super::report::Installed,
    reuse_prefix: bool,
    prefill_tile: usize,
    mem_sample: bool,
    arch: &'static str,
    tier: Value,
    load_warnings: Vec<LoadWarning>,
    single: Mutex<Option<RequestState<M::Cache, SingleExtra<X>>>>,
}

/// Request-level family choice; each variant retains a statically dispatched
/// decoder and one loaded model. No enum dispatch occurs inside the token loop.
pub enum Engines<P: Profile> {
    Dense(Engine<Qwen3_5<P>, lloom::models::qwen3_5::Qwen3_5MtpLayer<P>>),
    Moe(Engine<Qwen3_5Moe<P>, lloom::models::qwen3_5::Qwen3_5MoeMtpLayer<P>>),
}

impl<P: Profile> Engines<P> {
    fn into_served(self) -> Box<dyn Served> {
        match self {
            Self::Dense(engine) => Box::new(engine),
            Self::Moe(engine) => Box::new(engine),
        }
    }

    /// Existing chat execution inside this already-loaded owner's pass. The
    /// caller supplies a whole asynchronous peak, not just one tile's scratch.
    pub fn complete_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        prompt_ids: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        Ok(loaded.with_pass(predicted_extra, |engine, _context| {
            let out = match engine {
                Self::Dense(engine) => engine.complete_until(prompt_ids, params, &mut on_fragment),
                Self::Moe(engine) => engine.complete_until(prompt_ids, params, &mut on_fragment),
            };
            out.map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
        })?)
    }

    /// Common retained-trunk decisions on the same weights as chat. Prepared
    /// request storage retains its origin; fresh cache/snapshot/readout demand
    /// belongs to this pass. Default server activation still requires complete
    /// predictor/producer enrollment and verified tokenizer state boundaries.
    pub fn decide_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        request: &crate::decision::CausalRequest,
    ) -> Result<crate::decision::SystemOneResponse> {
        Ok(
            loaded.with_pass(predicted_extra, |engine, _context| match engine {
                Self::Dense(engine) => engine.decide_prepared(request),
                Self::Moe(engine) => engine.decide_prepared(request),
            })?,
        )
    }
}

#[derive(Clone, Copy)]
enum LoadAdmission {
    Legacy,
    CallerScoped,
}

/// Load the complete app engine inside one caller-supplied model domain, including
/// header/tokenizer metadata, model/MTP, initial cache/rings and warmup. No nested
/// legacy admission or model-only Loaded wrapper is introduced. Source-only
/// activation: ordinary/native/host producer enrollment remains prerequisite.
pub fn load_admitted_as<P: Profile>(
    opts: LoadOptions,
    device: &Device,
    load: lloom::nn::memory::LoadContext,
) -> Result<lloom::nn::memory::Loaded<Engines<P>>> {
    Ok(load_admitted_with::<P, _>(opts, device, load, |_| Ok(()))?.0)
}

fn load_admitted_with<P: Profile, T>(
    opts: LoadOptions,
    device: &Device,
    load: lloom::nn::memory::LoadContext,
    capture: impl FnOnce(&Engines<P>) -> Result<T>,
) -> Result<(lloom::nn::memory::Loaded<Engines<P>>, T)> {
    lloom_tensor::memory_admission::begin_model_load();
    if opts.cpu != device.is_cpu() {
        anyhow::bail!("supplied Qwen3.5 device disagrees with load options");
    }
    let mut metadata = None;
    let loaded = lloom::nn::memory::Loaded::new(device, load, |resolved, _context| {
        let engine = load_on::<P>(opts, resolved.clone(), LoadAdmission::CallerScoped)
            .map_err(|error| lloom_tensor::Error::msg(format!("{error:#}")))?;
        metadata =
            Some(capture(&engine).map_err(|error| lloom_tensor::Error::msg(format!("{error:#}")))?);
        Ok(engine)
    })?;
    Ok((
        loaded,
        metadata.expect("successful admitted load ran metadata capture"),
    ))
}

/// The GGUF's `general.architecture`, when it is this family's.
pub fn arch_of_path(path: &Path) -> Option<&'static str> {
    let mut f = std::fs::File::open(path).ok()?;
    let content = Content::read(&mut f).ok()?;
    arch_of(&content)
}

/// The base model's Hub repo the quantizer recorded, as a cache directory
/// name (`general.base_model.0.repo_url` → `models--Qwen--Qwen3.6-27B`).
fn base_model_snapshot(content: &Content) -> Option<PathBuf> {
    let url = content
        .metadata
        .get("general.base_model.0.repo_url")
        .and_then(|v| v.to_string().ok())?;
    let repo = url.trim_end_matches('/').rsplit("huggingface.co/").next()?;
    newest_snapshot(&format!("models--{}", repo.replace('/', "--")))
}

/// Where the tokenizer and the chat template come from, in order: `--tokenizer`,
/// a `tokenizer.json` beside the shard, the base model's snapshot in the HF
/// cache, the GGUF's own vocabulary (with its embedded template).
fn resolve_text(
    opts: &LoadOptions,
    content: &Content,
) -> Result<(Tokenizer, Option<Template>, String)> {
    let beside = opts.first_shard.with_file_name("tokenizer.json");
    let path = opts
        .tokenizer
        .clone()
        .or_else(|| beside.is_file().then_some(beside))
        .or_else(|| base_model_snapshot(content).map(|d| d.join("tokenizer.json")))
        .filter(|p| p.is_file());
    match path {
        Some(p) => {
            let tokenizer = Tokenizer::from_file(&p)
                .map_err(|e| anyhow::anyhow!("loading {}: {e}", p.display()))?;
            let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
            let template = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
                Ok(src) => Some(Template::compile(&src)?),
                Err(_) => match content
                    .metadata
                    .get("tokenizer.chat_template")
                    .and_then(|v| v.to_string().ok())
                {
                    Some(src) => Some(Template::compile(src)?),
                    None => None,
                },
            };
            Ok((tokenizer, template, format!("{}", p.display())))
        }
        None => {
            use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
            let tokenizer = Tokenizer::from_gguf(content)
                .map_err(|e| anyhow::anyhow!("the GGUF's tokenizer: {e}"))?;
            let template = match content
                .metadata
                .get("tokenizer.chat_template")
                .and_then(|v| v.to_string().ok())
            {
                Some(src) => Some(Template::compile(src)?),
                None => None,
            };
            Ok((tokenizer, template, "the GGUF's own vocabulary".into()))
        }
    }
}

/// The same owned tokenizer/template/seed moves from cheap preparation to the
/// full engine; no model load precedes these configuration refusals.
struct PreparedText {
    text: Arc<Text>,
    model_name: String,
    source: String,
}
impl PreparedText {
    fn validate(&self, cfg: &Config) -> lloom_tensor::Result<()> {
        let text = &self.text;
        let outside = |id: u32| id as usize >= cfg.vocab_size;
        if cfg.vocab_size == 0
            || text
                .tokenizer
                .get_vocab(true)
                .values()
                .copied()
                .any(outside)
            || text.eos_ids.iter().copied().any(outside)
            || text.seed_tokens.iter().copied().any(outside)
        {
            lloom_tensor::bail!(
                "Qwen serving tokenizer/EOS/seed ids must fit the resolved vocabulary"
            )
        }
        if let Some(tags) = text.think_tags() {
            if [Some(tags.start), Some(tags.end), tags.tool_call]
                .into_iter()
                .flatten()
                .any(outside)
            {
                lloom_tensor::bail!(
                    "Qwen serving thinking marker ids must fit the resolved vocabulary"
                )
            }
        }
        for kwarg in [None, Some(true), Some(false)] {
            for want in [None, Some(true), Some(false)] {
                if text
                    .generation_prompt(kwarg, want)
                    .is_some_and(|ids| ids.iter().copied().any(outside))
                {
                    lloom_tensor::bail!(
                        "Qwen serving prompt probe ids must fit the resolved vocabulary"
                    )
                }
            }
        }
        Ok(())
    }
}
fn prepare_text(opts: &mut LoadOptions, content: &Content, arch: &str) -> Result<PreparedText> {
    if opts.kv_capacity == 0 {
        anyhow::bail!("Qwen serving context capacity must be positive")
    }
    let (tokenizer, template, source) = resolve_text(opts, content)?;
    let mut eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    eos_ids.dedup();
    if eos_ids.is_empty() {
        anyhow::bail!("the tokenizer has neither <|im_end|> nor <|endoftext|>");
    }
    let seed_tokens = tokenizer
        .encode(opts.seed_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizing the prompt seed: {e}"))?
        .get_ids()
        .to_vec();
    if seed_tokens.is_empty() {
        anyhow::bail!("the prompt seed tokenized to nothing");
    }
    let model_name = opts
        .model_name
        .take()
        .unwrap_or_else(|| match base_model_snapshot(content) {
            Some(dir) => model_name_from_dir(&dir),
            None => opts
                .first_shard
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| arch.to_string()),
        });
    let text = Text {
        tokenizer,
        template,
        bos_token: String::new(),
        bos_id: None,
        eos_ids,
        seed_tokens,
        think: Default::default(),
        gen_prompts: Default::default(),
    };
    text.prime();
    Ok(PreparedText {
        text: Arc::new(text),
        model_name,
        source,
    })
}

/// Detect, hold the regime, load — at `Ring × One` under
/// `speculative.method = mtp` (the verify window), `No × One` otherwise.
pub fn load(opts: LoadOptions) -> Result<Box<dyn Served>> {
    lloom_tensor::memory_admission::begin_model_load();
    use lloom::profile::{NoRewind, One, Rewind as _, Ring, Rows as _};
    let key = ProfileKey::at_contract(
        if opts.mtp { Ring::NAME } else { NoRewind::NAME },
        One::NAME,
    );
    lloom::profile::dispatch!(key, lloom::profile::QWEN3_5, P => load_as::<P>(opts))
}

/// The evidence the speculative driver takes: the `Ring` shape gives it,
/// the `No` shape refuses by name.
fn rewindable<M: Shaped>() -> Result<RewindProof<M::Profile>> {
    RewindProof::<M::Profile>::ask().with_context(|| {
        format!(
            "the speculative body on the {} shape: speculative.method = mtp runs on the `Ring` shape",
            M::Profile::KEY
        )
    })
}

fn load_as<P: Profile>(opts: LoadOptions) -> Result<Box<dyn Served>> {
    let device = if opts.cpu {
        Device::Cpu
    } else {
        open_device()?
    };
    load_on::<P>(opts, device, LoadAdmission::Legacy).map(Engines::into_served)
}

fn load_on<P: Profile>(
    mut opts: LoadOptions,
    device: Device,
    admission: LoadAdmission,
) -> Result<Engines<P>> {
    lloom_tensor::memory_admission::begin_model_load();
    let window = if opts.mtp {
        lloom::nn::speculator::checked_window_width(opts.mtp_k.max(1), 1)?
    } else {
        0
    };
    let shards = lloom::hub::gguf_shards(&opts.first_shard);
    let mut f = std::fs::File::open(&opts.first_shard)
        .with_context(|| format!("opening {}", opts.first_shard.display()))?;
    let content = Content::read(&mut f)
        .with_context(|| format!("reading the GGUF header of {}", opts.first_shard.display()))?;
    let arch = arch_of(&content).context("not a qwen35 / qwen35moe GGUF")?;
    drop(f);
    let text = prepare_text(&mut opts, &content, arch)?;
    let installed = super::common::install_executor(&device, &opts.executor, opts.mtp)?;
    #[cfg(feature = "metal")]
    if let Some(mb) = opts.memory.free_budget_mb {
        if let Device::Metal(m) = &device {
            m.set_free_budget(mb * 1024 * 1024);
        }
    }
    if let Some(p) = &opts.mtp_sidecar {
        if !p.is_file() {
            anyhow::bail!("speculative.model {}: no such file", p.display());
        }
    }
    // The ledger's quota: the fraction of the working set every pass-level
    // admission and the state's growth read against (§9.5c), set before
    // the load so the load's own allocations are on it.
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        let ws = m.device().recommended_max_working_set_size() as f64;
        m.ledger()
            .set_quota((ws * opts.memory_fraction.max(0.0)) as usize);
    }
    let mload = ModelLoad {
        kv_capacity: opts.kv_capacity,
        kv_precision: opts.kv_precision,
        memory_fraction: Some(opts.memory_fraction),
        dense_f32: false,
        mtp: opts.mtp,
        // The verify window of `k + 1` positions as one pass over the rings.
        window,
        machine: match admission {
            LoadAdmission::Legacy => Some(lloom::nn::memory::MachineLoad {
                gate: opts.memory.machine_gate(),
                label: super::common::machine_label(arch, &opts.first_shard),
                rows: 1,
                kv_tokens: opts.kv_capacity,
                prefill_tile: opts.prefill_tile,
            }),
            LoadAdmission::CallerScoped => None,
        },
        checkpoints: lloom::models::qwen3_5::LoadOptions::default().checkpoints,
    };
    let t = Instant::now();
    let mut warnings = Vec::new();
    match arch {
        lloom::models::qwen3_5::ARCH_DENSE | lloom::models::qwen3_5::ARCH_CLEF => {
            let (model, head): (Qwen3_5<P>, _) = Qwen3_5::from_gguf_shards_with_mtp_preflight(
                &shards,
                opts.mtp_sidecar.as_ref(),
                &device,
                &mload,
                |cfg| text.validate(cfg),
            )
            .context("loading Qwen3.5 (dense)")?;
            let tier = json!({ "arch": arch, "shards": shards.len(), "load_ms": ms(t.elapsed()) });
            Engine::finish(
                model, head, device, opts, text, arch, tier, installed, warnings,
            )
            .map(Engines::Dense)
        }
        _ => {
            if opts.experts != "resident" {
                let note = format!(
                    "--experts {}: the streamed expert tier is Qwen4-exp's; qwen3_5_moe loads every expert resident",
                    opts.experts
                );
                eprintln!("[experts: {note}]");
                warnings.push(LoadWarning::new(
                    lloom::load_warning::LoadWarningKind::Readback,
                    note,
                ));
            }
            let (model, head): (Qwen3_5Moe<P>, _) =
                Qwen3_5Moe::from_gguf_shards_with_mtp_preflight(
                    &shards,
                    opts.mtp_sidecar.as_ref(),
                    &device,
                    &mload,
                    |cfg| text.validate(cfg),
                )
                .context("loading Qwen3.5 (MoE, resident)")?;
            let tier = json!({ "arch": arch, "experts": "resident", "shards": shards.len(), "load_ms": ms(t.elapsed()) });
            Engine::finish(
                model, head, device, opts, text, arch, tier, installed, warnings,
            )
            .map(Engines::Moe)
        }
    }
}

impl<M, X> Engine<M, X>
where
    M: Qwen3_5Model,
    M::Cache: CacheReset + CacheRewind + Send + Sync + 'static,
    X: Layer + 'static,
    X::State: Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    fn finish(
        model: M,
        mtp: Option<NextnHead<X>>,
        device: Device,
        opts: LoadOptions,
        prepared: PreparedText,
        arch: &'static str,
        mut tier: Value,
        installed: super::report::Installed,
        load_warnings: Vec<LoadWarning>,
    ) -> Result<Self> {
        if opts.mtp && mtp.is_none() {
            anyhow::bail!("speculative.method = mtp was given but the loader found no MTP block");
        }
        let PreparedText {
            text,
            model_name,
            source,
        } = prepared;
        tier["tokenizer"] = json!(source);
        tier["first_shard"] = json!(opts.first_shard.display().to_string());
        tier["mtp"] = json!(mtp.is_some());
        if let Some(p) = &opts.mtp_sidecar {
            tier["mtp_sidecar"] = json!(p.display().to_string());
        }
        let mut cache = model.new_cache(model_config(&model), &device)?;
        if mtp.is_some() {
            M::arm_nextn(&mut cache);
        }
        // The turn checkpoints (`sequence-reuse-design.md` §2.3b): the ring
        // a request re-rendering the previous turn restores from, sized
        // from the recurrent layers' state and admitted beside the weights.
        let (slots, frame) = (model.capacity().checkpoints, model.checkpoint_frame_bytes());
        eprintln!(
            "[checkpoint ring: {slots} × {:.1} MiB ({:.1} MiB; 0 = every re-rendered turn prefills whole)]",
            frame as f64 / (1024.0 * 1024.0),
            (slots * frame) as f64 / (1024.0 * 1024.0)
        );
        tier["turn_checkpoints"] =
            json!({ "slots": slots, "frame_bytes": frame, "bytes": slots * frame });
        let engine = Engine {
            model: Arc::new(model),
            mtp,
            mtp_k: opts.mtp_k.max(1),
            installed,
            device,
            text,
            model_name,
            kv_capacity: opts.kv_capacity,
            readback: opts.readback,
            executor: opts.executor.clone(),
            reuse_prefix: opts.reuse_prefix,
            prefill_tile: opts.prefill_tile,
            mem_sample: opts.mem_sample,
            arch,
            tier,
            load_warnings,
            single: Mutex::new(Some(RequestState {
                sequence: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                extra: SingleExtra {
                    decode: opts.decode.clone(),
                    lanes: None,
                },
                needs_recovery: false,
            })),
        };
        if opts.warmup {
            let t = Instant::now();
            #[cfg(feature = "metal")]
            let _paused = match &engine.device {
                Device::Metal(m) => Some(m.pause_pass_plans()?),
                _ => None,
            };
            let (libs, pipes) = engine.with_single(|s, _extra| Ok(s.warmup()?))?;
            eprintln!(
                "[warmup: {libs} libraries, {pipes} pipelines compiled in {:.1} ms]",
                ms(t.elapsed())
            );
        }
        Ok(engine)
    }

    fn with_single<T>(
        &self,
        f: impl FnOnce(&mut DecodeSession<'_, M>, &mut SingleExtra<X>) -> Result<T>,
    ) -> Result<T> {
        let mut lease = RequestLease::<M, _>::checkout(&self.single, true)?;
        let decode = lease.extra().decode.clone();
        lease.start_decode(&self.model, &self.device, decode);
        let (session, extra) = lease.parts_mut();
        let out = f(session, extra);
        if out.is_ok() {
            lease.mark_complete();
        }
        out
    }

    fn decide_prepared(
        &self,
        request: &crate::decision::CausalRequest,
    ) -> lloom_tensor::Result<crate::decision::SystemOneResponse>
    where
        M::Cache: lloom::nn::branch::RetainState,
    {
        self.check_fits(request.max_prompt_tokens(), 0)
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))?;
        let lease = RequestLease::<M, _>::checkout(&self.single, true)?;
        let options = lease.extra().decode.options(0, ReadbackArm::Blocking, 0);
        let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, &self.device)?;
        request.execute(
            &*self.model,
            model_config(&*self.model),
            &mut readback,
            &options,
        )
    }

    fn detok(&self) -> impl FnMut(&[u32]) -> lloom_tensor::Result<String> + '_ {
        |ids: &[u32]| -> lloom_tensor::Result<String> {
            self.text
                .tokenizer
                .decode(ids, false)
                .map_err(|e| lloom_tensor::Error::Msg(format!("detokenize: {e}")))
        }
    }

    fn check_fits(&self, prompt: usize, gen: usize) -> Result<()> {
        super::common::check_context_capacity(self.kv_capacity, prompt, gen, 0)
    }

    /// The MTP drafter for one run: the head, the shared embedding and
    /// `lm_head`, and `lanes` (a previous run's, else a fresh one-layer
    /// state). Cheap to build per request.
    fn drafter(&self, lanes: Option<NextnLanes<X::State>>) -> Result<NextnDrafter<'_, X>> {
        let head = self
            .mtp
            .as_ref()
            .context("no MTP head loaded (speculative.method = mtp)")?;
        Ok(NextnDrafter::new(
            head,
            self.model.embed_table(),
            self.model.lm_head(),
            &self.device,
            self.model.capacity(),
        )?
        .with_k(self.mtp_k)
        .with_lanes(lanes))
    }
}

/// The family config a decoder of either alias carries.
fn model_config<M: CausalLM<Config = Config>>(_m: &M) -> &'static Config {
    // `new_cache` ignores its config argument on a `Decoder` (the capacity
    // is the model's); a static placeholder keeps the call shape.
    static CFG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    CFG.get_or_init(Config::tiny)
}

impl<M, X> Served for Engine<M, X>
where
    M: Qwen3_5Model,
    M::Cache: CacheReset + CacheRewind + Send + Sync + 'static,
    X: Layer + 'static,
    X::State: Send + Sync,
{
    fn device(&self) -> &Device {
        &self.device
    }
    fn installed(&self) -> Option<&super::report::Installed> {
        Some(&self.installed)
    }
    fn model_name(&self) -> &str {
        &self.model_name
    }
    fn kv_capacity(&self) -> usize {
        self.kv_capacity
    }
    fn readback(&self) -> ReadbackArm {
        self.readback
    }
    fn prefill_tile(&self) -> usize {
        self.prefill_tile
    }
    fn batch(&self) -> usize {
        1
    }
    fn eos_ids(&self) -> &[u32] {
        &self.text.eos_ids
    }
    fn bos_id(&self) -> Option<u32> {
        None
    }
    fn health(&self) -> Value {
        json!({
            "family": "qwen3_5",
            "arch": self.arch,
            "batch": 1,
            "prefill_tile": self.prefill_tile,
            "kv_capacity": self.kv_capacity,
            "tier": self.tier,
        })
    }
    fn describe(&self) -> String {
        format!(
            "family=qwen3_5 arch={} batch=1 experts={} mtp={} mtp_k={} prefill_tile={} mem_sample={} {}",
            self.arch,
            self.tier["experts"].as_str().unwrap_or("-"),
            self.mtp.is_some(),
            self.mtp_k,
            self.prefill_tile,
            self.mem_sample,
            self.executor.line(),
        )
    }
    fn capabilities(&self) -> super::family::Capabilities {
        super::family::QWEN3_5_CAPABILITIES
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        let mut w = self.load_warnings.clone();
        w.extend(lloom::load_warning::emulated(&self.device));
        w
    }
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>) {
        self.load_warnings.splice(0..0, warnings);
    }
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<u32>> {
        // No BOS on these checkpoints: the template starts with the turn tag.
        Ok(self
            .text
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    }
    fn token_str(&self, id: u32) -> String {
        self.text
            .tokenizer
            .decode(&[id], false)
            .unwrap_or_else(|_| format!("<{id}>"))
    }
    fn token_str_checked(&self, id: u32) -> Result<String> {
        if id as usize >= self.model.vocab_size() {
            anyhow::bail!("token display id outside loaded vocabulary")
        }
        self.text
            .tokenizer
            .decode(&[id], false)
            .map_err(|error| anyhow::anyhow!("decoding token: {error}"))
    }
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        self.text
            .template
            .as_ref()
            .context("no chat template (none beside the tokenizer, none in the GGUF); /v1/chat/completions needs one")?
            .render_checked(messages, "", kwargs, checked)
    }
    fn template_reads(&self, name: &str) -> Option<bool> {
        self.text.template.as_ref()?.reads(name)
    }
    fn think_tags(&self) -> Option<super::template::ThinkTags> {
        self.text.think_tags().cloned()
    }
    fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<Vec<u32>> {
        self.text
            .generation_prompt(kwarg, want)
            .map(<[u32]>::to_vec)
    }
    fn template_renders(&self) -> u64 {
        self.text.template.as_ref().map_or(0, |t| t.renders())
    }
    fn text(&self) -> Option<&super::common::Text> {
        Some(&self.text)
    }
    fn decode_body(&self) -> lloom::nn::Body {
        if self.mtp.is_some() {
            lloom::nn::Body::Window
        } else {
            lloom::nn::Body::Step {
                ring: matches!(self.readback, ReadbackArm::Ring(_)),
            }
        }
    }
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32> {
        let seed = &self.text.seed_tokens;
        (0..n).map(|i| seed[(i + row * 7) % seed.len()]).collect()
    }
    fn complete_until(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        self.check_fits(prompt_ids.len(), params.max_tokens)?;
        let reuse = self.reuse_prefix;
        let detok = self.detok();
        self.with_single(|s, extra| {
            if !reuse {
                s.reset();
            }
            if self.mtp.is_some() {
                // Fallible construction precedes consuming the retained lanes.
                let drafter = self.drafter(None)?.with_lanes(extra.lanes.take());
                let mut lease = DrafterLease {
                    slot: &mut extra.lanes,
                    drafter: Some(drafter),
                };
                let out = s.decode_speculative_until(
                    rewindable::<M>()?,
                    lease
                        .drafter
                        .as_mut()
                        .expect("drafter lease owns its lanes"),
                    prompt_ids,
                    detok,
                    params,
                    |f| on_fragment(f),
                );
                out.map_err(anyhow::Error::from)
            } else {
                s.decode_until(prompt_ids, detok, params, |f| on_fragment(f))
                    .map_err(anyhow::Error::from)
            }
        })
    }
    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        self.check_batch(batch)?;
        self.check_fits(n, 1)?;
        let ids = self.seed_prompt_row(n, 0);
        let device = self.device.clone();
        let tile = self.prefill_tile;
        let gauge = MemGauge::start(&device, self.mem_sample);
        let (prefill, forwards) = self.with_single(|s, _extra| {
            s.reset();
            let params = DecodeParams {
                max_tokens: 1,
                eos_ids: NEVER_EOS.to_vec(),
                prefill_tile: tile,
                ..DecodeParams::default()
            };
            let t = Instant::now();
            let out = s.prefill(&ids, &params)?;
            device.synchronize()?;
            let elapsed = t.elapsed();
            s.reset();
            Ok((elapsed, out.units))
        })?;
        Ok(PrefillTiming {
            prompt_tokens: n,
            batch: 1,
            prefill_ms: ms(prefill),
            forwards,
            mem: gauge.finish(&device),
        })
    }
    fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming> {
        self.check_batch(batch)?;
        super::common::check_context_capacity(self.kv_capacity, prompt_tokens, decode_tokens, 1)?;
        let prompt = self.seed_prompt_row(prompt_tokens, 0);
        let device = self.device.clone();
        let readback = self.readback;
        let prefill_tile = self.prefill_tile;
        let detok = self.detok();
        let gauge = MemGauge::start(&device, self.mem_sample);
        let mut timing = self.with_single(|s, _extra| {
            s.reset();
            let mut detok = detok;
            if self.mtp.is_some() {
                // One run of `D` tokens through the speculative body, the
                // prompt's submit time subtracted (the prefill's GPU tail
                // is inside the bracket), as the other engines bracket
                // their drafters.
                let params = DecodeParams {
                    max_tokens: decode_tokens,
                    eos_ids: NEVER_EOS.to_vec(),
                    readback,
                    prefill_tile,
                    ..DecodeParams::default()
                };
                super::common::profile_reset();
                let t = Instant::now();
                let mut drafter = self.drafter(None)?;
                let run = s.decode_speculative_until(
                    rewindable::<M>()?,
                    &mut drafter,
                    &prompt,
                    &mut detok,
                    &params,
                    |_| true,
                )?;
                device.synchronize()?;
                let wall = ms(t.elapsed());
                let prof = super::common::profile_take();
                let prefill_ms = run.stats.prefill_ns as f64 / 1e6;
                if run.tokens.len() != decode_tokens {
                    anyhow::bail!(
                        "the decode run produced {} tokens, expected {decode_tokens}",
                        run.tokens.len()
                    );
                }
                s.reset();
                let rounds = run.stats.rounds.len();
                let accepted: f64 = run
                    .stats
                    .rounds
                    .iter()
                    .map(|r| r.accepted as f64)
                    .sum::<f64>()
                    / rounds.max(1) as f64;
                let phase_ms = |f: fn(&lloom::nn::speculator::RoundTiming) -> u64| -> f64 {
                    run.stats
                        .rounds
                        .iter()
                        .map(|r| f(&r.timing) as f64 / 1e6)
                        .sum()
                };
                return Ok(DecodeTiming {
                    prompt_tokens,
                    decode_tokens,
                    batch: 1,
                    prefill_ms,
                    decode_ms: wall - prefill_ms,
                    readback: run.readback,
                    bracket: "run wall minus the prompt's submit; the prefill's GPU tail is inside",
                    row_tokens: vec![run.tokens.len()],
                    row_rounds: vec![rounds],
                    row_accepted: vec![accepted],
                    row_draft_ms: vec![phase_ms(|t| t.draft_ns)],
                    row_verify_ms: vec![phase_ms(|t| t.verify_ns)],
                    row_observe_ms: vec![phase_ms(|t| t.observe_ns)],
                    profile: prof,
                    mem: MemReport::default(),
                });
            }
            let params = DecodeParams {
                max_tokens: 1,
                eos_ids: NEVER_EOS.to_vec(),
                readback,
                prefill_tile,
                ..DecodeParams::default()
            };
            let t = Instant::now();
            let first = s.decode(&prompt, &mut detok, &params, |_| {})?;
            device.synchronize()?;
            let prefill_ms = ms(t.elapsed());
            if first.tokens.len() != 1 {
                anyhow::bail!(
                    "the context run produced {} tokens, expected 1",
                    first.tokens.len()
                );
            }
            let mut all: Vec<u32> = prompt.clone();
            all.extend_from_slice(&first.tokens);
            let params = DecodeParams {
                max_tokens: decode_tokens,
                eos_ids: NEVER_EOS.to_vec(),
                readback,
                prefill_tile,
                ..DecodeParams::default()
            };
            super::common::profile_reset();
            let t = Instant::now();
            let run = s.decode(&all, &mut detok, &params, |_| {})?;
            device.synchronize()?;
            let decode_ms = ms(t.elapsed());
            let prof = super::common::profile_take();
            if run.tokens.len() != decode_tokens {
                anyhow::bail!(
                    "the decode run produced {} tokens, expected {decode_tokens}",
                    run.tokens.len()
                );
            }
            if run.prefilled != 0 {
                anyhow::bail!(
                    "the decode run prefilled {} tokens; the context was expected to resume ({:?})",
                    run.prefilled,
                    run.reuse
                );
            }
            s.reset();
            Ok(DecodeTiming {
                prompt_tokens,
                decode_tokens,
                batch: 1,
                prefill_ms,
                decode_ms,
                readback: run.readback,
                bracket: "D steps after the context, resumed from held logits",
                row_tokens: vec![run.tokens.len()],
                row_rounds: Vec::new(),
                row_accepted: Vec::new(),
                row_draft_ms: Vec::new(),
                row_verify_ms: Vec::new(),
                row_observe_ms: Vec::new(),
                profile: prof,
                mem: MemReport::default(),
            })
        })?;
        timing.mem = gauge.finish(&device);
        Ok(timing)
    }
    fn next_token_logprobs(&self, prompt_ids: &[u32], k: usize) -> Result<(u32, Vec<(u32, f32)>)> {
        self.check_fits(prompt_ids.len(), 1)?;
        let device = self.device.clone();
        let ids = prompt_ids.to_vec();
        let logits = self.with_single(|s, _extra| {
            s.reset();
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let logits = s.model().forward(&input, 0, s.cache_mut())?;
            s.reset();
            Ok(logits.squeeze(0)?.to_dtype(DType::F32)?)
        })?;
        let logp = lloom::nn::ops::log_softmax(&logits, D::Minus1)?;
        let v: Vec<f32> = logp.to_vec1()?;
        let mut idx: Vec<usize> = (0..v.len()).collect();
        let k = k.min(v.len());
        idx.select_nth_unstable_by(k.saturating_sub(1).min(v.len() - 1), |&a, &b| {
            v[b].total_cmp(&v[a])
        });
        let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i as u32, v[i])).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let argmax = top.first().map(|t| t.0).context("empty vocabulary")?;
        Ok((argmax, top))
    }
    fn check_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 {
            anyhow::bail!("batch must be positive");
        }
        if batch > 1 {
            anyhow::bail!("batch {batch}: qwen3_5 serves one sequence (no --batch N)");
        }
        Ok(())
    }
    fn shutdown(&self) {}
}

/// Supplied-domain owning Served route over the full Qwen/MTP owner.
/// No default registration until complete built-in prediction/enrollment.
pub mod admitted;
