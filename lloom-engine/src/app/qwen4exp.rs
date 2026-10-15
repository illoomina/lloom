//! The Qwen4-exp engine: one sequence over the resident or the streamed
//! expert tier (`lloom::models::qwen4exp`, `expert-streaming.md` §20),
//! driven by the same `DecodeSession` the LFM2 engine drives. The N-gram PLE
//! hashes each input id into a host-mapped table, so a decode forward needs
//! its id on the host: the blocking arm hands it over
//! (`CausalLM::note_input_ids`), and a ring arm that stamps each token feeds
//! it through the driver's host fill, the GPU waiting for the rows
//! (`CausalLM::host_feed`). `auto` is the blocking arm: the fill is faster
//! while the streamed expert pool holds the working set and slower while it
//! misses (`resolve_readback`, `design/docs/expert-streaming.md` §20.13c).
//! `--mtp` puts the MTP head on the speculative body
//! (`decode_speculative_until`): one drafted token per round, verified by the
//! target, byte-identical to greedy (issue #749). `speculative.method =
//! dflash` puts the DFlash block drafter there instead (the DSpark family's
//! backbone over the target's tapped HC residuals, `models::qwen4exp::dflash`):
//! a block a round, the same verify window. `--batch N`
//! runs the rows through the scheduler (the weights once over the rows, the
//! state per row).

use super::common::{
    lock, model_name_from_dir, ms, newest_snapshot, open_device, DecodeTiming, MemGauge, MemReport,
    PrefillTiming, Text, NEVER_EOS,
};
use super::served::Served;
use super::template::{Rendered, Template};
use crate::{
    spawn_batch, BatchConfig, BatchEvent, BatchHandle, BatchMode, BatchRequest, DecodeParams,
    DecodeSession, Decoded, Idle, ReadbackArm, RingPool, SlotPool, Stop, TextStream,
};
use anyhow::{Context, Result};
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::load_warning::{LoadWarning, LoadWarningKind};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::models::qwen4exp::loader::{
    LoadOptions as ModelLoad, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::{
    ExpertResidency, MtpDrafter, MtpLanes, Qwen4Exp, QwenHead, Resident, Shape as QwenShape, State,
    Streamed,
};
use lloom::nn::expert_stream::HasStreamStats;
use lloom::nn::model::CausalLM;
use lloom::nn::Sequence;
use lloom::nn::{DecodeOptions, Session};
use lloom::profile::{Own, Profile, ProfileKey};
use lloom_tensor::{DType, Device, Tensor, D};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

pub struct LoadOptions {
    pub first_shard: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub cpu: bool,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    /// The decode loop's options (`DecodeConfig`).
    pub decode: crate::DecodeConfig,
    pub reuse_prefix: bool,
    pub warmup: bool,
    pub seed_text: String,
    pub model_name: Option<String>,
    pub prefill_tile: usize,
    pub memory_fraction: f64,
    pub mem_sample: bool,
    pub experts: String,
    pub slots: String,
    pub loader: String,
    pub ple_cache_rows: usize,
    pub ple: bool,
    pub moe_act: String,
    pub mtp: Option<PathBuf>,
    /// The DFlash drafter's checkpoint directory (`speculative.method =
    /// dflash`); empty is the newest [`DFLASH_SNAPSHOT`] in the HF cache.
    pub dflash: Option<PathBuf>,
    /// `speculative.k`: the drafter's tokens a round (DFlash: clamped to its
    /// block; none is the block).
    pub spec_k: Option<usize>,
    /// The expert hotlist file (streamed experts): warm at load, write at
    /// shutdown.
    pub hotlist: Option<PathBuf>,
    /// Rows the batch scheduler serves at once (`--batch N`); 1 is the
    /// single-sequence engine.
    pub batch: usize,
    /// Tokens the MTP drafter proposes per round.
    pub mtp_k: usize,
    /// The streamed tier's prefetcher (`--prefetch`).
    pub prefetch: bool,
    /// The `io` loader's in-flight residency hints (`--io-hints`).
    pub io_hints: bool,
    /// `chunk | replay` (`--prefill-mode`).
    pub prefill_mode: String,
    /// Experts pinned per layer from the hotlist (`--pin`).
    pub pin: usize,
    /// KV tokens the slot budget reserves for (`--kv-reserve-tokens`; default `kv_capacity`).
    pub kv_reserve_tokens: Option<usize>,
    /// The executor arms, installed on the device before the model.
    pub executor: crate::ExecutorConfig,
    /// The memory budgets.
    pub memory: crate::MemoryConfig,
    /// The full-attention KV store (`model.axes.kv-quant`, resolved by the
    /// family); `None` is the checkpoint's f32.
    pub kv_precision: Option<lloom::nn::ops::KvPrecision>,
}

/// The routed experts' activation dtype for `--moe-act`: `auto` is `f16`
/// under the `fast` contract (one of its arms, `Contract::FAST_ARMS`) and
/// `f32` under `deterministic`, the byte-identical routed arm; `f32` and
/// `f16` pin it either way.
pub fn routed_act(
    spelling: &str,
    contract: lloom_tensor::tuning::Contract,
) -> anyhow::Result<DType> {
    Ok(match spelling {
        "auto" if contract.is_fast() => DType::F16,
        "auto" | "f32" => DType::F32,
        "f16" => DType::F16,
        other => anyhow::bail!("--moe-act {other}: auto | f32 | f16"),
    })
}

/// `auto` is a preference, not permission to dispatch a missing kernel.
/// Check every routed stack, including MTP and shards beyond the first.
#[cfg(feature = "metal")]
fn routed_half_supported<'a>(
    tensors: impl IntoIterator<Item = (&'a str, lloom_tensor::quantized::GgmlDType)>,
) -> bool {
    let mut found = false;
    for (name, dtype) in tensors {
        if name.starts_with("blk.")
            && [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            found = true;
            if !lloom_tensor::kernels::GgmlDType::from(dtype).has_gemv_triple() {
                return false;
            }
        }
    }
    found
}

/// The HF cache directory of the DFlash drafter trained for this target.
pub const DFLASH_SNAPSHOT: &str = "models--PixelML--Qwen3.8-Flash-Next-NVFP4-DFlash";

/// DFlash's tokens a round when `speculative.k` names none: three of the
/// block's seven. Measured through `lloom chat` (`speculation.md`
/// §10.2z-vi): the verify window's rows cost more than the deeper drafts
/// return on this target, and K = 3 is at or above K = 7 on every prompt
/// and turn taken.
const DFLASH_DEFAULT_K: usize = 3;

/// The DFlash drafter's directory: `dir`, or the newest [`DFLASH_SNAPSHOT`]
/// when `dir` is empty.
fn dflash_dir(dir: &Path) -> Result<PathBuf> {
    if !dir.as_os_str().is_empty() {
        return Ok(dir.to_path_buf());
    }
    newest_snapshot(DFLASH_SNAPSHOT).context(
        "no PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash snapshot in the HF cache and no drafter \
         directory; the drafter is trained against THIS target (speculation.md §10.2z-vi)",
    )
}

/// The largest absolute position a DFlash draft attends: the KV ceiling
/// plus the block and slack.
fn dflash_max_pos(kv_capacity: usize, dcfg: &DSparkConfig) -> usize {
    kv_capacity + dcfg.block_size + 64
}

/// What the DFlash drafter holds on the device, for the slot plan's claim:
/// its weights at f16 (the checkpoint is bf16, loaded as f16: Apple7
/// emulates bfloat), `fc` kept transposed at the taps' f32
/// (`DSparkProjection::fc_t`), the RoPE table at f16, and its context KV —
/// one page pool per layer, a bucket of tokens past the block's reserve.
fn dflash_device_bytes(dcfg: &DSparkConfig, max_pos: usize) -> usize {
    let h = dcfg.hidden_size;
    let weights = lloom::models::lfm2_dspark::drafter_params(dcfg) * 2;
    let fc_f32 = h * dcfg.target_layer_ids.len() * h * 4;
    let rope = 2 * max_pos * (dcfg.head_dim / 2) * 2;
    let ctx = dcfg.num_hidden_layers
        * 2
        * (4096usize + dcfg.block_size).next_multiple_of(256)
        * dcfg.num_key_value_heads
        * dcfg.head_dim
        * 2;
    weights + fc_f32 + rope + ctx
}

/// The model, shared with the batch thread.
pub struct QwenCore<P: Profile, R: ExpertResidency + HasStreamStats> {
    pub model: Qwen4Exp<P, R>,
}

/// The one sequence every request drives, and the rings kept across
/// requests.
struct Single<P: Profile> {
    seq: Option<Sequence<State<P>>>,
    rings: Option<RingPool>,
    /// The loop's options, handed to every request's session.
    decode: crate::DecodeConfig,
}

pub struct Engine<P: QwenShape, R: ExpertResidency + HasStreamStats> {
    core: Arc<QwenCore<P, R>>,
    device: Device,
    /// Rows the scheduler serves at once; 1 = the single sequence.
    batch: usize,
    text: Text,
    model_name: String,
    kv_capacity: usize,
    /// The arm requests take, resolved at load (`resolve_readback`).
    readback: ReadbackArm,
    /// The executor arms installed at load.
    executor: crate::ExecutorConfig,
    /// What they armed, for the engagement reports (`report::print_executor`).
    installed: super::report::Installed,
    reuse_prefix: bool,
    prefill_tile: usize,
    mem_sample: bool,
    /// The MTP head is loaded: requests decode through the speculative body.
    mtp: bool,
    /// Tokens the MTP drafter proposes per round.
    mtp_k: usize,
    /// Resolved private-cache ring geometry, including the DFlash window.
    private_k: usize,
    /// The DFlash drafter, when it is the speculative body: one sequence's,
    /// kept across requests so its context follows the target's prefix
    /// reuse (`Speculator::begin_at`).
    dflash: Option<Mutex<DSparkSpeculator<QwenHead>>>,
    /// The hotlist to write at shutdown (the usage since load merged in).
    hotlist: Option<PathBuf>,
    /// The tier's load-time description (`[loaded …]`, `/health`).
    tier: Value,
    /// The load's degradations (`Served::load_warnings`).
    load_warnings: Vec<LoadWarning>,
    single: Mutex<Single<P>>,
    /// The MTP drafter's lanes between requests: the head's own KV follows
    /// the single sequence's prefix reuse (`Speculator::begin_at`). Taken
    /// and put back under `single`'s lock.
    mtp_lanes: Mutex<Option<MtpLanes<Own<P>>>>,
    /// `--batch N`: the scheduler and its thread.
    sched: Option<BatchHandle<Qwen4Exp<P, R>, ()>>,
    sched_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Request-level residency choice; the resident/streamed hot loops remain
/// separately monomorphized and share one supplied-domain loaded owner.
pub enum Engines<P: QwenShape> {
    Resident(Engine<P, Resident>),
    Streamed(Engine<P, Streamed>),
}

impl<P: QwenShape> Engines<P> {
    fn into_served(self) -> Box<dyn Served> {
        match self {
            Self::Resident(engine) => Box::new(engine),
            Self::Streamed(engine) => Box::new(engine),
        }
    }

    /// Existing chat body inside the same owner used by decision requests.
    /// Whole-pass predictions include outstanding stream/IO work and staging;
    /// worker namespace/pass propagation is a prerequisite, not TLS inheritance.
    pub fn complete_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        prompt_ids: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        Ok(loaded.with_pass(predicted_extra, |engine, _context| {
            let out = match engine {
                Self::Resident(engine) => {
                    engine.complete_until(prompt_ids, params, &mut on_fragment)
                }
                Self::Streamed(engine) => {
                    engine.complete_until(prompt_ids, params, &mut on_fragment)
                }
            };
            out.map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
        })?)
    }

    /// Prepared retained-trunk decisions on the chat model's weights. Native
    /// streaming custody, allocator/host producers and tokenizer boundaries must
    /// be enrolled/validated before default endpoint activation.
    pub fn decide_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        request: &crate::decision::CausalRequest,
    ) -> Result<crate::decision::SystemOneResponse> {
        Ok(
            loaded.with_pass(predicted_extra, |engine, _context| match engine {
                Self::Resident(engine) => engine.decide_prepared(request),
                Self::Streamed(engine) => engine.decide_prepared(request),
            })?,
        )
    }
}

#[derive(Clone, Copy)]
enum LoadAdmission {
    Legacy,
    CallerScoped,
}

/// Additive complete-engine factory with one caller-supplied load domain. The
/// scope starts before drafter/header/slot planning and remains through expert
/// loading, hotlist warming, cache/rings, tokenizer and warmup. No model-only
/// wrapper or second legacy admission. Source activation is still conditional
/// on complete native/worker/host producer enrollment; existing routes remain.
pub fn load_admitted_as<P: QwenShape>(
    opts: LoadOptions,
    device: &Device,
    load: lloom::nn::memory::LoadContext,
) -> Result<lloom::nn::memory::Loaded<Engines<P>>> {
    lloom_tensor::memory_admission::begin_model_load();
    if opts.cpu != device.is_cpu() {
        anyhow::bail!("supplied Qwen4Exp device disagrees with load options");
    }
    if opts.batch != 1 {
        anyhow::bail!("admitted Qwen4Exp batch scheduler pass propagation is pending");
    }
    Ok(lloom::nn::memory::Loaded::new(
        device,
        load,
        |resolved, _context| {
            load_on::<P>(opts, Some(resolved.clone()), LoadAdmission::CallerScoped)
                .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
        },
    )?)
}

/// Why a ring arm on this engine cannot take the PLE's host fill
/// (`CausalLM::host_feed`), or `None` when it can. Without the fill a ring
/// arm's forward reads its fed id back.
fn host_fill_absent<P: Profile, R: ExpertResidency>(
    model: &Qwen4Exp<P, R>,
    device: &Device,
) -> Option<&'static str> {
    if !device.is_metal() {
        Some("the host fill needs Metal (a shared event the GPU waits on)")
    } else if !model.has_ple() {
        Some("no PLE is attached")
    } else if !model.host_fill() {
        Some("the host fill is off")
    } else {
        None
    }
}

/// The arm requests take, and the load's note on it.
///
/// `auto` is the blocking arm on Qwen4-exp. A ring arm that stamps each
/// token feeds the PLE through the driver's host fill instead, which is
/// faster than blocking while the streamed expert pool holds the working set
/// and slower while it misses (`expert-streaming.md` §20.13c); on any other
/// ring arm, or where the fill cannot run, the forward reads its fed id back.
/// An explicit arm is served as asked, and the note says which of the two it
/// is.
fn resolve_readback(
    asked: ReadbackArm,
    no_fill: Option<&'static str>,
) -> (ReadbackArm, Option<String>) {
    match asked {
        ReadbackArm::Auto => (
            ReadbackArm::Blocking,
            Some(
                "auto: blocking on Qwen4-exp. --readback ring-fixed feeds the PLE from the host \
                 instead: faster while the streamed expert pool holds the working set, slower \
                 while it misses (expert-streaming.md §20.13c)"
                    .to_string(),
            ),
        ),
        ReadbackArm::Ring(ring) => {
            let note = match reads_fed_id_back(ring, no_fill) {
                Some(why) => format!(
                    "{}: each decode forward reads its fed id back ({why}) — correct, and slower \
                     than blocking where measured (expert-streaming.md §20.13b)",
                    asked.name()
                ),
                None => format!(
                    "{}: the PLE fed by the driver's host fill — faster than blocking while the \
                     streamed expert pool holds the working set, slower while it misses \
                     (expert-streaming.md §20.13c)",
                    asked.name()
                ),
            };
            (asked, Some(note))
        }
        other => (other, None),
    }
}

/// Why a ring arm's decode forwards read their fed id back rather than
/// taking the PLE from the driver's host fill; `None` when the fill runs.
fn reads_fed_id_back(
    ring: crate::decode::RingArm,
    no_fill: Option<&'static str>,
) -> Option<&'static str> {
    no_fill.or((!crate::options::stamps_each_token(ring)).then_some(
        "a deferred EOS read on the fixed cadence stamps once per window, and the fill waits on \
         each token's stamp",
    ))
}

/// `first` plus its `-0000k-of-0000N` siblings, in shard order.
fn discover_shards(first: &Path) -> Vec<PathBuf> {
    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    let Some(pos) = name.find("-of-") else {
        return vec![first.to_path_buf()];
    };
    let tail = &name[pos + 4..];
    let count: usize = tail.trim_end_matches(".gguf").parse().unwrap_or(1);
    let head = &name[..pos.saturating_sub(5)];
    let mut out = Vec::with_capacity(count);
    for k in 1..=count {
        let sibling = first.with_file_name(format!("{head}{k:05}-of-{count:05}.gguf"));
        if sibling.exists() {
            out.push(sibling);
        }
    }
    if out.is_empty() {
        vec![first.to_path_buf()]
    } else {
        out
    }
}

/// The DFlash speculator over `model` from the checkpoint in `dir`, and the
/// layers the target's cache taps for it: the drafter's own weights (f16 on
/// Metal, f32 on the CPU), the target's `token_embd` and `lm_head` shared,
/// `speculative.k` (else [`DFLASH_DEFAULT_K`]) clamping the block, its
/// pipelines warmed off the clock.
fn load_dflash<P: Profile, R: ExpertResidency>(
    model: &Qwen4Exp<P, R>,
    dir: &Path,
    device: &Device,
    kv_capacity: usize,
    spec_k: Option<usize>,
    warmup: bool,
) -> Result<(DSparkSpeculator<QwenHead>, Vec<usize>)> {
    let dcfg = DSparkConfig::read_from_dir(dir)
        .with_context(|| format!("reading the DFlash drafter config in {}", dir.display()))?;
    let cfg = model.config();
    if dcfg.hidden_size != cfg.hidden_size || dcfg.num_target_layers != cfg.n_layers {
        anyhow::bail!(
            "the drafter in {} reads {} layers of width {}, the target has {} of {}: not this \
             target's drafter",
            dir.display(),
            dcfg.num_target_layers,
            dcfg.hidden_size,
            cfg.n_layers,
            cfg.hidden_size
        );
    }
    let file = dir.join("model.safetensors");
    if !file.exists() {
        anyhow::bail!("{} does not exist", file.display());
    }
    let dtype = if device.is_metal() {
        DType::F16
    } else {
        DType::F32
    };
    let t = Instant::now();
    let vb = unsafe { lloom::nn::VarBuilder::from_mmaped_safetensors(&[file], dtype, device)? };
    let taps = dcfg.tap_layers();
    let block = dcfg.max_draft();
    let drafter = DSparkDrafter::new(dcfg.clone(), dflash_max_pos(kv_capacity, &dcfg), vb)
        .context("constructing the DFlash drafter")?;
    let mut spec = DSparkSpeculator::with_head(drafter, QwenHead::of(model), 1)
        .context("constructing the DFlash speculator")?;
    spec.set_max_draft(spec_k.unwrap_or(DFLASH_DEFAULT_K));
    let k = lloom::nn::Speculator::max_draft(&spec);
    eprintln!(
        "[dflash: drafter loaded from {} in {:.1} ms, {} layers at {dtype:?}, K={k} of the block \
         {block} (verify window {} positions), taps {taps:?}, context kv {}]",
        dir.display(),
        ms(t.elapsed()),
        dcfg.num_hidden_layers,
        k + 1,
        spec.ctx_kv().name()
    );
    if warmup {
        spec.warmup(device)
            .context("warming the DFlash drafter's pipelines")?;
    }
    Ok((spec, taps))
}

/// The boundary (`profiles-and-memory-design.md` §7.4): the engine at the
/// profile the mode names — `Ring` under `--mtp` or `--speculate dflash`,
/// `Paged` at `--batch N`,
/// the regime from the contract — one `match` over the eight
/// (`lloom::profile::dispatch!`), bounded by what the family ships.
pub fn load(opts: LoadOptions) -> Result<Box<dyn Served>> {
    lloom_tensor::memory_admission::begin_model_load();
    use lloom::profile::{NoRewind, One, Paged, Rewind as _, Ring, Rows as _};
    let key = ProfileKey::at_contract(
        if opts.mtp.is_some() || opts.dflash.is_some() {
            Ring::NAME
        } else {
            NoRewind::NAME
        },
        if opts.batch > 1 {
            Paged::NAME
        } else {
            One::NAME
        },
    );
    lloom::profile::dispatch!(key, lloom::profile::QWEN4EXP, P => load_as::<P>(opts))
}

/// The evidence the speculative drivers take: the `Ring` shapes give it,
/// the `No` shapes refuse by name (`profiles-and-memory-design.md` §7.3).
fn rewindable<P: Profile>() -> Result<lloom::profile::RewindProof<P>> {
    lloom::profile::RewindProof::<P>::ask().with_context(|| {
        format!(
            "a speculative drafter on the {} shape: `--mtp` and `--speculate dflash` run on the `Ring` shape",
            P::KEY
        )
    })
}

/// Load the family `opts` names, on the tier `opts.experts` names, at `P`.
/// The turn checkpoints the device's ledger admits: `want`, or as many
/// frames of `frame_bytes` as the quota leaves uncommitted; `want` where
/// there is no ledger (CPU) or no quota.
fn admitted_checkpoints(device: &Device, frame_bytes: usize, want: usize) -> usize {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        let quota = m.ledger().quota();
        if quota > 0 && frame_bytes > 0 {
            let free = quota.saturating_sub(m.ledger().committed());
            return want.min(free / frame_bytes);
        }
    }
    let _ = (device, frame_bytes);
    want
}

fn load_as<P: QwenShape>(opts: LoadOptions) -> Result<Box<dyn Served>> {
    load_on::<P>(opts, None, LoadAdmission::Legacy).map(Engines::into_served)
}

fn load_on<P: QwenShape>(
    mut opts: LoadOptions,
    resolved: Option<Device>,
    admission: LoadAdmission,
) -> Result<Engines<P>> {
    lloom_tensor::memory_admission::begin_model_load();
    {
        // The DFlash drafter's directory and config first: the slot plan
        // claims what it will hold beside the experts.
        let mut drafter_bytes = 0usize;
        if let Some(d) = &opts.dflash {
            if opts.mtp.is_some() {
                anyhow::bail!(
                    "--mtp and speculative.method = dflash: one speculative body per run"
                );
            }
            let dir = dflash_dir(d)?;
            let dcfg = DSparkConfig::read_from_dir(&dir).with_context(|| {
                format!("reading the DFlash drafter config in {}", dir.display())
            })?;
            drafter_bytes = dflash_device_bytes(&dcfg, dflash_max_pos(opts.kv_capacity, &dcfg));
            opts.dflash = Some(dir);
        }
        let device = match resolved {
            Some(device) => device,
            None if opts.cpu => Device::Cpu,
            None => open_device()?,
        };
        let mut shards = discover_shards(&opts.first_shard);
        if let Some(p) = &opts.mtp {
            if !p.is_file() {
                anyhow::bail!("--mtp {}: no such file", p.display());
            }
            shards.push(p.clone());
        }
        #[cfg(feature = "metal")]
        if opts.moe_act == "auto"
            && device.is_metal()
            && routed_act(&opts.moe_act, lloom_tensor::tuning::contract())? == DType::F16
        {
            // Header-only inspection at load, never on the decode hot path.
            let headers = shards
                .iter()
                .map(|path| {
                    let mut file = std::fs::File::open(path)?;
                    lloom_tensor::quantized::gguf_file::Content::read(&mut file)
                        .map_err(anyhow::Error::from)
                })
                .collect::<Result<Vec<_>>>()?;
            if !routed_half_supported(headers.iter().flat_map(|h| {
                h.tensor_infos
                    .iter()
                    .map(|(name, ti)| (name.as_str(), ti.ggml_dtype))
            })) {
                opts.moe_act = "f32".into();
                eprintln!(
                    "[moe-act auto: f32; checkpoint has no complete native f16 routed kernel set]"
                );
            }
        }
        let moe_act = match routed_act(&opts.moe_act, lloom_tensor::tuning::contract())? {
            DType::F16 => Some(DType::F16),
            _ => None,
        };
        let mload = ModelLoad {
            ple_shard: None,
            ple: opts.ple,
            moe_act,
            ple_cache_rows: (opts.ple_cache_rows > 0).then_some(opts.ple_cache_rows),
            mtp: opts.mtp.is_some(),
            memory_fraction: Some(opts.memory_fraction),
            kv_precision: opts.kv_precision,
            machine: match admission {
                LoadAdmission::Legacy => Some(lloom::nn::memory::MachineLoad {
                    gate: opts.memory.machine_gate(),
                    label: super::common::machine_label("qwen4exp", &opts.first_shard),
                    rows: opts.batch.max(1),
                    kv_tokens: opts.kv_reserve_tokens.unwrap_or(opts.kv_capacity).max(1),
                    prefill_tile: opts.prefill_tile,
                }),
                LoadAdmission::CallerScoped => None,
            },
        };
        // The ledger's quota, set before the load so the load's own
        // allocations are admitted against it (§9.5c).
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &device {
            let ws = m.device().recommended_max_working_set_size() as f64;
            m.ledger()
                .set_quota((ws * opts.memory_fraction.max(0.0)) as usize);
        }
        let prefill_mode = match opts.prefill_mode.as_str() {
            "chunk" => PrefillMode::Chunk,
            "replay" => PrefillMode::Replay,
            other => anyhow::bail!("--prefill-mode {other}: chunk | replay"),
        };
        let installed =
            super::common::install_executor(&device, &opts.executor, opts.mtp.is_some())?;
        #[cfg(feature = "metal")]
        if let Some(mb) = opts.memory.free_budget_mb {
            if let Device::Metal(m) = &device {
                m.set_free_budget(mb * 1024 * 1024);
            }
        }
        let t = Instant::now();
        match opts.experts.as_str() {
            "resident" => {
                let mut model: Qwen4Exp<P, Resident> =
                    Qwen4Exp::from_gguf_shards(&shards, &device, &mload)
                        .context("loading Qwen4-exp (resident experts)")?;
                model.set_prefill_mode(prefill_mode);
                model.set_memory_fraction(opts.memory_fraction);
                let tier = json!({ "experts": "resident", "shards": shards.len(), "load_ms": ms(t.elapsed()), "prefill_mode": opts.prefill_mode });
                Engine::<P, Resident>::finish(model, device, opts, tier, installed)
                    .map(Engines::Resident)
            }
            "stream" => {
                let slots = match opts.slots.as_str() {
                    "auto" => SlotBudget::Auto {
                        fraction: opts.memory_fraction,
                        rows: opts.batch.max(1),
                        kv_tokens: opts.kv_reserve_tokens.unwrap_or(opts.kv_capacity).max(1),
                        prefill_tile: opts.prefill_tile,
                    },
                    n => SlotBudget::Fixed(n.parse().context("--slots: a count or `auto`")?),
                };
                let loader = match opts.loader.as_str() {
                    "pread" => StreamLoader::Pread,
                    "io" => StreamLoader::Io,
                    other => anyhow::bail!("--loader {other}: pread | io"),
                };
                let stream = StreamOptions {
                    slots,
                    loader,
                    prefetch: opts.prefetch,
                    io_hints: opts.io_hints,
                    drafter_bytes,
                };
                let mut model: Qwen4Exp<P, Streamed> =
                    Qwen4Exp::from_gguf_streamed(&shards, &device, &mload, &stream)
                        .context("loading Qwen4-exp (streamed experts)")?;
                model.set_prefill_mode(prefill_mode);
                model.set_memory_fraction(opts.memory_fraction);
                let mut warmed = json!(null);
                if let Some(path) = &opts.hotlist {
                    if path.exists() {
                        let tw = std::time::Instant::now();
                        let hot = lloom::models::qwen4exp::hotlist::Hotlist::read(path)
                            .with_context(|| format!("reading the hotlist {}", path.display()))?;
                        let loaded = model.warm_from_hotlist(&hot)?;
                        let pinned = if opts.pin > 0 {
                            model.pin_from_hotlist(&hot, opts.pin)?
                        } else {
                            0
                        };
                        warmed = json!({
                            "path": path.display().to_string(),
                            "loaded": loaded,
                            "pinned": pinned,
                            "bytes": model.stream_stats().bytes_loaded,
                            "ms": ms(tw.elapsed()),
                        });
                    }
                }
                let tier = json!({
                    "experts": "stream",
                    "slots_per_layer": model.slots_per_layer(),
                    "loader": opts.loader,
                    "prefetch": opts.prefetch,
                    "io_hints": opts.io_hints,
                    "prefill_mode": opts.prefill_mode,
                    "pass_plans": opts.executor.pass_plans_armed(),
                    "shards": shards.len(),
                    "load_ms": ms(t.elapsed()),
                    "warm": warmed,
                });
                Engine::<P, Streamed>::finish(model, device, opts, tier, installed)
                    .map(Engines::Streamed)
            }
            other => anyhow::bail!("--experts {other}: resident | stream"),
        }
    }
}

impl<P: QwenShape, R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Engine<P, R> {
    fn finish(
        mut model: Qwen4Exp<P, R>,
        device: Device,
        opts: LoadOptions,
        mut tier: Value,
        installed: super::report::Installed,
    ) -> Result<Self> {
        let tokenizer_path = match &opts.tokenizer {
            Some(p) => p.clone(),
            None => newest_snapshot("models--Qwen--Qwen3.8-Flash-Next")
                .context("no Qwen/Qwen3.8-Flash-Next snapshot in the HF cache; pass --tokenizer")?
                .join("tokenizer.json"),
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", tokenizer_path.display()))?;
        let tokenizer_dir = tokenizer_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        // The turn end and the checkpoint's own EOS, as `qwen4exp-chat` stops.
        let mut eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        eos_ids.dedup();
        if eos_ids.is_empty() {
            anyhow::bail!("the tokenizer has neither <|im_end|> nor <|endoftext|>");
        }
        let template = match std::fs::read_to_string(tokenizer_dir.join("chat_template.jinja")) {
            Ok(src) => Some(Template::compile(&src)?),
            Err(_) => None,
        };
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
            .unwrap_or_else(|| model_name_from_dir(&tokenizer_dir));
        let no_fill = host_fill_absent(&model, &device);
        let (readback, note) = resolve_readback(opts.readback, no_fill);
        let mut load_warnings = Vec::new();
        if let Some(note) = note {
            eprintln!("[readback {note}]");
            // Only the arm that reads its fed id back is slower than asked;
            // `auto`'s resolution and the host fill are the arm as asked.
            if let ReadbackArm::Ring(ring) = readback {
                if reads_fed_id_back(ring, no_fill).is_some() {
                    load_warnings.push(LoadWarning::new(LoadWarningKind::Readback, note));
                }
            }
        }
        let mtp = opts.mtp.is_some();
        if mtp && model.mtp_head().is_none() {
            anyhow::bail!("--mtp was given but the loader found no MTP head in the shards");
        }
        let dflash = match &opts.dflash {
            Some(dir) => Some(load_dflash(
                &model,
                dir,
                &device,
                opts.kv_capacity,
                opts.spec_k,
                opts.warmup,
            )?),
            None => None,
        };
        // The sequence's state at the shape `P` names (`Shape::single`): its
        // own on `No × One`; under `--mtp` or DFlash the verify window's
        // states in phase rings (`K` phases for the drafter's `K`), an accept
        // by phase; DFlash's cache names the layers it taps.
        let k = match &dflash {
            Some((spec, _)) => lloom::nn::Speculator::max_draft(spec),
            None => opts.mtp_k.max(1),
        };
        // The turn-checkpoint ring (`sequence-reuse-design.md` §2.3b)
        // beside the experts' budget: as many of the model's frames as the
        // ledger's quota leaves uncommitted after the load (the streamed
        // tier's pool is a `Cache` claim the arbiter already sized; the
        // ring is a capability a sequence holds).
        let frame = model.checkpoint_frame_bytes();
        let slots = admitted_checkpoints(&device, frame, model.turn_checkpoints());
        model.set_turn_checkpoints(slots);
        eprintln!(
            "[checkpoint ring: {slots} × {:.1} MiB ({:.1} MiB; 0 = every re-rendered turn prefills whole)]",
            frame as f64 / (1024.0 * 1024.0),
            (slots * frame) as f64 / (1024.0 * 1024.0)
        );
        tier["turn_checkpoints"] =
            json!({ "slots": slots, "frame_bytes": frame, "bytes": slots * frame });
        let mut cache = P::single(&model, &device, k)?;
        if let Some((_, taps)) = &dflash {
            cache.set_tap_layers(taps)?;
        }
        cache.want_nextn = mtp;
        tier["mtp"] = json!(mtp);
        tier["dflash"] = json!(opts.dflash.as_ref().map(|d| d.display().to_string()));
        tier["ple"] = json!(opts.ple);
        tier["ple_cache_rows"] = json!(opts.ple_cache_rows);
        tier["moe_act"] = json!(
            match routed_act(&opts.moe_act, lloom_tensor::tuning::contract())? {
                DType::F16 => "f16",
                _ => "f32",
            }
        );
        tier["config"] = json!(model.config().axes_line());
        tier["first_shard"] = json!(opts.first_shard.display().to_string());
        let mut engine = Engine {
            core: Arc::new(QwenCore { model }),
            installed,
            device,
            text: Text {
                tokenizer,
                template,
                bos_token: String::new(),
                bos_id: None,
                eos_ids,
                seed_tokens,
                think: Default::default(),
                gen_prompts: Default::default(),
            },
            model_name,
            kv_capacity: opts.kv_capacity,
            readback,
            executor: opts.executor.clone(),
            reuse_prefix: opts.reuse_prefix,
            prefill_tile: opts.prefill_tile,
            mem_sample: opts.mem_sample,
            mtp,
            mtp_k: opts.mtp_k.max(1),
            private_k: k,
            dflash: dflash.map(|(spec, _)| Mutex::new(spec)),
            hotlist: opts.hotlist.clone(),
            tier,
            load_warnings,
            single: Mutex::new(Single {
                seq: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                decode: opts.decode.clone(),
            }),
            mtp_lanes: Mutex::new(None),
            batch: opts.batch.max(1),
            sched: None,
            sched_thread: Mutex::new(None),
        };
        if opts.batch > 1 {
            // `--batch N`: N private caches in a slot pool and the continuous
            // K = 1 scheduler over `forward_rows` (no prefix sharing: the
            // recurrent channels have no page to share).
            // The rows' recurrent state in one slot-indexed pool: the
            // DeltaNet step runs as one launch per kernel over the rows.
            let (_pools, mut caches) = P::rows(
                &engine.core.model,
                &engine.device,
                opts.batch,
                opts.mtp_k.max(1),
            )?;
            let pool = SlotPool::new(opts.batch, || caches.pop().expect("one cache per slot"));
            let cfg = BatchConfig {
                device: engine.device.clone(),
                pool,
                rings: RingPool::new(),
                opts: DecodeOptions {
                    lookahead: 0,
                    prefill_tile: opts.prefill_tile,
                    ..DecodeOptions::new(0)
                },
                max_rows: opts.batch,
                reuse_prefix: opts.reuse_prefix,
                share_prefix: false,
                mode: BatchMode::Continuous,
                extra: (),
                on_counters: None,
            };
            let (handle, thread) =
                spawn_batch(Arc::clone(&engine.core), |c: &QwenCore<P, R>| &c.model, cfg)
                    .map_err(|e| anyhow::anyhow!("starting the batch scheduler: {e}"))?;
            engine.sched = Some(handle);
            *lock(&engine.sched_thread) = Some(thread);
            eprintln!(
                "[batch: scheduler up, {} rows per step (Qwen4-exp: the weights once over the rows, the state per row)]",
                opts.batch
            );
        }
        if opts.warmup {
            let t = Instant::now();
            // The warmup's passes belong to no sequence and must leave no plan
            // behind (`DESIGN.md` §9.2aa).
            #[cfg(feature = "metal")]
            let _paused = match &engine.device {
                Device::Metal(m) => Some(m.pause_pass_plans()?),
                _ => None,
            };
            let (libs, pipes) = engine.with_single(|s| Ok(s.warmup()?))?;
            eprintln!(
                "[warmup: {libs} libraries, {pipes} pipelines compiled in {:.1} ms]",
                ms(t.elapsed())
            );
        }
        engine.text.prime();
        Ok(engine)
    }

    /// Run `f` over a session on the held sequence, and put the sequence
    /// back whatever `f` returned.
    fn with_single<T>(
        &self,
        f: impl FnOnce(&mut DecodeSession<'_, Qwen4Exp<P, R>>) -> Result<T>,
    ) -> Result<T> {
        let mut single = lock(&self.single);
        let seq = single
            .seq
            .take()
            .context("the sequence is checked out — a previous request panicked mid-decode")?;
        let rings = single.rings.take().unwrap_or_default();
        let mut session =
            DecodeSession::from_parts(&self.core.model, seq, self.device.clone(), rings)
                .with_config(single.decode.clone());
        let out = f(&mut session);
        let (seq, rings) = session.into_parts();
        single.seq = Some(seq);
        single.rings = Some(rings);
        out
    }

    fn decide_prepared(
        &self,
        request: &crate::decision::CausalRequest,
    ) -> lloom_tensor::Result<crate::decision::SystemOneResponse> {
        self.check_fits(request.max_prompt_tokens(), 0)
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))?;
        let single = lock(&self.single);
        let options = single.decode.options(0, ReadbackArm::Blocking, 0);
        let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, &self.device)?;
        request.execute_with_cache_factory(
            &self.core.model,
            &mut readback,
            &options,
            |model, device| P::single(model, device, self.private_k),
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
    /// cache). Cheap to build per request.
    fn drafter(&self, lanes: Option<MtpLanes<Own<P>>>) -> Result<MtpDrafter<'_, Own<P>, R>> {
        let head = self
            .core
            .model
            .mtp_head()
            .context("no MTP head loaded (start with --mtp <gguf>)")?;
        Ok(MtpDrafter::new(
            head,
            self.core.model.embed_tokens(),
            self.core.model.lm_head(),
            self.core.model.config(),
            &self.device,
        )
        .with_k(self.mtp_k)
        .with_lanes(lanes))
    }

    /// The streamed tier's counters (zeros on the resident tier).
    fn tier_stats(&self) -> Value {
        let s = self.core.model.stream_stats();
        let st = self.core.model.staging_stats();
        let ple = self.core.model.ple_cache_stats();
        json!({
            "slots_per_layer": self.core.model.slots_per_layer(),
            "plans": s.plans, "groups": s.groups, "hits": s.hits, "misses": s.misses,
            "evictions": s.evictions, "bytes_loaded": s.bytes_loaded,
            "hit_rate": s.hits as f64 / (s.hits + s.misses).max(1) as f64,
            "staging": { "buffers": st.buffers, "bytes": st.bytes, "allocations": st.allocations,
                          "reuses": st.reuses, "overflows": st.overflows },
            "ple_cache": ple.map(|(h, m)| json!({ "hits": h, "misses": m })),
        })
    }
}

/// `n` sequences checked out of the pool for a closed job, reset, and the
/// slots to return them to.
struct TakenSessions<'m, P: QwenShape, R: ExpertResidency + HasStreamStats> {
    sessions: Vec<Session<'m, Qwen4Exp<P, R>>>,
    slots: Vec<usize>,
}

impl<'m, P: QwenShape, R: ExpertResidency + HasStreamStats> TakenSessions<'m, P, R> {
    fn give_back(self, idle: &mut Idle<'_, 'm, Qwen4Exp<P, R>, ()>) {
        for (slot, mut s) in self.slots.into_iter().zip(self.sessions) {
            s.reset();
            idle.pool.release(slot, s.into_sequence());
        }
    }
}

/// Run `f` over `n` sequences checked out of the pool and reset, and put
/// them back — reset again — whatever `f` returned.
fn with_sessions<'m, P: QwenShape, R: ExpertResidency + HasStreamStats, T>(
    idle: &mut Idle<'_, 'm, Qwen4Exp<P, R>, ()>,
    n: usize,
    f: impl FnOnce(
        &mut Idle<'_, 'm, Qwen4Exp<P, R>, ()>,
        &mut Vec<Session<'m, Qwen4Exp<P, R>>>,
    ) -> Result<T>,
) -> Result<T> {
    let mut sessions = Vec::with_capacity(n);
    let mut slots = Vec::with_capacity(n);
    for _ in 0..n {
        let (routed, seq) = idle
            .pool
            .acquire(&[])
            .context("a closed job found the pool empty")?;
        let mut s = Session::from_sequence(idle.model, seq);
        s.reset();
        sessions.push(s);
        slots.push(routed.slot);
    }
    let mut taken = TakenSessions { sessions, slots };
    let out = f(idle, &mut taken.sessions);
    if taken.sessions.len() != taken.slots.len() {
        return Err(anyhow::anyhow!(
            "a closed job returned {} of {} sessions; the pool has lost slots",
            taken.sessions.len(),
            taken.slots.len()
        ));
    }
    taken.give_back(idle);
    out
}

impl<P: QwenShape, R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Engine<P, R> {
    /// Run a closed job on the scheduler's thread, the rows idle.
    fn closed<T: Send + 'static>(
        &self,
        job: impl for<'a, 'm> FnOnce(&mut Idle<'a, 'm, Qwen4Exp<P, R>, ()>) -> T + Send + 'static,
    ) -> Result<T> {
        let sched = self
            .sched
            .as_ref()
            .context("no batch scheduler (start lloom serve with --batch N)")?;
        sched
            .run_closed_blocking(job)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// A request through the batch scheduler: submitted, its tokens read off
    /// the channel and streamed as text, its outcome the row's.
    fn complete_batched(
        &self,
        sched: &BatchHandle<Qwen4Exp<P, R>, ()>,
        prompt_ids: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = sched.submit(BatchRequest {
            prompt_ids: prompt_ids.to_vec(),
            params: params.clone(),
            cancel: Arc::clone(&cancel),
        });
        let mut detok = self.detok();
        let mut stream = TextStream::new(params.max_tokens);
        let mut stopped_on_string = false;
        let mut cancelled = false;
        let done = loop {
            match rx.recv() {
                Ok(BatchEvent::Tokens(ids)) => {
                    if cancel.load(Ordering::Acquire) {
                        continue;
                    }
                    let (chunk, stopped) =
                        stream.publish(&ids, &mut detok, &params.stop_strings)?;
                    if let Some(chunk) = chunk {
                        if !on_fragment(&chunk) {
                            cancelled = true;
                            cancel.store(true, Ordering::Release);
                            continue;
                        }
                    }
                    if stopped {
                        stopped_on_string = true;
                        cancel.store(true, Ordering::Release);
                    }
                }
                Ok(BatchEvent::Done(d)) => break *d,
                Ok(BatchEvent::Error(e)) => anyhow::bail!("batch: {e}"),
                Err(_) => anyhow::bail!("batch: the scheduler dropped the request"),
            }
        };
        let stop = if cancelled || (done.cancelled && !stopped_on_string) {
            Stop::Cancelled
        } else if stopped_on_string {
            Stop::StopString
        } else if done.eos {
            Stop::Eos
        } else {
            Stop::Length
        };
        if stop != Stop::Cancelled && !stopped_on_string {
            if let Some(rest) = stream.held() {
                on_fragment(&rest);
            }
        }
        let (text, _) = stream.finish(&params.stop_strings);
        Ok(Decoded {
            text,
            tokens: done.tokens,
            stop,
            logits_sha256: None,
            reuse: done.reuse,
            reused: done.reused,
            prefilled: done.prefilled,
            turn: done.turn,
            checkpoint: Default::default(),
            checkpoint_at: None,
            readback: ReadbackArm::Batched {
                rows: done.joined_beside,
            },
            stats: done.stats,
            thinking: None,
            logprobs: None,
        })
    }

    /// `batch` prompts of `n` seed tokens through the scheduler's rows, the
    /// bracket closing when the device drains.
    fn prefill_throughput_rows(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        let prompts: Vec<Vec<u32>> = (0..batch).map(|r| self.seed_prompt_row(n, r)).collect();
        let tile = self.prefill_tile;
        let gauge = MemGauge::start(&self.device, self.mem_sample);
        let (elapsed, forwards) =
            self.closed(move |idle| -> Result<(std::time::Duration, usize)> {
                with_sessions(idle, batch, |idle, sessions| {
                    let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                    let opts = DecodeOptions {
                        lookahead: 0,
                        prefill_tile: tile,
                        ..DecodeOptions::new(0)
                    };
                    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                    let t = Instant::now();
                    let outs = Session::decode_batched(
                        sessions,
                        &mut sampler,
                        &refs,
                        &opts,
                        &NEVER_EOS,
                        idle.device,
                        &mut |_, _| Ok(()),
                    )?;
                    idle.device.synchronize()?;
                    let elapsed = t.elapsed();
                    Ok((elapsed, outs.iter().map(|o| o.stats.prefill_units).sum()))
                })
            })??;
        Ok(PrefillTiming {
            prompt_tokens: n,
            batch,
            prefill_ms: ms(elapsed),
            forwards,
            mem: gauge.finish(&self.device),
        })
    }

    /// Every row's context built outside the bracket, then `D` batched
    /// steps timed together: `decode_ms` is the batch's.
    fn decode_throughput_rows(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming> {
        let prompts: Vec<Vec<u32>> = (0..batch)
            .map(|r| self.seed_prompt_row(prompt_tokens, r))
            .collect();
        let tile = self.prefill_tile;
        let gauge = MemGauge::start(&self.device, self.mem_sample);
        let mut timing = self.closed(move |idle| -> Result<DecodeTiming> {
            with_sessions(idle, batch, |idle, sessions| {
                let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                // One run of `D` tokens per row (the blocking arm holds no
                // logits to resume from between runs): the rows' prefills
                // — each waited on by the driver — subtracted from the wall.
                let opts = DecodeOptions {
                    lookahead: 0,
                    prefill_tile: tile,
                    ..DecodeOptions::new(decode_tokens)
                };
                super::common::profile_reset();
                let t = Instant::now();
                let outs = Session::decode_batched(
                    sessions,
                    &mut sampler,
                    &refs,
                    &opts,
                    &NEVER_EOS,
                    idle.device,
                    &mut |_, _| Ok(()),
                )?;
                idle.device.synchronize()?;
                let wall = ms(t.elapsed());
                let prof = super::common::profile_take();
                let prefill_ms: f64 = outs.iter().map(|o| o.stats.prefill_ns as f64 / 1e6).sum();
                for (i, o) in outs.iter().enumerate() {
                    if o.ids.len() != decode_tokens {
                        anyhow::bail!(
                            "row {i}: the decode run produced {} tokens, expected {decode_tokens}",
                            o.ids.len()
                        );
                    }
                }
                let decode_ms = (wall - prefill_ms).max(0.0);
                Ok(DecodeTiming {
                    prompt_tokens,
                    decode_tokens,
                    batch,
                    prefill_ms,
                    decode_ms,
                    readback: ReadbackArm::Batched { rows: batch },
                    bracket: "run wall minus the rows' prefills (each waited on): the batched steps alone",
                    row_tokens: outs.iter().map(|o| o.ids.len()).collect(),
                    row_rounds: Vec::new(),
                    row_accepted: Vec::new(),
                    row_draft_ms: Vec::new(),
                    row_verify_ms: Vec::new(),
                    row_observe_ms: Vec::new(),
                    profile: prof,
                    mem: MemReport::default(),
                })
            })
        })??;
        timing.mem = gauge.finish(&self.device);
        Ok(timing)
    }
}

impl<P: QwenShape, R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Served
    for Engine<P, R>
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
    fn reuse_prefix(&self) -> Option<bool> {
        Some(self.reuse_prefix)
    }
    fn prefill_tile(&self) -> usize {
        self.prefill_tile
    }
    fn batch(&self) -> usize {
        self.batch
    }
    fn eos_ids(&self) -> &[u32] {
        &self.text.eos_ids
    }
    fn bos_id(&self) -> Option<u32> {
        None
    }
    fn health(&self) -> Value {
        json!({
            "family": "qwen4exp",
            "batch": self.batch,
            "prefill_tile": self.prefill_tile,
            "kv_capacity": self.kv_capacity,
            "tier": self.tier,
            "stream": self.tier_stats(),
        })
    }
    fn describe(&self) -> String {
        format!(
            "family=qwen4exp batch={} experts={} slots={} loader={} mtp={} dflash={} ple={} ple_cache_rows={} moe_act={} config=[{}] prefill_tile={} mem_sample={} {}",
            self.batch,
            self.tier["experts"].as_str().unwrap_or("?"),
            self.tier["slots_per_layer"].as_u64().unwrap_or(0),
            self.tier["loader"].as_str().unwrap_or("-"),
            self.mtp,
            self.dflash.is_some(),
            self.tier["ple"].as_bool().unwrap_or(true),
            self.tier["ple_cache_rows"].as_u64().unwrap_or(0),
            self.tier["moe_act"].as_str().unwrap_or("f32"),
            self.core.model.config().axes_line(),
            self.prefill_tile,
            self.mem_sample,
            self.executor.line(),
        )
    }
    fn capabilities(&self) -> super::family::Capabilities {
        super::family::QWEN4EXP_CAPABILITIES
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
        // No BOS on this checkpoint: the template starts with the turn tag.
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
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        self.text
            .template
            .as_ref()
            .context("no chat_template.jinja beside the tokenizer; /v1/chat/completions needs one")?
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
    fn thinking_budget_refused(&self) -> Option<&'static str> {
        self.sched.is_some().then_some(super::budgets::BATCHED)
    }
    fn decode_body(&self) -> lloom::nn::Body {
        if self.sched.is_some() {
            lloom::nn::Body::BatchedStep
        } else if self.mtp || self.dflash.is_some() {
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
        if let Some(sched) = &self.sched {
            return self.complete_batched(sched, prompt_ids, params, |f| on_fragment(f));
        }
        let reuse = self.reuse_prefix;
        let params = &*params.under_reuse(reuse);
        let detok = self.detok();
        self.with_single(|s| {
            if !reuse || params.cold {
                s.reset();
            }
            if let Some(d) = &self.dflash {
                let mut spec = lock(d);
                s.decode_speculative_until(
                    rewindable::<P>()?,
                    &mut *spec,
                    prompt_ids,
                    detok,
                    params,
                    |f| on_fragment(f),
                )
                .map_err(anyhow::Error::from)
            } else if self.mtp {
                let lanes = lock(&self.mtp_lanes).take();
                let mut drafter = self.drafter(lanes)?;
                let out = s.decode_speculative_until(
                    rewindable::<P>()?,
                    &mut drafter,
                    prompt_ids,
                    detok,
                    params,
                    |f| on_fragment(f),
                );
                *lock(&self.mtp_lanes) = Some(drafter.into_lanes());
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
        if batch > 1 || self.sched.is_some() {
            return self.prefill_throughput_rows(n, batch.max(1));
        }
        let ids = self.seed_prompt_row(n, 0);
        let device = self.device.clone();
        let tile = self.prefill_tile;
        let gauge = MemGauge::start(&device, self.mem_sample);
        let (prefill, forwards) = self.with_single(|s| {
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
        if batch > 1 || self.sched.is_some() {
            return self.decode_throughput_rows(prompt_tokens, decode_tokens, batch.max(1));
        }
        let prompt = self.seed_prompt_row(prompt_tokens, 0);
        let device = self.device.clone();
        let readback = self.readback;
        let prefill_tile = self.prefill_tile;
        let detok = self.detok();
        let gauge = MemGauge::start(&device, self.mem_sample);
        let mut timing = self.with_single(|s| {
            s.reset();
            if self.mtp || self.dflash.is_some() {
                // One run of `D` tokens through the speculative body; the
                // prompt's submit time subtracted (the prefill's GPU tail is
                // inside the bracket), as the LFM2 engine brackets DSpark.
                let params = DecodeParams {
                    max_tokens: decode_tokens,
                    eos_ids: NEVER_EOS.to_vec(),
                    readback,
                    ..DecodeParams::default()
                };
                super::common::profile_reset();
                let t = Instant::now();
                let run = match &self.dflash {
                    Some(d) => {
                        let mut spec = lock(d);
                        s.decode_speculative_until(
                            rewindable::<P>()?,
                            &mut *spec,
                            &prompt,
                            detok,
                            &params,
                            |_| true,
                        )?
                    }
                    None => {
                        let mut drafter = self.drafter(None)?;
                        s.decode_speculative_until(
                            rewindable::<P>()?,
                            &mut drafter,
                            &prompt,
                            detok,
                            &params,
                            |_| true,
                        )?
                    }
                };
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
                return Ok(DecodeTiming {
                    prompt_tokens,
                    decode_tokens,
                    batch: 1,
                    prefill_ms,
                    decode_ms: (wall - prefill_ms).max(0.0),
                    readback: run.readback,
                    bracket: "run wall minus the prompt's submit; the prefill's GPU tail is inside",
                    row_tokens: vec![run.tokens.len()],
                    row_rounds: vec![rounds],
                    row_accepted: vec![accepted],
                    row_draft_ms: vec![
                        run.stats
                            .rounds
                            .iter()
                            .map(|r| r.timing.draft_ns)
                            .sum::<u64>() as f64
                            / 1e6,
                    ],
                    row_verify_ms: vec![
                        run.stats
                            .rounds
                            .iter()
                            .map(|r| r.timing.verify_ns)
                            .sum::<u64>() as f64
                            / 1e6,
                    ],
                    row_observe_ms: vec![
                        run.stats
                            .rounds
                            .iter()
                            .map(|r| r.timing.observe_ns)
                            .sum::<u64>() as f64
                            / 1e6,
                    ],
                    profile: prof,
                    mem: MemReport::default(),
                });
            }
            let mut detok = detok;
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
        let logits = self.with_single(|s| {
            s.reset();
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            // No driver bracket: the model opens its own passes, keyed apart
            // from a driver's pass of the same shape (§9.2aa).
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
        if batch > self.batch {
            anyhow::bail!(
                "batch {batch} exceeds the server's --batch {} (start lloom serve with --batch >= {batch})",
                self.batch
            );
        }
        Ok(())
    }
    fn shutdown(&self) {
        if let Some(sc) = &self.sched {
            sc.shutdown();
        }
        if let Some(t) = lock(&self.sched_thread).take() {
            let _ = t.join();
        }
        // The hotlist: the file's profile plus the usage since load.
        if let Some(path) = &self.hotlist {
            use lloom::models::qwen4exp::hotlist::Hotlist;
            let mut hot = if path.exists() {
                match Hotlist::read(path) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("hotlist {}: not read ({e}); rewriting", path.display());
                        Hotlist::default()
                    }
                }
            } else {
                Hotlist::default()
            };
            hot.merge(&Hotlist::from_usage(&self.core.model.stream_usage()));
            if let Err(e) = hot.write(path) {
                eprintln!("hotlist {}: not written ({e})", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_readback;

    /// `--moe-act auto` follows the contract: f16 is a `fast` arm, f32 the
    /// `deterministic` routed arm; an explicit spelling holds under both.
    #[test]
    fn moe_act_auto_follows_the_contract() {
        use lloom_tensor::tuning::Contract::{Deterministic, Fast};
        use lloom_tensor::DType;
        assert_eq!(super::routed_act("auto", Fast).unwrap(), DType::F16);
        assert_eq!(
            super::routed_act("auto", Deterministic).unwrap(),
            DType::F32
        );
        for c in [Fast, Deterministic] {
            assert_eq!(super::routed_act("f32", c).unwrap(), DType::F32);
            assert_eq!(super::routed_act("f16", c).unwrap(), DType::F16);
        }
        assert!(super::routed_act("bf16", Fast).is_err());
        assert!(lloom_tensor::tuning::Contract::FAST_ARMS.contains(&"moe-act-f16"));
    }
    #[cfg(feature = "metal")]
    #[test]
    fn moe_auto_checks_every_routed_stack_not_dense_stems() {
        use lloom_tensor::quantized::GgmlDType::{F32, Q4_0, Q8_0};
        assert!(super::routed_half_supported([
            ("blk.0.ffn_gate_exps.weight", Q4_0),
            ("blk.0.ffn_up_exps.weight", Q4_0),
            ("blk.0.ffn_down_exps.weight", Q8_0),
            ("blk.0.attn_norm.weight", F32),
        ]));
        for missing in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let name = format!("blk.99.{missing}");
            assert!(!super::routed_half_supported([
                ("blk.0.ffn_gate_exps.weight", Q4_0),
                (name.as_str(), F32),
            ]));
        }
        assert!(!super::routed_half_supported(std::iter::empty()));
    }
    use crate::{ReadbackArm, RingArm};
    use lloom::generation::EosRead;

    /// `auto` is the blocking arm, fill or no fill, and the note names the
    /// ring arm that feeds the PLE from the host instead.
    #[test]
    fn auto_is_blocking_and_names_the_filled_ring() {
        for no_fill in [None, Some("no PLE is attached")] {
            let (arm, note) = resolve_readback(ReadbackArm::Auto, no_fill);
            assert_eq!(arm, ReadbackArm::Blocking);
            let note = note.expect("the load says what auto is");
            assert!(
                note.contains("blocking")
                    && note.contains("ring-fixed")
                    && note.contains("§20.13c"),
                "{note}"
            );
        }
    }

    /// An explicit arm is served as asked; a ring arm's note says whether the
    /// host fill feeds it or its forward reads the id back, and why.
    #[test]
    fn an_explicit_ring_arm_says_whether_the_fill_feeds_it() {
        for ring in [ReadbackArm::CROWN, ReadbackArm::RING_FIXED] {
            let (arm, note) = resolve_readback(ring, None);
            assert_eq!(arm, ring);
            let note = note.unwrap();
            assert!(
                note.contains("host fill") && note.contains("misses"),
                "{note}"
            );
        }
        let drain = ReadbackArm::Ring(RingArm {
            eos: EosRead::HostOnDrain,
            ..RingArm::FIXED
        });
        let note = resolve_readback(drain, None).1.unwrap();
        assert!(
            note.contains("reads its fed id back") && note.contains("deferred"),
            "{note}"
        );
        let note = resolve_readback(ReadbackArm::RING_FIXED, Some("no PLE is attached"))
            .1
            .unwrap();
        assert!(
            note.contains("reads its fed id back") && note.contains("no PLE is attached"),
            "{note}"
        );
        assert_eq!(
            resolve_readback(ReadbackArm::Blocking, None),
            (ReadbackArm::Blocking, None)
        );
    }
}
