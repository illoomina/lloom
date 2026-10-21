//! The LFM2 engine: the loaded model and the operations every endpoint is a
//! thin shape over (the former `lloom-serve/engine.rs`, moved verbatim; what
//! a second family shares with it lives in `common.rs`).
//!
//! Two serving shapes, selected at load by `--batch`:
//!
//! - **`--batch 1`** (the default): one sequence, one request at a time, the
//!   crown decode loop on the request's thread (`DecodeSession` over the
//!   device-resident ring) — the incumbent path, byte for byte. With
//!   `--dspark` the same shape runs the speculative body.
//! - **`--batch N`**: the continuous-batching scheduler (`lloom_engine::batch`)
//!   on its own thread over a pool of `N` sequences; requests from any number
//!   of client threads join the running batch and leave it at their EOS. The
//!   benchmark cells run as closed jobs on that thread with the engine to
//!   themselves. With `--dspark` the scheduler runs static batches through
//!   the pipelined speculative driver.
//!
//! Every operation here is expressed over `lloom`'s tier-1 API and the
//! `lloom_engine` seams — nothing reaches into a model's internals
//! (RESTRUCTURING §7.1b).

use super::common::{
    lock, model_name_from_dir, ms, newest_snapshot, open_device, profile_reset, profile_take,
    DecodeTiming, MemGauge, MemReport, MemSnapshot, PrefillTiming, Text, NEVER_EOS,
};
use super::request_slot::{AbortState, Lease as RequestLease, State as RequestState};
use super::template::{Rendered, Template};
use super::vision::{take_window, DecodedImage, ImageCensus, ImageRows};
use crate::{
    ring_eos_read, spawn_batch, BatchConfig, BatchEvent, BatchHandle, BatchMode, BatchRequest,
    DecodeParams, DecodeSession, Decoded, Idle, ReadbackArm, RingPool, RowSpec, SlotPool, Stop,
    TextStream,
};
use anyhow::{Context, Result};
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::load_warning::{LoadWarning, LoadWarningKind};
use lloom::models::lfm2::{
    AttnImpl, BatchPool, Config, KvAppend, Lfm2Config, MemoryBudget, Model, Shape as Lfm2Shape,
    State,
};
use lloom::models::lfm2_dspark::{CtxKv, DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::models::lfm2_presets as presets;
use lloom::models::lfm2_vl::{ImageProcessor, Lfm2VlConfig, VisionEncoder};
use lloom::nn::var_builder::{LoadRecord, RecordingBackend};
use lloom::nn::{DecodeOptions, DecodeOut, ReuseAction, Sequence, Session, VarBuilder};
use lloom::profile::{Profile, ProfileKey};
use lloom_tensor::{DType, Device, Tensor, D};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// The model and the device: the half every request borrows, shared with
/// the batch thread.
pub struct Core<P: Profile> {
    pub model: Model<P>,
    pub device: Device,
}

/// Side state travels with the held sequence under one request checkout.
struct SingleExtra {
    spec: Option<DSparkSpeculator>,
    decode: crate::DecodeConfig,
    /// Pixel keys for the rows in the held prefix, not merely image token ids.
    image_keys: Vec<[u8; 32]>,
}
impl AbortState for SingleExtra {
    fn abort(&mut self) {
        self.image_keys.clear();
        if let Some(spec) = &mut self.spec {
            spec.reset_contexts();
        }
    }
}

/// The vision half of an LFM2-VL checkpoint: its config and preprocessing
/// from load, the tower and projector only once an image arrives, so a
/// process that never sees one holds the language model's tensors alone
/// (`DESIGN.md` §5.5a).
pub struct Vision {
    config: Lfm2VlConfig,
    processor: ImageProcessor,
    files: Vec<PathBuf>,
    dtype: DType,
    text_hidden: usize,
    /// The load's execution purpose, applied to the tower when it loads.
    purpose: lloom::nn::attention::route::ExecutionPurpose,
    tower: Mutex<Option<Tower>>,
}

/// The tower and projector once loaded, and what loading them cost.
struct Tower {
    encoder: Arc<VisionEncoder>,
    record: LoadRecord,
    load_ms: f64,
    pool_delta: i64,
    /// The tower's row on the device's ledger, for as long as it is held.
    #[cfg(feature = "metal")]
    _tenant: Option<Arc<lloom_tensor::kernels::metal::ledger::Gauge>>,
}

impl Vision {
    /// The encoder, loading it on first use.
    fn encoder(&self, device: &Device) -> Result<Arc<VisionEncoder>> {
        let mut tower = lock(&self.tower);
        if let Some(t) = tower.as_ref() {
            return Ok(Arc::clone(&t.encoder));
        }
        device.synchronize()?;
        let before = MemSnapshot::read(device).pool_live as i64;
        let start = Instant::now();
        // SAFETY: the checkpoint is not modified while mapped, the contract
        // the language model's own load relies on.
        let (backend, record) = unsafe { RecordingBackend::over_safetensors(&self.files)? };
        let vb = VarBuilder::from_backend(Box::new(backend), self.dtype, device.clone());
        let encoder = VisionEncoder::new(&self.config, self.text_hidden, vb)
            .context("loading the vision tower and projector")?;
        self.purpose
            .admit("lfm2-vl", device, encoder.reference_attention())?;
        device.synchronize()?;
        let pool_delta = MemSnapshot::read(device).pool_live as i64 - before;
        let t = Tower {
            encoder: Arc::new(encoder),
            record,
            load_ms: ms(start.elapsed()),
            pool_delta,
            #[cfg(feature = "metal")]
            _tenant: match device {
                Device::Metal(m) => {
                    use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
                    let held = pool_delta.max(0) as usize;
                    let g = Gauge::new(
                        "lfm2-vision-tower",
                        Claim {
                            floor: held,
                            want: held,
                            unit: held.max(1),
                            class: Class::Capability,
                        },
                        held,
                    );
                    let t: Arc<dyn Tenant> = g.clone();
                    m.ledger().register(&t);
                    Some(g)
                }
                _ => None,
            },
        };
        let encoder = Arc::clone(&t.encoder);
        *tower = Some(t);
        Ok(encoder)
    }

    fn tower_json(&self) -> Value {
        match lock(&self.tower).as_ref() {
            None => json!({ "loaded": false }),
            Some(t) => {
                let (n, bytes) = t.record.totals();
                json!({
                    "loaded": true,
                    "tensors": n,
                    "bytes": bytes,
                    "load_ms": t.load_ms,
                    "pool_delta_bytes": t.pool_delta,
                })
            }
        }
    }
}

/// The two ways a VL checkpoint differs from a text one on disk: the language
/// config nests under `text_config` (with the token ids beside it at the top
/// level), and the language tensors under `model.language_model.`.
fn text_config_of(root: &Value) -> Result<Lfm2Config> {
    let mut text = root
        .get("text_config")
        .and_then(Value::as_object)
        .context("config.json: a vision_config without a text_config")?
        .clone();
    for key in ["bos_token_id", "eos_token_id"] {
        if !text.contains_key(key) {
            if let Some(v) = root.get(key) {
                text.insert(key.into(), v.clone());
            }
        }
    }
    serde_json::from_value(Value::Object(text)).context("parsing config.json's text_config")
}

/// The batch thread's ONE drafter (`--dspark --batch N`). On the pool arms
/// the continuous speculative driver takes it at start (`MakeDriver`) and
/// the closed cells run THROUGH that driver (`Idle::driver`,
/// `drive_closed_rows`); on the private arm the static pipelined body and
/// the closed cells borrow it here. One drafter's weights either way
/// (`batched-decode-design.md` §12.9).
pub struct Speculators {
    pub spec: Option<DSparkSpeculator>,
    /// Whether the scheduler's continuous driver is the speculative loop
    /// (the pool arms): `make_driver` then takes `spec` for it.
    pub continuous: bool,
    /// The continuous driver's EOS set (the verify's accept rule reads one
    /// set per batch).
    pub eos_ids: Vec<u32>,
}

impl<P: Profile> crate::MakeDriver<Model<P>> for Speculators {
    fn make_driver<'m>(
        &mut self,
        model: &'m Model<P>,
        opts: &DecodeOptions,
        device: &Device,
    ) -> lloom_tensor::Result<Option<crate::Driver<'m, Model<P>>>> {
        let _ = model;
        if !self.continuous {
            return Ok(None);
        }
        match self.spec.take() {
            Some(spec) => {
                // The drafter needs a rewindable state: the `Ring` shapes
                // give the evidence, the `No` shapes refuse by name.
                let Some(rewind) = lloom::profile::RewindProof::<P>::ask() else {
                    lloom_tensor::bail!(
                        "a DSpark drafter on the {} shape: the speculative loop needs `Ring` \
                         (`--cache rewindable` / `pool-spec`)",
                        P::KEY
                    )
                };
                let dec = lloom::nn::SpecBatchDecoder::<
                    'm,
                    Model<P>,
                    DSparkSpeculator,
                    Session<'m, Model<P>>,
                >::new(rewind, spec, opts.clone(), device.clone())?;
                Ok(Some(Box::new(dec)))
            }
            None => Ok(None),
        }
    }
}

pub type Extra = Speculators;

/// The private-sequence policy shared by initial chat and decision caches.
/// Pooled batch rows retain their separate shared-pool constructor.
#[derive(Clone, Copy)]
struct PrivateCachePolicy {
    capacity: usize,
    bucket: usize,
    grow_fraction: Option<f64>,
}

impl PrivateCachePolicy {
    fn create<P: Lfm2Shape>(
        self,
        config: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<State<P>> {
        let mut cache = P::state(config, dtype, device, self.capacity, None)
            .context("allocating the KV cache")?
            .with_kv_bucket(self.bucket)
            .context("memory.kv_bucket")?;
        if let Some(fraction) = self.grow_fraction {
            cache = cache
                .with_kv_grow_fraction(fraction)
                .context("memory.kv_grow_fraction")?;
        }
        Ok(cache)
    }
}

pub struct Engine<P: Profile> {
    pub core: Arc<Core<P>>,
    pub text: Arc<Text>,
    pub model_name: String,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    /// The resolved axes (`Name=Arm …`), read back from the config that
    /// loaded — the report line's, never the flags' (§7.1a-i).
    pub axes_line: String,
    /// The model group as requested (the preset and the overrides).
    pub model: crate::ModelConfig,
    /// The checkpoint's `generation_config.json`, when it carries one.
    pub generation: Option<lloom::hub::GenerationConfig>,
    /// The executor arms installed at load.
    pub executor: crate::ExecutorConfig,
    /// Reuse the cached prefix across requests (`Session::prepare`). Off by
    /// default: a benchmark cell measures a cold prefill, and a repeated prompt
    /// would otherwise resume from the previous request's state.
    pub reuse_prefix: bool,
    /// The batch width `N` of `--batch N`; `1` is the single-sequence path.
    pub batch: usize,
    /// The draft block width under `--dspark`: the drafter's block, or
    /// `--dspark-k` (clamped to the block).
    pub dspark_k: Option<usize>,
    pub prefill_tile: usize,
    pub prefill_chunk: usize,
    /// The pending rows' chunks one batched step carries, as resolved at
    /// load (`scheduler.prefill_batch`, bounded by the admission's room).
    pub prefill_batch: PrefillBatch,
    /// What the live rows do in a step that feeds prompt pieces.
    pub prefill_decode: lloom::nn::PrefillDecode,
    pub memory_fraction: f64,
    pub mem_sample: bool,
    /// How the rows of a batched step hold their state (`--rows`).
    pub rows: RowsArm,
    /// The rows' shared pool on the `pool` arms (`batched-decode-design.md`
    /// §12.2); `None` on `private` and at `--batch 1`.
    pub pool: Option<BatchPool>,
    /// `--share-prefix`.
    pub share_prefix: bool,
    /// The language model's load: which of the checkpoint's tensors it named.
    pub lm_record: LoadRecord,
    /// The vision half, on a VL checkpoint.
    pub vision: Option<Vision>,
    /// What the last request loaded and encoded (`Served::vision_report`).
    last_vision: Mutex<Value>,
    /// The resolved model config: what the bench modes build their own
    /// caches and pools from.
    pub config: Config,
    pub dtype: DType,
    /// The drafter's directory and config under `--dspark`.
    pub dspark_setup: Option<(PathBuf, DSparkConfig)>,
    /// The executor arms armed at load, for their engagement reports.
    pub installed: super::report::Installed,
    /// What warmup measured: the in-flight depth and the `(host_ns, gpu_ns)`
    /// balance it implies (the pacer takes `decode.in_flight`, not this).
    pub warm_in_flight: Option<(usize, (u64, u64))>,
    /// The load's degradations: `load_engine`'s notes, the model's, this
    /// engine's (`Served::load_warnings`).
    load_warnings: Vec<LoadWarning>,
    /// What the load reserved on the machine, released with the engine.
    pub machine: Option<Arc<lloom_tensor::machine::Holding>>,
    private_cache: PrivateCachePolicy,
    /// `/v1/systemone`'s executor binding: a native d1 checkpoint's own, or
    /// the template `decision.template` selected; `None` refuses by name.
    decisions: Option<ServedDecisions>,
    /// `scheduler.decision_ragged`, and the state the last ragged read kept.
    decision_ragged: bool,
    trunks: crate::decision::TrunkCache<lloom::models::lfm2::RetainedState>,
    single: parking_lot::Mutex<Option<RequestState<State<P>, SingleExtra>>>,
    sched: Option<BatchHandle<Model<P>, Extra>>,
    sched_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

pub struct LoadOptions {
    pub model_dir: PathBuf,
    pub cpu: bool,
    /// What the run is for (`execution.purpose`): production refuses at load
    /// an attention that takes the composed scores path on Metal.
    pub purpose: lloom::nn::attention::route::ExecutionPurpose,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    /// The decode loop's options (`DecodeConfig`): what `readback` and the
    /// prefill fields were resolved from, and the session's cadence.
    pub decode: crate::DecodeConfig,
    /// The model's axes: the cache preset and the per-axis overrides.
    pub model: crate::ModelConfig,
    pub reuse_prefix: bool,
    pub warmup: bool,
    pub seed_text: String,
    pub model_name: Option<String>,
    pub batch: usize,
    /// `Some(dir)` loads the DSpark drafter from `dir` (or the newest
    /// LFM2.5-2.6B-DSpark snapshot in the HF cache when the path is empty).
    pub dspark: Option<PathBuf>,
    pub prefill_tile: usize,
    pub prefill_chunk: usize,
    /// The fraction of `recommendedMaxWorkingSetSize` the process may claim
    /// (`MemoryBudget`, §9.5d); `0` turns admission off — the unguarded arm.
    pub memory_fraction: f64,
    /// Sample the memory gauges at 10 ms around every cell and report the
    /// peaks. A gauge run, not a timing run: the sampler perturbs TTFT
    /// (issue #728's `--mem-sample` finding).
    pub mem_sample: bool,
    /// How a batched step's rows hold their state (`--rows`).
    pub rows: RowsArm,
    /// Share a request's prefix from another row's cache by page (§12.3);
    /// the `pool` arms only.
    pub share_prefix: bool,
    /// `--dspark-k`: draft this many positions per round instead of the
    /// drafter's block (clamped to it).
    pub dspark_k: Option<usize>,
    /// The executor arms, installed on the device before the model.
    pub executor: crate::ExecutorConfig,
    /// The memory budgets (the KV bucket and grow fraction, the free-list budget).
    pub memory: crate::MemoryConfig,
    /// `--flash-k`: pages per FlashDecoding chunk (the model's
    /// `flash_pages_per_chunk`), under `--attn flash` only.
    pub flash_k: Option<usize>,
    /// `decision.template`: the systemone template the decision executor
    /// renders, when the checkpoint is not a native decision model or a
    /// different template is wanted (`decision::binding::Selection`).
    pub decision_template: Option<String>,
    /// `scheduler.decision_ragged`: read a request's questions as one ragged
    /// batch over its state (`CausalRequest::execute_ragged`), under `fast`.
    pub decision_ragged: bool,
    /// `model.text_only`: build no vision tower on a VL checkpoint.
    pub text_only: bool,
}

/// How the rows of `--batch N` hold their KV and conv state, and how a step
/// reads them (`batched-decode-design.md` §12.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowsArm {
    /// Every row a slot of one `BatchPool` (`Paged × FlashDecoding ×
    /// RotatingRing`); the attention and the conv of the whole batch in one
    /// dispatch per kernel. The default at `--batch N`.
    Pool,
    /// The same pool, the per-row loop over it — the A/B arm, one dispatch
    /// per row per kernel, the same arithmetic.
    PoolPerRow,
    /// A private crown cache per row (`Sdpa × InPlace × Shuffle`), the
    /// per-row loop — #742's arm.
    Private,
}

impl RowsArm {
    pub fn parse(s: &str, batched: bool) -> Result<Self> {
        Ok(match s {
            // `--batch N` is the pool on both loops: the continuous
            // speculative loop on the pool (the batched draft and re-key,
            // one drafter) beats the static pipelined driver on private
            // caches on every measured cell (`batched-decode-design.md`
            // §12.10); `private` stays the named control arm.
            "auto" => {
                if batched {
                    Self::Pool
                } else {
                    Self::Private
                }
            }
            "pool" => Self::Pool,
            "pool-per-row" => Self::PoolPerRow,
            "private" => Self::Private,
            other => {
                anyhow::bail!("--rows must be auto, pool, pool-per-row or private, got {other:?}")
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pool => "pool",
            Self::PoolPerRow => "pool-per-row",
            Self::Private => "private",
        }
    }

    pub fn pooled(self) -> bool {
        matches!(self, Self::Pool | Self::PoolPerRow)
    }
}

/// The evidence the speculative drivers take: the `Ring` shapes give it,
/// the `No` shapes refuse by name (`profiles-and-memory-design.md` §7.3).
fn rewindable<P: Profile>() -> lloom_tensor::Result<lloom::profile::RewindProof<P>> {
    lloom::profile::RewindProof::<P>::ask().ok_or_else(|| {
        lloom_tensor::Error::msg(format!(
            "the speculative body on the {} shape: it needs `Ring` (`--cache rewindable` / `pool-spec`)",
            P::KEY
        ))
    })
}

/// The drafter's directory under `--dspark`: the named one, or the HF
/// cache's LFM2.5-2.6B-DSpark snapshot when none is named.
fn drafter_dir(opts: &LoadOptions) -> Result<Option<PathBuf>> {
    Ok(match &opts.dspark {
        None => None,
        Some(p) if p.as_os_str().is_empty() => Some(
            newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark").context(
                "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir; the \
                 drafter is trained against THIS target (design/docs/speculation.md §10.2l)",
            )?,
        ),
        Some(p) => Some(p.clone()),
    })
}

/// The preset `--cache` names, over `config`: `auto` is the rule below; a
/// named one is refused where it needs the drafter's config it has not
/// got. The per-axis overrides are applied over it by the caller.
fn apply_preset(
    config: &mut Config,
    opts: &LoadOptions,
    dcfg: Option<&DSparkConfig>,
    pooled: bool,
) -> Result<()> {
    match opts.model.cache.as_str() {
        "auto" => match dcfg {
            Some(dcfg) if pooled => presets::pool_spec(config, dcfg),
            Some(dcfg) => presets::rewindable(config, dcfg.max_draft(), dcfg.tap_layers()),
            None if pooled => presets::pool(config),
            None => presets::crown(config),
        },
        "crown" => presets::crown(config),
        "pool" => presets::pool(config),
        // Without a drafter the speculative body's cache runs K = 1, so
        // the cost of leaving the crown separates from the verify loop's;
        // its taps are the 2.6B drafter's (the same per-token capture), or
        // none on `rewindable-notap`, which separates the taps' cost from
        // the ring's. The ring is sized for a `--dspark-k` window (the
        // drafter's block, 9, by default).
        "rewindable" => match dcfg {
            Some(dcfg) => presets::rewindable(config, dcfg.max_draft(), dcfg.tap_layers()),
            None => {
                let two_six = DSparkConfig::lfm2_2_6b();
                let k = opts.dspark_k.unwrap_or(two_six.max_draft());
                presets::rewindable(config, k, two_six.tap_layers())
            }
        },
        "rewindable-notap" => match dcfg {
            Some(_) => anyhow::bail!(
                "--cache rewindable-notap drops the taps the drafter's projection reads; \
                 under --dspark the preset is `rewindable`"
            ),
            None => {
                let k = opts
                    .dspark_k
                    .unwrap_or(DSparkConfig::lfm2_2_6b().max_draft());
                presets::rewindable(config, k, Vec::new())
            }
        },
        "pool-spec" => match dcfg {
            Some(dcfg) => presets::pool_spec(config, dcfg),
            None => anyhow::bail!("--cache pool-spec needs --dspark"),
        },
        other => anyhow::bail!(
            "--cache {other}: auto | crown | rewindable | rewindable-notap | pool | pool-spec"
        ),
    }
    Ok(())
}

/// The profile the resolved config names (`Config::shape`, at the contract
/// in force): what `load` instantiates the engine at.
fn shape_key(opts: &LoadOptions) -> Result<ProfileKey> {
    let bytes = std::fs::read(opts.model_dir.join("config.json"))?;
    let root: Value = serde_json::from_slice(&bytes).context("parsing config.json")?;
    let config: Lfm2Config = if root.get("vision_config").is_some() {
        text_config_of(&root)?
    } else {
        serde_json::from_slice(&bytes).context("parsing config.json")?
    };
    let mut config = config.into_config(false);
    let dcfg = match drafter_dir(opts)? {
        Some(dir) => Some(
            DSparkConfig::read_from_dir(&dir)
                .with_context(|| format!("reading the drafter config in {}", dir.display()))?,
        ),
        None => None,
    };
    let pooled = opts.batch > 1 && opts.rows.pooled();
    apply_preset(&mut config, opts, dcfg.as_ref(), pooled)?;
    opts.model
        .apply_axes(&mut config)
        .map_err(|e| anyhow::anyhow!("model axes: {e}"))?;
    let (rewind, rows) = config.shape();
    Ok(ProfileKey::at_contract(rewind, rows))
}

/// The boundary (`profiles-and-memory-design.md` §7.4): the engine at the
/// profile the resolved config names, one `match` over the eight
/// (`lloom::profile::dispatch!`), bounded by what the family ships.
pub fn load(opts: LoadOptions) -> Result<Box<dyn super::served::Served>> {
    lloom_tensor::memory_admission::begin_model_load();
    let key = shape_key(&opts)?;
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => {
        Engine::<P>::load_as(opts).map(|e| Box::new(e) as Box<dyn super::served::Served>)
    })
}

/// The bounds a served `/v1/systemone` request's images decode under.
pub(crate) const SERVED_DECISION_IMAGES: super::vision::ImageDecodeLimits =
    super::vision::ImageDecodeLimits {
        max_encoded_bytes: 64 << 20,
        max_width: 16384,
        max_height: 16384,
        max_pixels: 64 << 20,
        max_decoder_bytes: 3 * (64 << 20),
    };

/// A served LFM2's decision executor: the template bound at load, before
/// any weight is read (`causal_loaded::BoundPolicy`).
type ServedDecisions = super::causal_loaded::BoundPolicy;

fn served_decisions(config: &Value, selection: Option<&str>) -> Result<Option<ServedDecisions>> {
    let facts = crate::decision::binding::CheckpointDecision::from_config(config)?;
    Ok(ServedDecisions::resolve(&facts, selection)?)
}

#[cfg(test)]
mod served_decisions_tests {
    use super::served_decisions;
    use crate::decision::binding::DecisionType;
    use serde_json::json;

    fn d1() -> serde_json::Value {
        json!({"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1Model"}})
    }

    /// A native d1-3B config binds its shipped template with no selection; a
    /// plain LFM2-VL binds nothing unless `decision.template` names one, and
    /// then it is not native (so `/v1/models` does not advertise decisions).
    #[test]
    fn native_config_binds_itself_and_a_plain_one_only_when_selected() {
        let native = served_decisions(&d1(), None).unwrap().unwrap();
        assert_eq!(native.native, Some(DecisionType::Lfm2D1));
        assert!(native
            .origin
            .starts_with("lfm2-d1 (engine-shipped, sha256 b47201d7"));
        assert_eq!(native.policy.max_options, 255);
        let plain = json!({"model_type": "lfm2_vl"});
        assert!(served_decisions(&plain, None).unwrap().is_none());
        assert!(served_decisions(&plain, Some("generic")).unwrap().is_none());
        let selected = served_decisions(&plain, Some("lfm2-d1")).unwrap().unwrap();
        assert_eq!(selected.native, None);
        assert!(selected.policy.template.has_causal_certificate());
    }

    /// What this family cannot render is refused at load, by name: the omni
    /// layout, and a template the shared-state renderer was not audited for.
    #[test]
    fn omni_and_uncertified_templates_are_refused_at_load() {
        let omni = json!({"model_type": "d1_omni"});
        let error = served_decisions(&omni, None).err().unwrap();
        assert!(
            format!("{error:#}").contains("lfm2-d1-omni decision model"),
            "{error:#}"
        );
        let dir = std::env::temp_dir().join(format!("lloom947-template-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("custom.jinja");
        std::fs::write(&path, "{{ state }}QUESTION:{{ instructions }}").unwrap();
        let error = served_decisions(&d1(), path.to_str()).err().unwrap();
        assert!(
            format!("{error:#}").contains("not byte-identical"),
            "{error:#}"
        );
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}

/// Internal choice only: CallerScoped is reached exclusively inside Loaded::new.
#[derive(Clone, Copy)]
enum LoadAdmission {
    Legacy,
    CallerScoped,
}

/// Checked cold witness for the incumbent shutdown's worker-stop semantics.
/// It certifies neither native DATA nor callback/retirement quiescence.
struct NoWorkerWitness(());
impl NoWorkerWitness {
    fn shutdown(&self) {} // No batch worker was constructed; no fresh budget gate.
}

pub mod admitted;

impl<P: Lfm2Shape> Engine<P> {
    /// The engine at the shape `P` names; `load` picks `P` from the
    /// resolved config (`shape_key`) and dispatches here.
    pub fn load_as(opts: LoadOptions) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        let device = if opts.cpu {
            Device::Cpu
        } else {
            open_device()?
        };
        Self::load_on(opts, device, LoadAdmission::Legacy)
    }

    /// Additive embedding factory using one caller-supplied model domain.
    /// The caller admits checkpoint/header metadata and the complete load peak
    /// BEFORE this entry; no second `MachineLoad::admit` runs inside it.
    ///
    /// Source integration only: allocator/native/host producer enrollment must
    /// be complete before enabling this in the default CLI/server factory. All
    /// subsequent chat AND decision work must use the returned owner's pass
    /// scope; extracting an unscoped engine is deliberately not offered.
    ///
    /// The current admitted variant is single-sequence: the batch scheduler's
    /// cross-thread pass propagation is still pending. Incumbent batch/CPU/CUDA
    /// routes remain available unchanged through `load_as`.
    pub fn load_as_admitted(
        opts: LoadOptions,
        device: &Device,
        load: lloom::nn::memory::LoadContext,
    ) -> Result<lloom::nn::memory::Loaded<Self>> {
        Self::load_as_admitted_with(opts, device, load, |_, _| ()).map(|(loaded, ())| loaded)
    }

    // Private load-time capture only: no unscoped getter or owner extraction.
    fn load_as_admitted_with<T>(
        opts: LoadOptions,
        device: &Device,
        load: lloom::nn::memory::LoadContext,
        capture: impl FnOnce(&Self, NoWorkerWitness) -> T,
    ) -> Result<(lloom::nn::memory::Loaded<Self>, T)> {
        Self::load_as_admitted_with_preflight(opts, device, load, |_, _| Ok(()), capture)
    }

    fn load_as_admitted_with_preflight<T>(
        opts: LoadOptions,
        device: &Device,
        load: lloom::nn::memory::LoadContext,
        preflight: impl FnOnce(&Config, &Text) -> Result<()>,
        capture: impl FnOnce(&Self, NoWorkerWitness) -> T,
    ) -> Result<(lloom::nn::memory::Loaded<Self>, T)> {
        lloom_tensor::memory_admission::begin_model_load();
        if opts.cpu != device.is_cpu() {
            anyhow::bail!("supplied LFM2 device disagrees with load options");
        }
        if opts.batch != 1 {
            anyhow::bail!("admitted LFM2 batch scheduler pass propagation is pending");
        }
        let mut metadata = None;
        let loaded = lloom::nn::memory::Loaded::new(device, load, |resolved, _context| {
            let engine = Self::load_on_with_preflight(
                opts,
                resolved.clone(),
                LoadAdmission::CallerScoped,
                preflight,
            )
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))?;
            // The single-sequence branch returns before spawn_batch. Check the
            // actual owner too, so a future constructor cannot invalidate the
            // wrapper's narrowed shutdown without a refusal here.
            if engine.sched.is_some() || lock(&engine.sched_thread).is_some() {
                return Err(lloom_tensor::Error::msg(
                    "admitted LFM2 worker shutdown custody is not installed",
                ));
            }
            metadata = Some(capture(&engine, NoWorkerWitness(())));
            Ok(engine)
        })?;
        Ok((
            loaded,
            metadata.expect("successful admitted load ran metadata capture"),
        ))
    }

    /// Chat on the admitted loaded owner, using the incumbent decode loop and
    /// held chat cache. `predicted_extra` covers the complete asynchronous pass;
    /// borrowed prompt/callback allocations retain their own origin. This does
    /// not enable the default server factory before producer enrollment.
    pub fn complete_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        Ok(loaded.with_pass(predicted_extra, |engine, _context| {
            engine
                .complete_until(prompt_ids, params, on_fragment)
                .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
        })?)
    }

    /// Decisions on the SAME loaded model as `complete_admitted`, without
    /// touching its held chat sequence or loading another checkpoint. The
    /// already-prepared request's host storage keeps its origin; the pass must
    /// cover a fresh configured cache, retained frames and all readout scratch.
    /// A common request checkout excludes both callers from the model executor.
    /// Media expansion, raw HTTP rendering and full producer enrollment remain
    /// factory/frontend obligations, not implicit behavior of this token API.
    pub fn decide_admitted(
        loaded: &lloom::nn::memory::Loaded<Self>,
        predicted_extra: usize,
        request: &crate::decision::CausalRequest,
    ) -> Result<crate::decision::SystemOneResponse> {
        Ok(loaded.with_pass(predicted_extra, |engine, _context| {
            engine.decide_in_scope(request)
        })?)
    }

    // Both prepared and raw callers enter exactly one existing Loaded pass.
    fn decide_in_scope(
        &self,
        request: &crate::decision::CausalRequest,
    ) -> lloom_tensor::Result<crate::decision::SystemOneResponse> {
        self.check_fits(request.max_prompt_tokens(), 0)
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))?;
        let mut lease = self
            .checkout_single()
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))?;
        // Fresh decision caches still touch the same model scratch. Keep
        // Held exclusion and mark recovery on Err/unwind before any work.
        lease.start_external_work();
        let options = lease.extra().decode.options(0, ReadbackArm::Blocking, 0);
        let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, &self.core.device)?;
        let result = match self.ragged_config() {
            Some(config) => request.execute_ragged(
                &self.core.model,
                &self.core.model,
                &mut readback,
                &options,
                |_model, device| self.decision_cache(&config, device),
                Some(&self.trunks),
                true,
            ),
            None => request.execute_with_cache_factory(
                &self.core.model,
                &mut readback,
                &options,
                |_model, device| self.decision_cache(&self.config, device),
            ),
        };
        if result.is_ok() {
            lease.mark_complete();
        }
        result
    }

    /// The decision caches' config when the `decision-ragged` arm is taken
    /// (`scheduler.decision_ragged` under `fast`): the served config with
    /// Cat KV, the arm whose appends never write into the trunk the rows
    /// share (`AdoptRetained`).
    fn ragged_config(&self) -> Option<Config> {
        (self.decision_ragged && lloom_tensor::tuning::contract().is_fast()).then(|| {
            let mut config = self.config.clone();
            config.kv_append = lloom::models::lfm2::KvAppend::Cat;
            config
        })
    }

    fn decision_cache(&self, config: &Config, device: &Device) -> lloom_tensor::Result<State<P>> {
        self.private_cache
            .create::<P>(config, self.dtype, device)
            .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
    }

    fn load_on(opts: LoadOptions, device: Device, admission: LoadAdmission) -> Result<Self> {
        Self::load_on_with_preflight(opts, device, admission, |_, _| Ok(()))
    }

    fn load_on_with_preflight(
        mut opts: LoadOptions,
        device: Device,
        admission: LoadAdmission,
        preflight: impl FnOnce(&Config, &Text) -> Result<()>,
    ) -> Result<Self> {
        let model_dir = opts.model_dir.clone();
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        if opts.batch == 0 {
            anyhow::bail!("--batch must be positive");
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let config_bytes = std::fs::read(model_dir.join("config.json"))?;
        let root: Value = serde_json::from_slice(&config_bytes).context("parsing config.json")?;
        let vl = root.get("vision_config").is_some();
        let decisions = served_decisions(&root, opts.decision_template.as_deref())?;
        let config: Lfm2Config = if vl {
            text_config_of(&root)?
        } else {
            serde_json::from_slice(&config_bytes).context("parsing config.json")?
        };
        let mut config = config.into_config(false);

        // The drafter's config first: the target's cache arms depend on it
        // (the verify window sizes the conv ring, the taps feed the drafter).
        let dspark_dir = drafter_dir(&opts)?;
        // The speculative arm's pool budget: 1024 MiB, sized from measurement
        // (#148, the B=1 defaults pass):
        // a round is two models' activations plus the verify's logits,
        // released a few rounds later under the nonblocking readback, and at
        // the 256 MiB ceiling that working set is evicted as it frees — ~10
        // allocations and a residency commit per round; at 1024, none.
        #[cfg(feature = "metal")]
        if let Some(mb) = opts
            .memory
            .free_budget_mb
            .or(dspark_dir.is_some().then_some(1024))
        {
            if let Device::Metal(m) = &device {
                m.set_free_budget(mb * 1024 * 1024);
            }
        }
        // What is installed is what the load reports (`ExecutorConfig::under_body`).
        let (executor, executor_note) = opts.executor.under_body(dspark_dir.is_some());
        opts.executor = executor;
        let mut installed =
            super::common::install_executor(&device, &opts.executor, dspark_dir.is_some())?;
        installed.declined = executor_note;
        installed.pass_widths = match opts.decode.prefill.pass_widths {
            lloom::nn::PassWidths::Lru => "lru",
            lloom::nn::PassWidths::Ladder => "ladder",
        };
        // Measured on natural text (#148, record §8.3): the speculative loop
        // on the pool is +14 % at B=4 and a LOSS at B ≥ 8 (−7 % / −40 % at
        // 8 / 16, 2.1–2.4 accepted of 9 per round against a B × 10-position
        // verify); structured prompts (7.8–8.3 accepted) pay at every B.
        // Said once at load; the flag is honoured either way.
        let mut load_warnings = Vec::new();
        load_warnings.extend(installed.declined_warning());
        if dspark_dir.is_some() && opts.batch >= 16 && opts.dspark_k.is_none() {
            let note = format!(
                "at --batch {} the speculative loop measured slower than plain batched decode \
                 on natural text even at the batch's draft length (K=3: −13 % at 16; +52 % at \
                 4, +14 % at 8); it pays where the drafter's acceptance is high (structured, \
                 repetitive context). Serving it as asked.",
                opts.batch
            );
            eprintln!("[dspark: {note}]");
            load_warnings.push(LoadWarning::new(LoadWarningKind::Speculation, note));
        }
        let dcfg = match &dspark_dir {
            Some(dir) => {
                let dcfg = DSparkConfig::read_from_dir(dir)
                    .with_context(|| format!("reading the drafter config in {}", dir.display()))?;
                if dcfg.hidden_size != config.hidden_size {
                    anyhow::bail!(
                        "drafter hidden {} != target hidden {} — the drafter is not this target's \
                         (design/docs/speculation.md §10.2l)",
                        dcfg.hidden_size,
                        config.hidden_size
                    );
                }
                Some(dcfg)
            }
            None => None,
        };
        let pooled = opts.batch > 1 && opts.rows.pooled();
        if opts.share_prefix && !pooled {
            anyhow::bail!("--share-prefix needs `--batch N --rows pool` (the rows share one pool)");
        }
        // The preset: `auto` is the rule below; a named one is refused where
        // it needs the drafter's config it has not got. Then the per-axis
        // overrides (`--attn`, `--kv-append`, … or `--set
        // model.<axis>=<arm>`), applied over it in key order.
        apply_preset(&mut config, &opts, dcfg.as_ref(), pooled)?;
        opts.model
            .apply_axes(&mut config)
            .map_err(|e| anyhow::anyhow!("model axes: {e}"))?;
        refuse_kv_quant_shape(&config, &device, opts.batch)?;
        if let Some(w) = paged_verify_warning(&config, dcfg.is_some(), opts.batch) {
            eprintln!("[dspark: {}]", w.message);
            load_warnings.push(w);
        }
        // The route the model resolves at construction, admitted for the
        // run's purpose before any weight is read.
        let route =
            lloom::models::lfm2::AttnRoute::resolve(config.attn_impl, &config, dtype, &device)?;
        opts.purpose
            .admit("lfm2", &device, route.composed_reason(&device).as_deref())?;
        if let Some(k) = opts.flash_k {
            if config.attn_impl != AttnImpl::FlashDecoding {
                anyhow::bail!(
                    "--flash-k sizes FlashDecoding's chunk; this shape's attention is {:?} \
                     (select it with --attn flash)",
                    config.attn_impl
                );
            }
            config.flash_pages_per_chunk = k.max(1);
        }
        config.conv_prefill_ctx = conv_carry_needed(
            &config,
            opts.prefill_tile,
            opts.prefill_chunk,
            &opts.decode,
            opts.reuse_prefix,
        );
        let axes_line = config.axes_line();
        // Stop on the checkpoint's declared EOS plus `<|im_end|>` (the chat
        // turn end).
        let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        let bos_id = tokenizer.token_to_id("<|startoftext|>");
        let bos_token = bos_id
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();

        let template = match std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
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

        // Prepare the SAME Text once, before weight mapping/model/cache creation.
        // Executor/device setup above and host metadata remain cold producers;
        // this ordering is not a complete constructor/error-custody certificate.
        let text = Arc::new(Text {
            tokenizer,
            template,
            bos_token,
            bos_id,
            eos_ids,
            seed_tokens,
            think: Default::default(),
            gen_prompts: Default::default(),
        });
        text.prime();
        preflight(&config, &text)?;

        // Admission (§9.5d): predict the peak at the configured ceiling and
        // refuse at load rather than find the limit by dying. The activation
        // term is a function of the LONGEST forward this process will run
        // (#306): the tile when one is set, else the whole KV ceiling — a
        // one-shot prefill's working set grows with the prompt (~3.5 MB per
        // token on the crown, #728), which is what a 32k-token one-shot
        // prefill turned into a kernel panic on this machine (#148).
        let weight_bytes: usize = weight_files(&model_dir)?
            .iter()
            .map(|f| std::fs::metadata(f).map(|m| m.len() as usize).unwrap_or(0))
            .sum();
        if opts.memory_fraction > 0.0 {
            let mut budget = MemoryBudget::new(weight_bytes);
            budget.batch = opts.batch;
            budget.fraction = opts.memory_fraction;
            // The longest forward this process runs: the tile — or the chunk,
            // when the batched steps feed more prompt tokens than one tile per
            // step (`--prefill-chunk` above the tile fuses several rows' pieces
            // into one forward, whose working set is the chunk's).
            budget = budget.with_prefill(step_prompt_tokens(
                opts.prefill_tile,
                opts.prefill_chunk,
                1,
                opts.kv_capacity,
            ));
            if opts.batch > 1 {
                // The scheduler's step is ONE ragged forward: the chunk's
                // prompt tokens plus one token per live row, whatever `N` —
                // not `N` forwards of a tile each. KV and conv stay per row.
                budget.max_seq_len += opts.batch;
                budget.activation_rows = Some(1);
            }
            config.memory_budget = Some(budget);
        }
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &device {
            // The ledger's quota: the same fraction of the working set the
            // admission and the KV grow read against (§9.5c).
            let ws = m.device().recommended_max_working_set_size() as f64;
            m.ledger()
                .set_quota((ws * opts.memory_fraction.max(0.0)) as usize);
        }

        let files = weight_files(&model_dir)?;
        // SAFETY: the checkpoint is not modified while mapped
        // (`VarBuilder::from_mmaped_safetensors`'s contract; this is that
        // backend with a flag per tensor set on load).
        let (backend, lm_record) = unsafe { RecordingBackend::over_safetensors(&files)? };
        let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        let vb = if vl {
            vb.rename_f(|name: &str| match name.strip_prefix("model.") {
                Some(rest) => format!("model.language_model.{rest}"),
                None => name.to_string(),
            })
        } else {
            vb
        };
        // §4.8 (#892 A6a): a model whose weights do not fit the quota streams
        // its decoder layers — each layer's matrices one group — through the
        // weight-group tier; one that fits never streams (R1). The quota is
        // the ledger's; the resident weights are everything but the layers
        // (the embedding, the norms, the conv taps); the slots are what the
        // admission leaves after them and the predicted peak, three at least.
        // What the weights hold once loaded: every tensor at the dtype the
        // load converts it to (f32 on the CPU, twice the bf16 file).
        let loaded_weights = checkpoint_elems(&weight_files(&model_dir)?)? * dtype.size_in_bytes();
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut machine_weights = loaded_weights;
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &device {
            let quota = m.ledger().quota();
            let elem = dtype.size_in_bytes();
            // One group a layer: the mixer's projections and the MLP's
            // (`lfm2::layer_group_parts`); a slot is sized for the largest.
            let group_bytes: Vec<usize> = (0..config.num_hidden_layers)
                .map(|l| lloom::models::lfm2::layer_group_bytes(&config, l, dtype))
                .collect();
            let slot_bytes = group_bytes.iter().copied().max().unwrap_or(0);
            let streamed_bytes: usize = group_bytes.iter().sum();
            let resident = weight_bytes.saturating_sub(streamed_bytes);
            let kv_floor = opts.kv_capacity.min(4096)
                * config
                    .layer_types
                    .iter()
                    .filter(|k| matches!(k, lloom::models::lfm2::LayerType::FullAttention))
                    .count()
                * 2
                * config.num_key_value_heads
                * config.head_dim()
                * elem;
            if quota > 0 && weight_bytes + kv_floor > quota {
                // The room for slots is what §9.5's admission leaves once the
                // resident rest and its predicted peak (the KV at capacity,
                // the conv state, the activations, the scratch) are in — the
                // same arithmetic the KV cache refuses on — less the RoPE
                // tables it adds (`Cache::new`). Three slots at least: the
                // group in use, the previous stage's (its reader still open)
                // and one loading.
                let rope = State::<P>::rope_table_bytes(&config);
                let mut peak = String::from(
                    "no memory budget declared: the quota less the resident rest and the KV floor",
                );
                let room = match config.memory_budget {
                    Some(mut b) => {
                        b.weight_bytes = resident;
                        let a = State::<P>::admission_budget(&config, &b, opts.kv_capacity)
                            .admit(m.device().recommended_max_working_set_size());
                        // The classes, for the refusal: which one took the room.
                        let f = &a.footprint;
                        peak = format!(
                            "predicted peak {:.2} GB = weights {:.2} + KV {:.2} + conv {:.2} + \
                             activations {:.2} (a {}-token forward) + scratch {:.2}; RoPE {:.2}",
                            f.predicted() as f64 / 1e9,
                            f.weights as f64 / 1e9,
                            f.kv as f64 / 1e9,
                            f.conv as f64 / 1e9,
                            f.activations as f64 / 1e9,
                            b.max_seq_len,
                            f.scratch as f64 / 1e9,
                            rope as f64 / 1e9
                        );
                        if a.fits {
                            a.residual.saturating_sub(rope)
                        } else {
                            0
                        }
                    }
                    None => quota.saturating_sub(resident + kv_floor),
                };
                let slots = (room / slot_bytes.max(1)).clamp(3, config.num_hidden_layers.max(3));
                if room < 3 * slot_bytes {
                    anyhow::bail!(
                        "the weights do not fit the quota even with the layers streamed: the \
                         resident rest {:.2} GB with its predicted peak leaves {:.2} GB, three \
                         layer slots need {:.2} GB; quota {:.2} GB ({} x recommendedMaxWorkingSetSize \
                         {:.2} GB)\n  {}",
                        resident as f64 / 1e9,
                        room as f64 / 1e9,
                        3.0 * slot_bytes as f64 / 1e9,
                        quota as f64 / 1e9,
                        opts.memory_fraction,
                        m.device().recommended_max_working_set_size() as f64 / 1e9,
                        peak
                    );
                }
                eprintln!(
                    "[weights: layers streamed groups={} slots={} slot_bytes={:.1} MiB streamed={:.2} GB resident={:.2} GB room={:.2} GB quota={:.2} GB]",
                    config.num_hidden_layers,
                    slots,
                    slot_bytes as f64 / (1024.0 * 1024.0),
                    streamed_bytes as f64 / 1e9,
                    resident as f64 / 1e9,
                    room as f64 / 1e9,
                    quota as f64 / 1e9
                );
                config.stream_layers = Some(lloom::models::lfm2::StreamLayers {
                    files: files.clone(),
                    slots,
                    depth: 1,
                });
                // §9.5's admission (the KV cache's `admit_memory_budget`)
                // predicts the weights resident: with the MLPs streamed the
                // weight term is the resident rest plus the slots.
                if let Some(b) = config.memory_budget.as_mut() {
                    b.weight_bytes = resident + slots * slot_bytes;
                }
                machine_weights =
                    loaded_weights.saturating_sub(streamed_bytes) + slots * slot_bytes;
            }
        }
        // The machine (`memory.md` §9.10): what this load holds before the
        // KV grows past its first bucket, admitted against the host and the
        // other lloom processes before a weight is read.
        let kv_floor = opts.memory.kv_bucket.min(opts.kv_capacity).max(1);
        // A prefill of several forwards keeps the pacer's depth of them in
        // flight, each holding its activations until its command buffer
        // completes; the prefill's own bound, else decode's, else the floor.
        let in_flight = [opts.decode.prefill.in_flight, opts.decode.in_flight]
            .into_iter()
            .find(|&d| d > 0)
            .unwrap_or(lloom::nn::decode::IN_FLIGHT_FLOOR);
        // How many pending rows' chunks one batched step carries: the budget
        // and the gauge above were taken at one; a wider step is charged here
        // for its activations and kept only where this load still fits.
        let prefill_batch = resolve_prefill_batch::<P>(
            &opts,
            &config,
            dcfg.is_some(),
            dtype,
            machine_weights,
            kv_floor,
            in_flight,
            &device,
        );
        if prefill_batch.k > 1 {
            if let Some(b) = config.memory_budget.as_mut() {
                b.max_seq_len = step_prompt_tokens(
                    opts.prefill_tile,
                    opts.prefill_chunk,
                    prefill_batch.k,
                    opts.kv_capacity,
                ) + opts.batch;
            }
        }
        let longest = match config.memory_budget {
            Some(b) => b.max_seq_len,
            None => step_prompt_tokens(
                opts.prefill_tile,
                opts.prefill_chunk,
                prefill_batch.k,
                opts.kv_capacity,
            ),
        };
        let mut peak = machine_peak::<P>(
            &config,
            dtype,
            machine_weights,
            opts.batch,
            kv_floor,
            longest,
            in_flight,
            config.memory_budget.and_then(|b| b.activation_rows),
        );
        if let Some(dir) = &dspark_dir {
            let drafter = std::fs::metadata(dir.join("model.safetensors")).map_or(0, |m| m.len());
            peak.push("drafter", drafter);
        }
        let machine = match admission {
            LoadAdmission::Legacy => {
                lloom::nn::memory::MachineLoad {
                    gate: opts.memory.machine_gate(),
                    label: super::common::machine_label("lfm2", &model_dir),
                    rows: opts.batch.max(1),
                    kv_tokens: kv_floor,
                    prefill_tile: opts.prefill_tile,
                }
                .admit(&device, peak)?
                .holding
            }
            // Loaded<Self> retains the caller's admitted domain. Attaching a
            // second legacy Holding here would double-admit the same model.
            LoadAdmission::CallerScoped => None,
        };
        let model = Model::new(&config, vb).context("constructing LFM2")?;
        load_warnings.splice(0..0, model.load_warnings().iter().cloned());
        if vl && opts.text_only {
            eprintln!("[lfm2: text only: the vision tower is not built]");
        }
        let vision = if vl && !opts.text_only {
            let vl_config: Lfm2VlConfig = serde_json::from_value(root.clone())
                .context("parsing config.json's vision half")?;
            Some(Vision {
                processor: ImageProcessor::from_checkpoint(&model_dir)
                    .map_err(|e| anyhow::anyhow!("the image preprocessing: {e}"))?,
                config: vl_config,
                files: files.clone(),
                dtype,
                text_hidden: config.hidden_size,
                purpose: opts.purpose,
                tower: Mutex::new(None),
            })
        } else {
            None
        };
        let private_cache = PrivateCachePolicy {
            capacity: opts.kv_capacity,
            bucket: opts.memory.kv_bucket,
            grow_fraction: opts.memory.kv_grow_fraction,
        };
        let make_cache = |device: &Device| private_cache.create::<P>(&config, dtype, device);
        let cache = make_cache(&device)?;

        let model_name = opts
            .model_name
            .unwrap_or_else(|| model_name_from_dir(&model_dir));

        let core = Arc::new(Core { model, device });
        let mut engine = Engine {
            core: Arc::clone(&core),
            text,
            model_name,
            kv_capacity: opts.kv_capacity,
            readback: opts.readback,
            axes_line,
            model: opts.model.clone(),
            generation: lloom::hub::read_generation_config(&model_dir).unwrap_or_default(),
            executor: opts.executor.clone(),
            reuse_prefix: opts.reuse_prefix,
            batch: opts.batch,
            dspark_k: dcfg.as_ref().map(|d| {
                opts.dspark_k
                    .unwrap_or(match opts.batch {
                        0 | 1 => d.max_draft(),
                        2 | 3 => 5,
                        _ => 3,
                    })
                    .clamp(1, d.max_draft())
            }),
            prefill_tile: opts.prefill_tile,
            prefill_chunk: opts.prefill_chunk,
            prefill_batch,
            prefill_decode: opts.decode.prefill.decode,
            memory_fraction: opts.memory_fraction,
            mem_sample: opts.mem_sample,
            rows: if opts.batch > 1 {
                opts.rows
            } else {
                RowsArm::Private
            },
            pool: None,
            share_prefix: opts.share_prefix && opts.batch > 1 && opts.rows.pooled(),
            lm_record,
            vision,
            last_vision: Mutex::new(Value::Null),
            config: config.clone(),
            dtype,
            dspark_setup: dspark_dir.clone().zip(dcfg.clone()),
            installed,
            warm_in_flight: None,
            load_warnings,
            machine,
            private_cache,
            decisions,
            decision_ragged: opts.decision_ragged,
            trunks: Default::default(),
            single: parking_lot::Mutex::new(Some(RequestState {
                sequence: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                extra: SingleExtra {
                    spec: None,
                    decode: opts.decode.clone(),
                    image_keys: Vec::new(),
                },
                needs_recovery: false,
            })),
            sched: None,
            sched_thread: Mutex::new(None),
        };

        if opts.warmup {
            let ((libs, pipes), measured) =
                engine.with_single(|s, _| Ok((s.warmup()?, s.measured_in_flight())))?;
            engine.warm_in_flight = Some(measured);
            eprintln!("[warmup: {libs} libraries, {pipes} pipelines compiled]");
        }
        // The two rings every request will want, built here rather than inside
        // the first request of each kind (#702: ~35 ms each, plus a sync).
        if !core.device.is_cpu() && opts.batch == 1 {
            let eos = engine.text.eos_ids.clone();
            let t = Instant::now();
            let built = engine.with_single(|s, _| {
                let device = s.device().clone();
                let a = s.rings().prepare(&eos, &device, ring_eos_read())?;
                let b = s.rings().prepare(&NEVER_EOS, &device, ring_eos_read())?;
                Ok(usize::from(a) + usize::from(b))
            })?;
            eprintln!(
                "[rings: {built} built at load in {:.1} ms (the EOS set, never-EOS); reused by every request]",
                ms(t.elapsed())
            );
        }

        // The drafter: one per serving shape (the single path's, or the
        // batch thread's with `N` slots), its pipelines warmed off the clock.
        let build_spec = |n_seq: usize| -> Result<DSparkSpeculator> {
            let (dir, dcfg) = (dspark_dir.as_ref().unwrap(), dcfg.as_ref().unwrap());
            let draft_file = dir.join("model.safetensors");
            if !draft_file.exists() {
                anyhow::bail!("{} does not exist", draft_file.display());
            }
            // The largest absolute position a draft attends: the KV ceiling
            // plus the block and slack.
            let max_pos = opts.kv_capacity + dcfg.block_size + 64;
            let dvb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, &core.device)?
            };
            let drafter = DSparkDrafter::new(dcfg.clone(), max_pos, dvb)
                .context("constructing DSpark drafter")?;
            opts.purpose.admit(
                "dspark drafter",
                &core.device,
                drafter.reference_attention(),
            )?;
            // The drafter's context KV: an explicit `tuning.dspark.ctx_kv`
            // (`--dspark-ctx`, the `LLOOM_DSPARK_CTX_KV` alias) wins on every
            // arm; `auto` is one paged pool set shared by every sequence on
            // the pool arms at N > 1 (the shape the batched draft's block
            // attention reads in ONE dispatch, §12.9), else paged per
            // sequence on Metal (§10.2z: the arm whose round allocates
            // nothing; a per-sequence pool is the pipelined driver's own
            // buffers, so it keeps its overlap), else contiguous.
            let ctx_kv =
                drafter_ctx_kv(CtxKv::from_tuning()?, core.device.is_metal(), pooled, n_seq);
            let mut spec = DSparkSpeculator::new_with_ctx(
                drafter,
                core.model.draft_embedding().clone(),
                core.model.lm_head_linear().clone(),
                n_seq,
                ctx_kv,
            )
            .context("constructing DSpark speculator")?;
            // The draft length by batch (lloom #148, record §8.11): the
            // verify window is `K + 1` positions per row and its cost per row
            // is the window, while the accepted prefix on natural text barely
            // moves with K (2.43 at 9, 2.20 at 3) — at B=1 the window rides
            // one weight sweep for free and the full block wins (156 / 126 /
            // 96 tok/s at 9 / 5 / 3); at B=2 K=5 (240 vs 217 / 170); from
            // B=4 K=3 (196 vs 142 at 4, 267 vs 216 at 8, 391 vs 259 at 16).
            // `--dspark-k` overrides.
            let k = opts.dspark_k.unwrap_or(match opts.batch {
                0 | 1 => dcfg.max_draft(),
                2 | 3 => 5,
                _ => 3,
            });
            spec.set_max_draft(k);
            let k = lloom::nn::Speculator::max_draft(&spec);
            eprintln!(
                "[dspark: drafter loaded from {}, {} layers, K={} (verify window {} positions{}), \
                 {n_seq} slots, context kv {}]",
                dir.display(),
                dcfg.num_hidden_layers,
                k,
                k + 1,
                if opts.dspark_k.is_some() {
                    format!("; --dspark-k, the drafter's block is {}", dcfg.max_draft())
                } else if k < dcfg.max_draft() {
                    format!(
                        "; the batch's default, the drafter's block is {}",
                        dcfg.max_draft()
                    )
                } else {
                    String::new()
                },
                spec.ctx_kv().name()
            );
            if opts.warmup {
                // The batched draft's pipelines too (`m = n_seq·γ`, the
                // ragged paged attention, the batched Markov chain).
                let before = pipeline_cache_sizes(&core.device);
                spec.warmup_rows(n_seq, &core.device)
                    .context("warming DSpark drafter pipelines")?;
                let after = pipeline_cache_sizes(&core.device);
                eprintln!(
                    "[dspark: drafter warmup +{} libs +{} pipelines]",
                    after.0.saturating_sub(before.0),
                    after.1.saturating_sub(before.1),
                );
            }
            Ok(spec)
        };

        if opts.batch == 1 {
            if dcfg.is_some() {
                let spec = build_spec(1)?;
                let mut lease = RequestLease::<Model<P>, _>::checkout(&engine.single, true)?;
                lease.extra_mut().spec = Some(spec);
            }
            return Ok(engine);
        }

        // `--batch N`: the pool and the scheduler thread. On the `pool` arms
        // every row is a slot of one `BatchPool` (§12.2); on `private` a
        // crown cache of its own.
        let mut caches = Vec::with_capacity(opts.batch);
        if pooled {
            let pool = BatchPool::new(&config, &core.device, opts.batch, opts.kv_capacity)
                .context("allocating the rows' shared pool")?;
            pool.set_batched_arms(opts.rows == RowsArm::Pool);
            for _ in 0..opts.batch {
                caches.push(
                    P::state(&config, dtype, &core.device, opts.kv_capacity, Some(&pool))
                        .context("leasing a row's slot")?,
                );
            }
            engine.pool = Some(pool);
        } else {
            for _ in 0..opts.batch {
                caches.push(make_cache(&core.device)?);
            }
        }
        let pool = SlotPool::new(opts.batch, || caches.pop().expect("one cache per slot"));
        // The one drafter: the continuous driver's on the pool arms (the
        // closed cells run through the driver), the static body's and the
        // closed cells' on the private arm (§12.9).
        let extra: Extra = Speculators {
            spec: if dcfg.is_some() {
                Some(build_spec(opts.batch)?)
            } else {
                None
            },
            continuous: dcfg.is_some() && pooled,
            eos_ids: engine.text.eos_ids.clone(),
        };
        let static_spec = dcfg.is_some() && !pooled;
        let batch_opts = if static_spec {
            static_dspark_scheduler_options(&opts.decode, opts.prefill_tile, opts.prefill_chunk)
        } else {
            DecodeOptions {
                lookahead: 1,
                prefill_tile: opts.prefill_tile,
                prefill_chunk: opts.prefill_chunk,
                ..DecodeOptions::new(0)
            }
        };
        // `--dspark` on private caches runs the static pipelined driver; on
        // the pool arms the continuous speculative driver (§12.4).
        let mode: BatchMode<Model<P>, Extra> = if static_spec {
            BatchMode::Static(Box::new(static_dspark_body))
        } else {
            BatchMode::Continuous
        };
        let cfg = BatchConfig {
            device: core.device.clone(),
            pool,
            rings: RingPool::new(),
            opts: batch_opts,
            max_rows: opts.batch,
            reuse_prefix: opts.reuse_prefix,
            share_prefix: opts.share_prefix,
            prefill_batch: engine.prefill_batch.k,
            prefill_decode: engine.prefill_decode,
            mode,
            extra,
            on_counters: None,
        };
        let (handle, thread) = spawn_batch(Arc::clone(&core), |c: &Core<P>| &c.model, cfg)
            .map_err(|e| anyhow::anyhow!("starting the batch scheduler: {e}"))?;
        eprintln!(
            "[batch: scheduler up, {} sequences, {} rows per step, tile {} chunk {}, prefill {}, {}]",
            opts.batch,
            opts.batch,
            opts.prefill_tile,
            opts.prefill_chunk,
            engine.prefill_batch.describe(engine.prefill_decode),
            match (dcfg.is_some(), pooled) {
                (true, true) => "continuous speculative batching (one drafter, batched drafts)",
                (true, false) => "static batches through the pipelined speculative driver",
                (false, _) => "continuous batching",
            }
        );
        engine.sched = Some(handle);
        *lock(&engine.sched_thread) = Some(thread);
        if opts.warmup {
            // The batched GEMM pipelines at `m = N` key their own `align_m`
            // variants (§13.1i), the prefill piece's at `m = tile` and, under
            // `--dspark`, the batched draft's and the ragged verify's: compile
            // them off every request's clock, with `N` rows — a one-row
            // warmup left the first batched cell paying them (§12.8: the
            // rotation-1 pool cells up to 15 % under rotation 2's).
            let n = opts.batch;
            let tile = opts.prefill_tile.max(8);
            let dspark = dcfg.is_some();
            engine
                .closed(move |idle| -> Result<()> {
                    with_sessions(idle, n, |idle, sessions| {
                        // One prompt a tile long beside the short ones: the
                        // piece GEMMs and the long row's attention compile too.
                        let short: Vec<u32> = vec![0u32; 8];
                        let long: Vec<u32> = vec![0u32; tile];
                        let prompts: Vec<&[u32]> = (0..n)
                            .map(|i| {
                                if i == 0 {
                                    long.as_slice()
                                } else {
                                    short.as_slice()
                                }
                            })
                            .collect();
                        let warm = DecodeOptions {
                            lookahead: 1,
                            prefill_tile: tile,
                            ..DecodeOptions::new(4)
                        };
                        if dspark {
                            let _ = run_spec_rows(
                                pooled,
                                idle,
                                sessions,
                                &prompts,
                                &warm,
                                &NEVER_EOS,
                                &mut |_, _| Ok(()),
                            )?;
                        } else {
                            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                            let _ = Session::decode_batched(
                                sessions,
                                &mut sampler,
                                &prompts,
                                &warm,
                                &NEVER_EOS,
                                idle.device,
                                &mut |_, _| Ok(()),
                            )?;
                        }
                        idle.device.synchronize()?;
                        Ok(())
                    })
                })
                .context("warming the batched pipelines")??;
            eprintln!(
                "[batch: warmed the m={} pipelines{} ({} rows, one prompt of {tile})]",
                opts.batch,
                if dspark {
                    " and the speculative round's"
                } else {
                    ""
                },
                n
            );
        }
        Ok(engine)
    }

    fn fence_failed_request(&self) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = &self.core.device {
            let marker = device.flush_marker()?;
            device.wait_marker_prefix_data(&marker)?;
            return Ok(());
        }
        self.core.device.synchronize()?;
        Ok(())
    }

    fn checkout_single(&self) -> Result<RequestLease<'_, Model<P>, SingleExtra>> {
        let lease = RequestLease::<Model<P>, _>::checkout(&self.single, true)?;
        // Retained target scratch also needs quiescence without speculation.
        // Refusal restores Held custody and the marker before any reset.
        lease.before_recovery(|| self.fence_failed_request())?;
        Ok(lease)
    }

    fn with_single<T>(
        &self,
        f: impl FnOnce(&mut DecodeSession<'_, Model<P>>, &mut SingleExtra) -> Result<T>,
    ) -> Result<T> {
        let mut lease = self.checkout_single()?;
        let decode = lease.extra().decode.clone();
        lease.start_decode(&self.core.model, &self.core.device, decode);
        let (session, extra) = lease.parts_mut();
        let out = f(session, extra);
        if out.is_ok() {
            lease.mark_complete();
        }
        out
    }

    /// Run a closed job on the batch thread and wait for it.
    pub fn closed<T: Send + 'static>(
        &self,
        job: impl for<'a, 'm> FnOnce(&mut Idle<'a, 'm, Model<P>, Extra>) -> T + Send + 'static,
    ) -> Result<T> {
        let sched = self
            .sched
            .as_ref()
            .context("no batch scheduler (start lloom-serve with --batch N)")?;
        sched
            .run_closed_blocking(job)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Stop the batch thread, if one runs, after the rows it holds retire.
    pub fn shutdown(&self) {
        if let Some(s) = &self.sched {
            s.shutdown();
        }
        if let Some(t) = lock(&self.sched_thread).take() {
            let _ = t.join();
        }
    }

    // ---- tokens ---------------------------------------------------------

    /// Encode `text`; with `add_special` the BOS the checkpoint's template
    /// starts with is prepended, which is what `/tokenize` must count for a
    /// `/v1/completions` prompt to report the same `prompt_tokens`.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let mut ids = self
            .text
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec();
        if add_special {
            if let Some(bos) = self.text.bos_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        Ok(ids)
    }

    /// The token string for one id, as the logprobs entries report it.
    pub fn token_str(&self, id: u32) -> String {
        self.text
            .tokenizer
            .decode(&[id], false)
            .unwrap_or_else(|_| format!("<{id}>"))
    }

    /// The seed tiled to `n` ids.
    pub fn seed_prompt(&self, n: usize) -> Vec<u32> {
        let seed = &self.text.seed_tokens;
        (0..n).map(|i| seed[i % seed.len()]).collect()
    }

    /// The seed tiled to `n` ids, starting `row` seed tokens in, so `B` rows
    /// of one batch carry different prompts (a shared prompt would share a
    /// prefix and skip the prefill work a cell exists to measure).
    pub fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32> {
        let seed = &self.text.seed_tokens;
        (0..n).map(|i| seed[(i + row * 7) % seed.len()]).collect()
    }

    /// The checkpoint's own chat template.
    pub fn template(&self) -> Result<&Template> {
        self.text
            .template
            .as_ref()
            .context("the checkpoint has no chat_template.jinja; /v1/chat/completions needs one")
    }

    /// Render `messages` through the checkpoint's own chat template with the
    /// generation prompt appended, `kwargs` laid over the template's
    /// variables and each of `checked` tested against it
    /// (`Template::render_checked`).
    pub fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        self.template()?
            .render_checked(messages, &self.text.bos_token, kwargs, checked)
    }

    fn detok(&self) -> impl FnMut(&[u32]) -> lloom_tensor::Result<String> + '_ {
        |ids: &[u32]| -> lloom_tensor::Result<String> {
            self.text
                .tokenizer
                .decode(ids, false)
                .map_err(|e| lloom_tensor::Error::Msg(format!("detokenize: {e}")))
        }
    }

    // ---- decode ---------------------------------------------------------

    /// One decode over `prompt_ids` from a fresh sequence (unless prefix reuse
    /// is on), streaming text fragments to `on_fragment`.
    pub fn complete(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) + Send,
    ) -> Result<Decoded> {
        self.complete_until(prompt_ids, params, |frag| {
            on_fragment(frag);
            true
        })
    }

    /// [`Self::complete`] whose hook can end the run early (`false`).
    pub fn complete_until(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        self.check_fits(prompt_ids.len(), params.max_tokens)?;
        if let Some(v) = &self.vision {
            let n = prompt_ids
                .iter()
                .filter(|&&t| t == v.config.image_token_id)
                .count();
            if n > 0 {
                anyhow::bail!(
                    "the prompt holds {n} image position(s) (`<image>`) and no image is attached"
                );
            }
        }
        *lock(&self.last_vision) = self.load_json(0);
        if let Some(sched) = &self.sched {
            return self.complete_batched(sched, prompt_ids, params, on_fragment);
        }
        let reuse = self.reuse_prefix;
        let params = &*params.under_reuse(reuse);
        let detok = self.detok();
        self.with_single(|s, extra| {
            // A text prompt leaves no image rows in the held cache.
            extra.image_keys.clear();
            if !reuse || params.cold {
                s.reset();
            }
            match &mut extra.spec {
                Some(spec) => s
                    .decode_speculative_until(
                        rewindable::<P>()?,
                        spec,
                        prompt_ids,
                        detok,
                        params,
                        on_fragment,
                    )
                    .map_err(anyhow::Error::from),
                None => s
                    .decode_until(prompt_ids, detok, params, on_fragment)
                    .map_err(anyhow::Error::from),
            }
        })
    }

    /// What this process has loaded, for the per-request report: the language
    /// model's tensors (266 on LFM2.5-VL-3B, whatever the request) and whether
    /// the vision tower is resident.
    fn load_json(&self, images: usize) -> Value {
        let (tensors, bytes) = self.lm_record.totals();
        json!({
            "family": "lfm2",
            "model_tensors": tensors,
            "model_bytes": bytes,
            "tower": match &self.vision {
                Some(v) => v.tower_json(),
                None => json!("none: the checkpoint has no vision_config"),
            },
            "images": images,
        })
    }

    /// The refusal for an image on a checkpoint without a tower.
    fn no_tower(&self) -> anyhow::Error {
        anyhow::anyhow!(
            "{} has no vision tower: its config.json names no vision_config (an LFM2-VL \
             checkpoint such as LiquidAI/LFM2.5-VL-3B carries one)",
            self.model_name
        )
    }

    /// The expansion of one image's `<image>`: its layout's delimiters and
    /// token runs (`Lfm2VlProcessor`).
    pub fn image_placeholder(&self, image: &DecodedImage) -> Result<String> {
        let v = self.vision.as_ref().ok_or_else(|| self.no_tower())?;
        let layout = v.processor.layout(image.height, image.width);
        Ok(v.processor
            .placeholder(&layout, v.config.use_image_special_tokens))
    }

    /// A request whose image positions take `images`' rows, in order.
    ///
    /// The tower loads on the first image (before the census windows open,
    /// so its weight uploads are not counted as encode dispatches). Then, on
    /// the one held sequence: keep the cached prefix when this request extends
    /// it with the same images (`Single::image_keys`), encode only the images
    /// whose rows are not in it, prefill the rest through [`ImageRows`], and
    /// decode from the logits the prefill held. Three recorder windows —
    /// encode, prefill, decode — make the dispatch census under
    /// `CANDLE_METAL_PROFILE=1` (`lfm2-vl --census`'s check, on this path).
    ///
    /// `--batch N` and `--dspark` are refused: their prefills feed token ids
    /// to the model directly, and neither has an image seam (#784 scope).
    pub fn complete_images_until(
        &self,
        prompt_ids: &[u32],
        images: &[DecodedImage],
        params: &DecodeParams,
        on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        let v = self.vision.as_ref().ok_or_else(|| self.no_tower())?;
        if images.is_empty() {
            return self.complete_until(prompt_ids, params, on_fragment);
        }
        if self.sched.is_some() {
            anyhow::bail!(
                "image input at --batch {} is not built: the scheduler's rows prefill token ids \
                 (serve images at --batch 1)",
                self.batch
            );
        }
        if self.dspark_k.is_some() {
            anyhow::bail!(
                "image input under --dspark is not built: the speculative prefill feeds token ids"
            );
        }
        self.check_fits(prompt_ids.len(), params.max_tokens)?;
        let image_token = v.config.image_token_id;
        let rows_of: Vec<usize> = images
            .iter()
            .map(|im| {
                v.processor
                    .image_tokens(&v.processor.layout(im.height, im.width))
            })
            .collect();
        let total: usize = rows_of.iter().sum();
        let reserved = prompt_ids.iter().filter(|&&t| t == image_token).count();
        if reserved != total {
            anyhow::bail!(
                "the prompt reserves {reserved} image position(s) and its {} image(s) take {total}",
                images.len()
            );
        }
        let keys: Vec<[u8; 32]> = images.iter().map(|im| im.key).collect();
        let device = self.core.device.clone();
        let mut lease = self.checkout_single()?;
        let decode = lease.extra().decode.clone();
        // Install stage custody and recover abandoned target/drafter state before
        // encoder work or comparing the kept token/pixel prefix.
        lease.with_model(&self.core.model, &device, decode.clone(), |s, _| {
            s.reconcile();
            Ok::<_, anyhow::Error>(())
        })?;
        let encoder = v.encoder(&device)?;

        let start = {
            let seq = lease.sequence();
            let cached = seq.tokens();
            let extends = self.reuse_prefix
                && seq.pos() == cached.len()
                && cached.len() < prompt_ids.len()
                && prompt_ids.starts_with(cached);
            // The images whose rows the kept prefix already holds must be
            // these images, in this order.
            let held_rows = cached.iter().filter(|&&t| t == image_token).count();
            let (mut k, mut rows) = (0, 0);
            while k < rows_of.len() && rows < held_rows {
                rows += rows_of[k];
                k += 1;
            }
            let same_images =
                lease.extra().image_keys.len() >= k && lease.extra().image_keys[..k] == keys[..k];
            if extends && same_images {
                cached.len()
            } else {
                0
            }
        };
        // Encode every image with a row at or past `start`.
        let rows_kept = prompt_ids[..start]
            .iter()
            .filter(|&&t| t == image_token)
            .count();
        let (mut first, mut first_rank) = (0, 0);
        while first < rows_of.len() && first_rank + rows_of[first] <= rows_kept {
            first_rank += rows_of[first];
            first += 1;
        }
        let prepared = images[first..]
            .iter()
            .map(|im| {
                v.processor
                    .prepare_rgb8(&im.rgb, im.height, im.width, &device, v.dtype)
                    .map_err(|e| {
                        anyhow::anyhow!("preparing a {}x{} image: {e}", im.height, im.width)
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let tiles: usize = prepared.iter().map(|p| p.tiles.len()).sum();
        // The encode window holds the tower and the projector only: the
        // patches' upload and cast are the preprocessing's, as in `lfm2-vl
        // --census`.
        take_window(&device)?;
        let mut parts = Vec::with_capacity(prepared.len());
        for p in &prepared {
            parts.push(encoder.encode_image(p).context("encoding the image")?);
        }
        let encode_w = take_window(&device)?;

        let detok = self.detok();
        let features = match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(Tensor::cat(&parts, 0)?),
        };
        let mut rows_placed = 0;
        let run = match features {
            Some(features) => {
                let lm = ImageRows::try_new(
                    &self.core.model,
                    prompt_ids.to_vec(),
                    image_token,
                    features,
                    first_rank,
                )?;
                let run = lease.with_model(&lm, &device, decode.clone(), |s, _| {
                    if start == 0 {
                        s.reset();
                    }
                    let pre = s.prefill(&prompt_ids[start..], params)?;
                    let prefill_w = take_window(&device)?;
                    let d = s.decode_until(prompt_ids, detok, params, on_fragment)?;
                    let decode_w = take_window(&device)?;
                    Ok::<_, anyhow::Error>((d, pre, prefill_w, decode_w))
                });
                rows_placed = lm.spliced();
                run
            }
            // Every image row is in the kept prefix: the rest is text.
            None => lease.with_model(&self.core.model, &device, decode, |s, _| {
                let pre = s.prefill(&prompt_ids[start..], params)?;
                let prefill_w = take_window(&device)?;
                let d = s.decode_until(prompt_ids, detok, params, on_fragment)?;
                let decode_w = take_window(&device)?;
                Ok((d, pre, prefill_w, decode_w))
            }),
        };
        let (mut decoded, pre, prefill_w, decode_w) = match run {
            Ok(r) => r,
            Err(e) => {
                lease.extra_mut().image_keys.clear();
                return Err(e);
            }
        };
        lease.extra_mut().image_keys = keys;
        if rows_placed != total - rows_kept {
            anyhow::bail!(
                "image prefill placed {rows_placed} image rows; the uncached prompt holds {}",
                total - rows_kept
            );
        }
        decoded.reuse = if start > 0 {
            ReuseAction::Extend
        } else {
            ReuseAction::Fresh
        };
        decoded.reused = start;
        decoded.prefilled = prompt_ids.len() - start;
        decoded.stats.prefill_ns = pre.prefill_ns;
        decoded.stats.prefill_units = pre.units;
        let census = match (encode_w, prefill_w, decode_w) {
            (Some(encode), Some(prefill), Some(decode)) => Some(ImageCensus {
                encode,
                prefill,
                decode,
                decode_steps: decoded.tokens.len(),
            }),
            _ => None,
        };
        let mut report = self.load_json(images.len());
        report["encoded_images"] = json!(images.len() - first);
        report["tiles"] = json!(tiles);
        report["image_rows"] = json!(total);
        report["image_rows_cached"] = json!(rows_kept);
        report["image_rows_placed"] = json!(rows_placed);
        report["census"] = census.map(|c| c.json()).unwrap_or(Value::Null);
        *lock(&self.last_vision) = report;
        lease.mark_complete();
        Ok(decoded)
    }

    /// A request through the batch scheduler: submitted, its tokens read off
    /// the channel and streamed as text, its outcome the row's.
    fn complete_batched(
        &self,
        sched: &BatchHandle<Model<P>, Extra>,
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
                            stream.refuse_last();
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
        if !cancelled && !done.cancelled && !stopped_on_string {
            let (rest, stopped) = stream.flush(&mut detok, &params.stop_strings)?;
            stopped_on_string |= stopped;
            if let Some(rest) = rest {
                if !on_fragment(&rest) {
                    stream.refuse_last();
                    cancelled = true;
                }
            }
        }
        let stop = if cancelled || (done.cancelled && !stopped_on_string) {
            Stop::Cancelled
        } else if stopped_on_string {
            Stop::StopString
        } else if done.eos {
            Stop::Eos
        } else {
            Stop::Length
        };
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
            readback: match self.dspark_k {
                Some(k) => ReadbackArm::Speculative { k },
                None => ReadbackArm::Batched {
                    rows: done.joined_beside,
                },
            },
            sampling: params.sampling()?,
            readback_note: None,
            stats: done.stats,
            thinking: None,
            logprobs: None,
        })
    }

    /// Time one prefill of `n` seed tokens on a fresh cache: the forward
    /// submitted and the device drained, nothing else in the bracket. At
    /// `batch > 1`, `batch` prompts of `n` tokens each arrive together and
    /// are fed as the scheduler feeds them — at most `--prefill-chunk`
    /// prompt tokens per step as ragged rows of one forward, each piece
    /// tile-bounded — and the bracket closes when the device drains; the
    /// aggregate is `batch × n` over that. A chunk above the tile fuses
    /// several rows' pieces into one forward whose working set is the
    /// chunk's, which is why the budget is sized at the chunk (`load`).
    pub fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        self.check_fits(n, 1)?;
        if batch <= 1 {
            let ids = self.seed_prompt(n);
            let device = self.core.device.clone();
            let tile = self.prefill_tile;
            let gauge = MemGauge::start(&device, self.mem_sample);
            let (prefill, forwards) = self.with_single(|s, _| {
                s.reset();
                // The prompt in tiles of at most `prefill_tile` tokens (all
                // of it in one forward at tile 0), each forward paced against
                // the GPU (`DecodeSession::prefill`), so the working set is
                // bounded by the tile rather than by the prompt.
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
                // The forwards wrote into the cache behind the session's token
                // log; nothing downstream may read that state.
                s.reset();
                Ok((elapsed, out.units))
            })?;
            return Ok(PrefillTiming {
                prompt_tokens: n,
                batch: 1,
                prefill_ms: ms(prefill),
                forwards,
                mem: gauge.finish(&device),
            });
        }
        self.check_batch(batch)?;
        let prompts: Vec<Vec<u32>> = (0..batch).map(|r| self.seed_prompt_row(n, r)).collect();
        let tile = self.prefill_tile;
        let chunk = self.prefill_chunk;
        let dspark = self.dspark_k.is_some();
        let pooled = self.rows.pooled();
        let gauge = MemGauge::start(&self.core.device, self.mem_sample);
        let (elapsed, forwards) = self.closed(move |idle| -> Result<(Duration, usize)> {
            with_sessions(idle, batch, |idle, sessions| {
                let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                let t = Instant::now();
                let forwards = if dspark {
                    // The speculative driver has no prefill-only entry; its
                    // prefill is the speculative loop's streamed one, one
                    // token of output per row.
                    let opts = DecodeOptions {
                        prefill_tile: tile,
                        ..DecodeOptions::new(1)
                    };
                    let outs = run_spec_rows(
                        pooled,
                        idle,
                        sessions,
                        &refs,
                        &opts,
                        &NEVER_EOS,
                        &mut |_, _| Ok(()),
                    )?;
                    outs.iter().map(|o| o.stats.prefill_units).sum()
                } else {
                    let opts = DecodeOptions {
                        lookahead: 1,
                        prefill_tile: tile,
                        prefill_chunk: chunk,
                        ..DecodeOptions::new(0)
                    };
                    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                    let outs = Session::decode_batched(
                        sessions,
                        &mut sampler,
                        &refs,
                        &opts,
                        &NEVER_EOS,
                        idle.device,
                        &mut |_, _| Ok(()),
                    )?;
                    outs.iter().map(|o| o.stats.prefill_units).sum()
                };
                idle.device.synchronize()?;
                let elapsed = t.elapsed();
                Ok((elapsed, forwards))
            })
        })??;
        Ok(PrefillTiming {
            prompt_tokens: n,
            batch,
            prefill_ms: ms(elapsed),
            forwards,
            mem: gauge.finish(&self.core.device),
        })
    }

    /// Time `decode_tokens` decode steps after a `prompt_tokens`-deep context
    /// (`llama-bench --n-depth P --n-gen D`'s shape): the prompt is prefilled
    /// and its first token sampled outside the bracket, the device drained,
    /// then `D` tokens — `D` forwards and `D` samples — are timed to the last
    /// one landing. EOS never fires. At `batch > 1` every row's context is
    /// built outside the bracket and the `D` batched steps are timed
    /// together: `decode_ms` is the batch's, `generation_tps` the aggregate.
    pub fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming> {
        super::common::check_context_capacity(self.kv_capacity, prompt_tokens, decode_tokens, 1)?;
        if batch <= 1 && self.sched.is_none() {
            return self.decode_throughput_single(prompt_tokens, decode_tokens);
        }
        self.check_batch(batch)?;
        let prompts: Vec<Vec<u32>> = (0..batch)
            .map(|r| self.seed_prompt_row(prompt_tokens, r))
            .collect();
        let tile = self.prefill_tile;
        let dspark_k = self.dspark_k;
        let pooled = self.rows.pooled();
        let gauge = MemGauge::start(&self.core.device, self.mem_sample);
        let mut timing = self.closed(move |idle| -> Result<DecodeTiming> {
            with_sessions(idle, batch, |idle, sessions| {
                let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                let out = if let Some(k) = dspark_k {
                    // The speculative body cannot split the context from the
                    // decode (its verify holds no resumable logits): one run of
                    // `D` tokens per row, the prompts' submit time subtracted;
                    // the prefills' GPU tail is inside the bracket.
                    let opts = DecodeOptions {
                        prefill_tile: tile,
                        ..DecodeOptions::new(decode_tokens)
                    };
                    profile_reset();
                    let t = Instant::now();
                    let outs = run_spec_rows(
                        pooled,
                        idle,
                        sessions,
                        &refs,
                        &opts,
                        &NEVER_EOS,
                        &mut |_, _| Ok(()),
                    )?;
                    idle.device.synchronize()?;
                    let wall = ms(t.elapsed());
                    let prof = profile_take();
                    let prefill_ms: f64 =
                        outs.iter().map(|o| o.stats.prefill_ns as f64 / 1e6).sum();
                    DecodeTiming {
                        prompt_tokens,
                        decode_tokens,
                        batch,
                        prefill_ms,
                        decode_ms: (wall - prefill_ms).max(0.0),
                        readback: ReadbackArm::Speculative { k },
                        bracket: if pooled {
                            "run wall minus the join prefills (each waited on: the rounds alone)"
                        } else {
                            "run wall minus the prompts' submit; the prefills' GPU tail is inside"
                        },
                        row_tokens: outs.iter().map(|o| o.ids.len()).collect(),
                        row_rounds: outs.iter().map(|o| o.stats.rounds.len()).collect(),
                        row_accepted: outs
                            .iter()
                            .map(|o| {
                                let n = o.stats.rounds.len().max(1) as f64;
                                o.stats
                                    .rounds
                                    .iter()
                                    .map(|r| r.accepted as f64)
                                    .sum::<f64>()
                                    / n
                            })
                            .collect(),
                        row_draft_ms: outs
                            .iter()
                            .map(|o| {
                                o.stats
                                    .rounds
                                    .iter()
                                    .map(|r| r.timing.draft_ns)
                                    .sum::<u64>() as f64
                                    / 1e6
                            })
                            .collect(),
                        row_verify_ms: outs
                            .iter()
                            .map(|o| {
                                o.stats
                                    .rounds
                                    .iter()
                                    .map(|r| r.timing.verify_ns)
                                    .sum::<u64>() as f64
                                    / 1e6
                            })
                            .collect(),
                        row_observe_ms: outs
                            .iter()
                            .map(|o| {
                                o.stats
                                    .rounds
                                    .iter()
                                    .map(|r| r.timing.observe_ns)
                                    .sum::<u64>() as f64
                                    / 1e6
                            })
                            .collect(),
                        profile: prof,
                        mem: MemReport::default(),
                    }
                } else {
                    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                    let ctx = DecodeOptions {
                        lookahead: 1,
                        prefill_tile: tile,
                        ..DecodeOptions::new(1)
                    };
                    let t = Instant::now();
                    let first = Session::decode_batched(
                        sessions,
                        &mut sampler,
                        &refs,
                        &ctx,
                        &NEVER_EOS,
                        idle.device,
                        &mut |_, _| Ok(()),
                    )?;
                    idle.device.synchronize()?;
                    let prefill_ms = ms(t.elapsed());
                    for (i, o) in first.iter().enumerate() {
                        if o.ids.len() != 1 {
                            anyhow::bail!(
                                "row {i}: the context run produced {} tokens, expected 1",
                                o.ids.len()
                            );
                        }
                    }
                    // The rows resume from the logits the context run held: an
                    // empty prompt per row, nothing re-fed.
                    let empty: Vec<&[u32]> = vec![&[]; batch];
                    let dec = DecodeOptions {
                        lookahead: 1,
                        ..DecodeOptions::new(decode_tokens)
                    };
                    profile_reset();
                    let t = Instant::now();
                    let outs = Session::decode_batched(
                        sessions,
                        &mut sampler,
                        &empty,
                        &dec,
                        &NEVER_EOS,
                        idle.device,
                        &mut |_, _| Ok(()),
                    )?;
                    idle.device.synchronize()?;
                    let decode_ms = ms(t.elapsed());
                    let prof = profile_take();
                    for (i, o) in outs.iter().enumerate() {
                        if o.ids.len() != decode_tokens {
                            anyhow::bail!(
                            "row {i}: the decode run produced {} tokens, expected {decode_tokens}",
                            o.ids.len()
                        );
                        }
                        if o.stats.prefill_units != 0 {
                            anyhow::bail!(
                                "row {i}: the decode run prefilled ({} forwards); the context was \
                             expected to resume",
                                o.stats.prefill_units
                            );
                        }
                    }
                    DecodeTiming {
                        prompt_tokens,
                        decode_tokens,
                        batch,
                        prefill_ms,
                        decode_ms,
                        readback: ReadbackArm::Batched { rows: batch },
                        bracket:
                            "D batched steps after every row's context, resumed from held logits",
                        row_tokens: outs.iter().map(|o| o.ids.len()).collect(),
                        row_rounds: Vec::new(),
                        row_accepted: Vec::new(),
                        row_draft_ms: Vec::new(),
                        row_verify_ms: Vec::new(),
                        row_observe_ms: Vec::new(),
                        profile: prof,
                        mem: MemReport::default(),
                    }
                };
                Ok(out)
            })
        })??;
        timing.mem = gauge.finish(&self.core.device);
        Ok(timing)
    }

    fn decode_throughput_single(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
    ) -> Result<DecodeTiming> {
        let prompt = self.seed_prompt(prompt_tokens);
        let device = self.core.device.clone();
        let readback = self.readback;
        let prefill_tile = self.prefill_tile;
        let detok = self.detok();
        let gauge = MemGauge::start(&device, self.mem_sample);
        let mut timing = self.with_single(|s, extra| {
            s.reset();
            if let Some(spec) = &mut extra.spec {
                // One run of `D` tokens; the prompt's submit time subtracted
                // (the prefill's GPU tail is inside the bracket).
                let params = throughput_params(decode_tokens, readback, prefill_tile);
                let t = Instant::now();
                let run = s.decode_speculative_until(
                    rewindable::<P>()?,
                    spec,
                    &prompt,
                    detok,
                    &params,
                    |_| true,
                )?;
                device.synchronize()?;
                let wall = ms(t.elapsed());
                let prefill_ms = run.stats.prefill_ns as f64 / 1e6;
                if run.tokens.len() != decode_tokens {
                    anyhow::bail!(
                        "the decode run produced {} tokens, expected {decode_tokens}",
                        run.tokens.len()
                    );
                }
                s.reset();
                return Ok(DecodeTiming {
                    prompt_tokens,
                    decode_tokens,
                    batch: 1,
                    prefill_ms,
                    decode_ms: (wall - prefill_ms).max(0.0),
                    readback: run.readback,
                    bracket: "run wall minus the prompt's submit; the prefill's GPU tail is inside",
                    row_tokens: vec![run.tokens.len()],
                    row_rounds: Vec::new(),
                    row_accepted: Vec::new(),
                    row_draft_ms: Vec::new(),
                    row_verify_ms: Vec::new(),
                    row_observe_ms: Vec::new(),
                    profile: None,
                    mem: MemReport::default(),
                });
            }
            let mut detok = detok;
            let params = throughput_params(1, readback, prefill_tile);
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
            // The whole sequence so far; `Session::prepare` resumes from the
            // logits the context run held rather than re-feeding anything.
            let mut all: Vec<u32> = prompt.clone();
            all.extend_from_slice(&first.tokens);
            let params = throughput_params(decode_tokens, readback, prefill_tile);
            let t = Instant::now();
            let run = s.decode(&all, &mut detok, &params, |_| {})?;
            let decode_ms = ms(t.elapsed());
            if run.tokens.len() != decode_tokens {
                anyhow::bail!(
                    "the decode run produced {} tokens, expected {decode_tokens}",
                    run.tokens.len()
                );
            }
            if run.prefilled != 0 {
                anyhow::bail!(
                    "the decode run prefilled {} tokens; the context was expected to resume \
                     ({:?})",
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
                profile: None,
                mem: MemReport::default(),
            })
        })?;
        timing.mem = gauge.finish(&device);
        Ok(timing)
    }

    /// The next-token log-probabilities after `prompt_ids`: the `k` most
    /// likely ids with their log-probs, and the argmax. One forward on a fresh
    /// cache; the logits come to the host once.
    pub fn next_token_logprobs(
        &self,
        prompt_ids: &[u32],
        k: usize,
    ) -> Result<(u32, Vec<(u32, f32)>)> {
        self.check_fits(prompt_ids.len(), 1)?;
        let device = self.core.device.clone();
        let ids = prompt_ids.to_vec();
        let logits = if self.sched.is_some() {
            self.closed(move |idle| -> Result<Tensor> {
                with_sessions(idle, 1, |idle, sessions| {
                    let s = &mut sessions[0];
                    let input = Tensor::new(ids.as_slice(), idle.device)?.unsqueeze(0)?;
                    let logits = idle.model.forward(&input, 0, s.cache_mut())?;
                    let v = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    Tensor::from_vec(v, logits.dim(1)?, &Device::Cpu).map_err(anyhow::Error::from)
                })
            })??
        } else {
            self.with_single(|s, _| {
                s.reset();
                let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
                let logits = s.model().forward(&input, 0, s.cache_mut())?;
                s.reset();
                Ok(logits.squeeze(0)?.to_dtype(DType::F32)?)
            })?
        };
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

    /// The verbose report's family lines: the pacer's depth against what
    /// warmup measured, and the held sequence's KV slab with the process's
    /// device allocation — the memory a run held at the length it reached,
    /// not the ceiling it could have (§9.5's admission is a prediction; this
    /// is the reading).
    pub fn diagnostics(&self) -> Vec<String> {
        let mut out = Vec::new();
        let slot = self.single.lock();
        if let Some(single) = slot.as_ref() {
            let flag = single.extra.decode.in_flight;
            if let Some((measured, (host_ns, gpu_ns))) = self.warm_in_flight {
                out.push(format!(
                "[in-flight: depth {} (warmup measured {:.2} ms host encode / {:.2} ms gpu step -> \
                 {measured}){}]",
                if flag > 0 { flag } else { lloom::nn::decode::IN_FLIGHT_FLOOR },
                host_ns as f64 / 1e6,
                gpu_ns as f64 / 1e6,
                if flag > 0 {
                    ", overridden"
                } else {
                    "; the pacer takes decode.in_flight, whose 0 is the floor, not the measured depth"
                },
            ));
            }
            if let Some(seq) = single.sequence.as_ref() {
                let kv = seq.cache().kv_report();
                let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
                #[cfg(feature = "metal")]
                let device_line = match &self.core.device {
                    Device::Metal(m) => format!(
                        " device_allocated={:.2} GB working_set={:.2} GB",
                        m.device().current_allocated_size() as f64 / 1e9,
                        m.device().recommended_max_working_set_size() as f64 / 1e9
                    ),
                    _ => String::new(),
                };
                #[cfg(not(feature = "metal"))]
                let device_line = String::new();
                out.push(format!(
                "[kv: len={} slab={} ceiling={} bucket={} grows={} kv_bytes={:.1} MiB{device_line}]",
                kv.kv_len,
                kv.slab,
                kv.capacity,
                kv.bucket,
                kv.grows,
                mib(kv.bytes)
            ));
            }
        } else {
            out.push("[kv: request owner busy; cache report unavailable]".to_string());
        }
        drop(slot);
        // §4.8: the weight-group tier's shape and counters, when the layers
        // stream (nothing at full quota — R1, the tier is not there).
        if let Some((slots, groups, slot_bytes, g)) = self.core.model.streamed_layers_stats() {
            out.push(format!(
                "[groups: slots={slots} groups={groups} slot_bytes={:.1} MiB hits={} misses={} \
                 evictions={} loaded={:.2} GB loader_waits={} device_waits={}]",
                slot_bytes as f64 / (1024.0 * 1024.0),
                g.hits,
                g.misses,
                g.evictions,
                g.bytes_loaded as f64 / 1e9,
                g.waits,
                g.device_waits
            ));
        }
        if let Some(line) = lloom::nn::memory::ledger_line(&self.core.device) {
            out.push(line);
        }
        out
    }

    /// `lloom bench --mode reuse | concurrent` on this engine's model: the
    /// modes build their own sessions, caches and — under `--rows pool*` —
    /// the rows' pool, so the engine's own sequence is untouched.
    pub fn run_mode(&self, spec: &super::modes::ModeSpec) -> Result<()> {
        if self.sched.is_some() {
            anyhow::bail!(
                "`lloom bench --mode {}` drives its own sequences; load it at --batch 1 (the \
                 scheduler would hold the pool the mode builds)",
                spec.mode.name()
            );
        }
        let pooled = spec.rows.pooled();
        if pooled && self.config.kv_append != KvAppend::Paged {
            anyhow::bail!(
                "--rows {} makes every sequence a slot of one paged pool; the model loaded on \
                 {:?} (load it on --cache pool, pool-spec under --dspark)",
                spec.rows.name(),
                self.config.kv_append
            );
        }
        let mut lease = self.checkout_single()?;
        lease.start_external_work();
        let out = self.run_mode_body(spec, pooled);
        if out.is_ok() {
            lease.mark_complete();
        }
        out
    }

    // Only reached while the same model's request lease is held. The modes
    // retain all incumbent fresh/reference/batched/speculative cache paths.
    fn run_mode_body(&self, spec: &super::modes::ModeSpec, pooled: bool) -> Result<()> {
        let pool = if pooled {
            let p = BatchPool::new(
                &self.config,
                &self.core.device,
                spec.sequences.max(1) + 3,
                self.kv_capacity,
            )
            .context("allocating the rows' shared pool")?;
            p.set_batched_arms(spec.rows == RowsArm::Pool);
            Some(p)
        } else {
            None
        };
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(self
                .text
                .tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
                .get_ids()
                .to_vec())
        };
        super::modes::run(super::modes::Ctx {
            model: &self.core.model,
            config: &self.config,
            device: &self.core.device,
            dtype: self.dtype,
            eos_ids: &self.text.eos_ids,
            spec,
            pool: pool.as_ref(),
            dspark_setup: self.dspark_setup.clone(),
            encode: &encode,
        })
    }

    fn check_fits(&self, prompt: usize, gen: usize) -> Result<()> {
        super::common::check_context_capacity(self.kv_capacity, prompt, gen, 0)
    }

    /// A cell's `batch` must fit the pool the server was started with.
    pub fn check_batch(&self, batch: usize) -> Result<()> {
        if batch == 0 {
            anyhow::bail!("batch must be positive");
        }
        if batch > self.batch {
            anyhow::bail!(
                "batch {batch} exceeds the server's --batch {} (start lloom-serve with --batch >= {batch})",
                self.batch
            );
        }
        Ok(())
    }
}

/// `n` sequences checked out of the pool for a closed job, reset, and the
/// slots to return them to.
struct TakenSessions<'m, P: Profile> {
    sessions: Vec<Session<'m, Model<P>>>,
    slots: Vec<usize>,
}

impl<'m, P: Profile> TakenSessions<'m, P> {
    fn give_back(self, idle: &mut Idle<'_, 'm, Model<P>, Extra>) {
        for (slot, mut s) in self.slots.into_iter().zip(self.sessions) {
            s.reset();
            idle.pool.release(slot, s.into_sequence());
        }
    }
}

/// Run `f` over `n` sequences checked out of the pool and reset, and put
/// them back — reset again — whatever `f` returned: a failed cell must not
/// cost the pool a slot. `f` may move the sessions out and back (a run
/// through the scheduler's driver does), as long as it returns them all.
fn with_sessions<'m, P: Profile, T>(
    idle: &mut Idle<'_, 'm, Model<P>, Extra>,
    n: usize,
    f: impl FnOnce(&mut Idle<'_, 'm, Model<P>, Extra>, &mut Vec<Session<'m, Model<P>>>) -> Result<T>,
) -> Result<T> {
    let mut taken = take_sessions(idle, n)?;
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

fn take_sessions<'m, P: Profile>(
    idle: &mut Idle<'_, 'm, Model<P>, Extra>,
    n: usize,
) -> Result<TakenSessions<'m, P>> {
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
    Ok(TakenSessions { sessions, slots })
}

/// A closed speculative run over `sessions`: THROUGH the scheduler's
/// continuous driver on the pool arms (the speculative batched loop and its
/// one drafter, `drive_closed_rows`), the static pipelined driver over the
/// batch thread's drafter on private caches. Greedy, one EOS set.
fn run_spec_rows<'m, P: Profile>(
    pooled: bool,
    idle: &mut Idle<'_, 'm, Model<P>, Extra>,
    sessions: &mut Vec<Session<'m, Model<P>>>,
    prompts: &[&[u32]],
    opts: &DecodeOptions,
    eos_ids: &[u32],
    on_tokens: &mut dyn FnMut(usize, &[u32]) -> lloom_tensor::Result<()>,
) -> Result<Vec<lloom::nn::DecodeOut>> {
    if pooled {
        let owned = std::mem::take(sessions);
        let (back, outs) = crate::drive_closed_rows(
            idle.driver,
            owned,
            prompts,
            eos_ids,
            opts.max_tokens,
            on_tokens,
        );
        *sessions = back;
        return Ok(outs?);
    }
    let spec = idle
        .extra
        .spec
        .as_mut()
        .context("no drafter on the batch thread")?;
    Ok(Session::decode_speculative_pipelined(
        sessions,
        rewindable::<P>()?,
        spec,
        prompts,
        opts,
        eos_ids,
        idle.device,
        on_tokens,
    )?)
}

/// Resolve the static scheduler's seed options at load, retaining the legacy
/// tile/chunk overrides and the batch lookahead. Continuous rows have their own
/// option contract; this does not add schedule support to that driver.
fn static_dspark_scheduler_options(
    decode: &crate::DecodeConfig,
    prefill_tile: usize,
    prefill_chunk: usize,
) -> DecodeOptions {
    let mut opts = decode.options(0, ReadbackArm::Blocking, prefill_tile);
    opts.lookahead = 1;
    if prefill_chunk > 0 {
        opts.prefill_chunk = prefill_chunk;
    }
    opts
}

/// The static body under `--dspark --batch N --rows private`: every row's
/// round pipelined through the one drafter
/// (`Session::decode_speculative_pipelined`, §10.2y). Greedy, one EOS set
/// per batch — mixed sets are refused rather than served under the first
/// row's.

fn static_dspark_body<'a, 'm, P: Profile>(
    idle: &mut Idle<'a, 'm, Model<P>, Extra>,
    sessions: &mut [Session<'m, Model<P>>],
    prompts: &[&[u32]],
    specs: &[RowSpec],
    on_tokens: &mut dyn FnMut(usize, &[u32]) -> lloom_tensor::Result<()>,
) -> lloom_tensor::Result<Vec<DecodeOut>> {
    let spec = idle
        .extra
        .spec
        .as_mut()
        .ok_or_else(|| lloom_tensor::Error::msg("no drafter on the batch thread"))?;
    let first = specs
        .first()
        .ok_or_else(|| lloom_tensor::Error::msg("an empty static batch"))?;
    for (i, s) in specs.iter().enumerate() {
        lloom::nn::refuse(
            lloom::nn::Body::BatchedWindow,
            &lloom::nn::Asks {
                sampling: &s.sampling,
                processors: false,
                budget: false,
                digest: false,
                logprobs: None,
            },
        )
        .map_err(|e| lloom_tensor::Error::Msg(format!("row {i}: {e}")))?;
        if s.eos_ids != first.eos_ids {
            lloom_tensor::bail!(
                "row {i}: a static speculative batch runs one EOS set; this row's differs from \
                 row 0's"
            );
        }
    }
    let caps: Vec<usize> = specs.iter().map(|s| s.max_tokens).collect();
    Session::decode_speculative_pipelined_with_caps(
        sessions,
        rewindable::<P>()?,
        spec,
        prompts,
        idle.opts,
        &caps,
        &first.eos_ids,
        idle.device,
        on_tokens,
    )
}

/// The shapes the quantized KV arms have no path on, refused by the flags that
/// select them, from the config the preset and the overrides resolved to: the
/// paged pool (the model's `Cache` has no quantized arm for it) and a device
/// other than Metal (`quantize_kv_into` and `sdpa_quant` are Metal
/// kernels).
fn refuse_kv_quant_shape(config: &Config, device: &Device, batch: usize) -> Result<()> {
    if !config.kv_quant.is_on() {
        return Ok(());
    }
    let arm = config.kv_quant.name().to_ascii_lowercase();
    if config.kv_append == KvAppend::Paged {
        anyhow::bail!(
            "kv-quant = {arm} has no arm on the paged KV pool, which this shape selects ({}): \
             use `--rows private` at `--batch N`, or the crown",
            if batch > 1 {
                "`--batch N` serves the pool rows by default"
            } else {
                "`--cache pool|pool-spec` or `--kv-append paged`"
            }
        );
    }
    if !device.is_metal() {
        anyhow::bail!(
            "kv-quant = {arm} is Metal-only (the quantized append and read are Metal kernels): \
             drop `--cpu`, or build with the `metal` feature"
        );
    }
    Ok(())
}

/// The load warning of a speculative body at one sequence on the paged KV:
/// every verify window is a multi-row pass, which on the pool attends over a
/// copy of the cache gathered out of it, a per-round cost that grows with
/// the context and that the in-place KV does not pay (`speculation.md`
/// §10.2z-x). `None` on every other shape.
fn paged_verify_warning(config: &Config, speculative: bool, batch: usize) -> Option<LoadWarning> {
    (speculative && batch == 1 && config.kv_append == KvAppend::Paged).then(|| {
        LoadWarning::new(
            LoadWarningKind::Speculation,
            "at --batch 1 the speculative body runs on the paged KV (`context.kv.layout = paged`), \
             whose verify window attends over a copy of the cache gathered out of the pool every \
             round, ~3 ms a round per 1000 tokens of context; the in-place KV \
             (`--kv-append in-place`, chat's and serve's default here) reads it where it lies. \
             Serving it as asked."
                .to_string(),
        )
    })
}

/// Whether the conv's left-context carry must be on: a prefill fed in more
/// than one forward — a tile, a chunk, the spark's ramp — or onto a cache a
/// held sequence extends zero-pads each piece's left edge without it, and
/// computes something one pass does not (§13.2f defect 1). One forward onto
/// a cold cache is identical either way.
/// This resolves load-known options, not later request/config mutations.
fn conv_carry_needed(
    config: &Config,
    prefill_tile: usize,
    prefill_chunk: usize,
    decode: &crate::DecodeConfig,
    reuse_prefix: bool,
) -> bool {
    config.conv_prefill_ctx
        || prefill_tile > 0
        || prefill_chunk > 0
        || decode.prefill.tile > 0
        || decode.prefill.chunk > 0
        || decode.prefill.ubatch > 0
        || decode.prefill.spark.is_some()
        || reuse_prefix
}

/// The B=1 throughput requests share one constructor on both loop bodies.
fn throughput_params(
    max_tokens: usize,
    readback: ReadbackArm,
    prefill_tile: usize,
) -> DecodeParams {
    DecodeParams {
        max_tokens,
        eos_ids: NEVER_EOS.to_vec(),
        readback,
        prefill_tile,
        ..DecodeParams::default()
    }
}

/// The drafter's context KV: the arm `tuning.dspark.ctx_kv` names when it
/// names one, else one paged pool shared by every sequence on the pool arms
/// at `N > 1` (the shape the batched draft reads in one dispatch, §12.9),
/// else paged per sequence on Metal — the round allocates nothing and the
/// pipelined driver keeps its overlap, since the sequences' buffers are
/// disjoint (§10.2z) — else contiguous, the CPU's.
fn drafter_ctx_kv(tuned: Option<CtxKv>, metal: bool, pooled: bool, n_seq: usize) -> CtxKv {
    match tuned {
        Some(arm) => arm,
        None if pooled && n_seq > 1 => CtxKv::Paged {
            page_size: CtxKv::DEFAULT_PAGE_SIZE,
            shared: true,
        },
        None if metal => CtxKv::Paged {
            page_size: CtxKv::DEFAULT_PAGE_SIZE,
            shared: false,
        },
        None => CtxKv::Contiguous,
    }
}

/// The Metal kernel cache's `(libraries, pipelines)` sizes, `(0, 0)` off
/// Metal: what a warmup compiled, by difference.
fn pipeline_cache_sizes(device: &Device) -> (usize, usize) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.kernels().cache_sizes();
    }
    let _ = device;
    (0, 0)
}

/// The prompt tokens one batched step's forward carries at most: the tile,
/// or `k` rows' chunks when that is wider; the whole KV ceiling without a
/// tile. At `k = 1` the chunk, which is what a step fed before
/// `scheduler.prefill_batch`.
fn step_prompt_tokens(
    prefill_tile: usize,
    prefill_chunk: usize,
    k: usize,
    kv_capacity: usize,
) -> usize {
    if prefill_tile > 0 {
        prefill_tile.max(prefill_chunk.saturating_mul(k.max(1)))
    } else {
        kv_capacity
    }
}

/// `scheduler.prefill_batch` as a load resolved it: the pending rows' chunks
/// one batched step carries (`k`), what was asked (`0` = up to the batch
/// width), and what bounded `k` below the ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillBatch {
    pub k: usize,
    pub asked: usize,
    pub bound: Option<&'static str>,
}

impl PrefillBatch {
    fn one() -> Self {
        Self {
            k: 1,
            asked: 1,
            bound: None,
        }
    }

    /// `2 chunks a step (asked 4; the room admits 2), ride` for the load line.
    pub fn describe(&self, decode: lloom::nn::PrefillDecode) -> String {
        let asked = match self.asked {
            0 => "up to the batch".to_string(),
            n => n.to_string(),
        };
        match self.bound {
            None if self.asked == self.k => format!("{} chunk(s) a step, {decode}", self.k),
            None => format!("{} chunk(s) a step (asked {asked}), {decode}", self.k),
            Some(why) => format!(
                "{} chunk(s) a step (asked {asked}; {why}), {decode}",
                self.k
            ),
        }
    }
}

/// How many pending rows' chunks one batched step carries
/// (`scheduler.prefill_batch`, `batched-decode-design.md` §11.8): the ask
/// (`0` = the batch width), at most the batch width, and at most what this
/// load's admission still admits — the most `k` whose step (`k` chunks plus
/// one token per live row) fits both the process budget at this load's
/// weights and the machine's room for the peak, `in_flight` such steps'
/// activations at once. 1 where the K=1 batched loop does not run: at one
/// sequence, without a chunk budget, under `--dspark` (its driver prefills
/// each row alone).
#[allow(clippy::too_many_arguments, unused_variables)]
fn resolve_prefill_batch<P: Lfm2Shape>(
    opts: &LoadOptions,
    config: &Config,
    dspark: bool,
    dtype: DType,
    machine_weights: usize,
    kv_floor: usize,
    in_flight: usize,
    device: &Device,
) -> PrefillBatch {
    let asked = opts.decode.prefill.batch;
    if opts.batch <= 1 || opts.prefill_chunk == 0 {
        return PrefillBatch::one();
    }
    if dspark {
        return PrefillBatch {
            k: 1,
            asked,
            bound: (asked != 1).then_some("the speculative driver prefills each row alone"),
        };
    }
    let cap = if asked == 0 {
        opts.batch
    } else {
        asked.min(opts.batch)
    };
    let mut bound = (asked > opts.batch).then_some("at most the batch width");
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        let gate = lloom::nn::memory::MachineLoad {
            gate: opts.memory.machine_gate(),
            label: String::new(),
            rows: opts.batch,
            kv_tokens: kv_floor,
            prefill_tile: opts.prefill_tile,
        }
        .gate_for(device);
        let room = gate.room();
        let pools = lloom::nn::memory::pool_bound_bytes(device) as i128;
        let ws = m.device().recommended_max_working_set_size();
        let rope = State::<P>::rope_table_bytes(config);
        let fits = |k: usize| -> bool {
            let longest =
                step_prompt_tokens(opts.prefill_tile, opts.prefill_chunk, k, opts.kv_capacity)
                    + opts.batch;
            let budget_ok = config.memory_budget.is_none_or(|mut b| {
                b.max_seq_len = longest;
                let a = State::<P>::admission_budget(config, &b, opts.kv_capacity).admit(ws);
                a.fits && a.residual >= rope
            });
            let machine_ok = room.is_none_or(|room| {
                let peak = machine_peak::<P>(
                    config,
                    dtype,
                    machine_weights,
                    opts.batch,
                    kv_floor,
                    longest,
                    in_flight,
                    config.memory_budget.and_then(|b| b.activation_rows),
                );
                peak.total() as i128 + pools <= room
            });
            budget_ok && machine_ok
        };
        let mut k = cap;
        while k > 1 && !fits(k) {
            k -= 1;
        }
        if k < cap {
            bound = Some("the room admits no wider step");
        }
        return PrefillBatch { k, asked, bound };
    }
    PrefillBatch {
        k: cap,
        asked,
        bound,
    }
}

/// What an LFM2 load holds at its peak before its KV grows past
/// `kv_floor` positions (`memory.md` §9.10): the weights it will hold, each
/// row's state floor (`lfm2::state_floor_bytes`), and — on Metal — the
/// activation class of its longest forward once per forward the pacer keeps
/// in flight (`in_flight`), the scratch class and the RoPE tables, from the
/// arithmetic the KV cache is admitted on (§9.5d).
#[allow(unused_variables)]
fn machine_peak<P: Profile>(
    config: &Config,
    dtype: DType,
    weights: usize,
    batch: usize,
    kv_floor: usize,
    longest_forward: usize,
    in_flight: usize,
    activation_rows: Option<usize>,
) -> lloom_tensor::machine::Peak {
    let rows = batch.max(1);
    let mut peak = lloom_tensor::machine::Peak::new().with("weights", weights as u64);
    for (name, bytes) in lloom::models::lfm2::state_floor_bytes(config, dtype, kv_floor) {
        let name = if rows > 1 {
            format!("{name} x {rows} rows")
        } else {
            name.to_string()
        };
        peak.push(name, (bytes * rows) as u64);
    }
    #[cfg(feature = "metal")]
    {
        let mut b = MemoryBudget::new(weights);
        b.batch = rows;
        b.max_seq_len = longest_forward;
        b.activation_rows = activation_rows;
        let f = State::<P>::admission_budget(config, &b, kv_floor).footprint();
        let in_flight = in_flight.max(1);
        peak.push(
            format!("activations ({longest_forward}-position forward x {in_flight} in flight)"),
            (f.activations * in_flight) as u64,
        );
        peak.push("scratch", f.scratch as u64);
        peak.push("RoPE tables", State::<P>::rope_table_bytes(config) as u64);
    }
    peak
}

/// Elements of every tensor in `files`, from their safetensors headers
/// (the header alone is read).
fn checkpoint_elems(files: &[PathBuf]) -> Result<usize> {
    use std::io::Read;
    let mut n = 0usize;
    for f in files {
        let mut file =
            std::fs::File::open(f).with_context(|| format!("opening {}", f.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut header)?;
        let v: Value = serde_json::from_slice(&header)
            .with_context(|| format!("the safetensors header of {}", f.display()))?;
        for (name, t) in v.as_object().context("a safetensors header is an object")? {
            if name == "__metadata__" {
                continue;
            }
            n += t["shape"]
                .as_array()
                .map(|s| {
                    s.iter()
                        .map(|d| d.as_u64().unwrap_or(0) as usize)
                        .product::<usize>()
                })
                .unwrap_or(0);
        }
    }
    Ok(n)
}

/// The safetensors shards for a checkpoint: the sharded index if present,
/// else the single `model.safetensors`.
pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let json: Value =
        serde_json::from_slice(&std::fs::read(&index)?).context("reading safetensors index")?;
    let weight_map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("safetensors index has no weight_map")?;
    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

/// The LFM2.5-2.6B snapshot in the HF cache, if it is there
/// (`LFM2_MODEL_DIR` first).
pub fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    newest_snapshot("models--LiquidAI--LFM2.5-2.6B")
}

impl<P: Lfm2Shape> super::served::Served for Engine<P> {
    fn device(&self) -> &Device {
        &self.core.device
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
        self.text.bos_id
    }
    fn health(&self) -> Value {
        json!({
            "family": "lfm2",
            "batch": self.batch,
            "prefill_tile": self.prefill_tile,
            "prefill_chunk": self.prefill_chunk,
            "prefill_batch": self.prefill_batch.k,
            "prefill_decode": self.prefill_decode.name(),
            "memory_fraction": self.memory_fraction,
            "kv_capacity": self.kv_capacity,
            "rows": self.rows.name(),
            "share_prefix": self.share_prefix,
            "dspark_k": self.dspark_k,
            "decision_ragged": self.decision_ragged.then(|| self.trunks.report()),
            "pool": self.pool.as_ref().map(|p| {
                let r = p.report();
                json!({
                    "slots": r.slots,
                    "slots_free": r.slots_free,
                    "pages_in_use": r.pages_in_use,
                    "pages_free": r.pages_free,
                    "kv_bytes": r.kv_bytes,
                    "conv_bytes": r.conv_bytes,
                    "batched_arms": p.batched_arms(),
                    "shared_calls": p.shared_calls(),
                })
            }),
        })
    }
    fn describe(&self) -> String {
        format!(
            "family=lfm2 batch={} dspark={} prefill_tile={} prefill_chunk={} prefill_batch={} prefill_decode={} memory_fraction={} mem_sample={} rows={} cache={} {} {}",
            self.batch,
            self.dspark_k
                .map(|k| format!("k={k}"))
                .unwrap_or_else(|| "off".to_string()),
            self.prefill_tile,
            self.prefill_chunk,
            self.prefill_batch.k,
            self.prefill_decode,
            self.memory_fraction,
            self.mem_sample,
            self.rows.name(),
            self.model.cache,
            self.axes_line,
            self.executor.line(),
        )
    }
    fn generation_config(&self) -> Option<lloom::hub::GenerationConfig> {
        self.generation.clone()
    }
    fn capabilities(&self) -> super::family::Capabilities {
        super::family::Capabilities {
            decisions: self.decisions.is_some(),
            vision: self.vision.is_some(),
            vision_missing: "no vision tower is built: the checkpoint names no vision_config, \
                or it was loaded text only (model.text_only, --text-only)",
            ..super::family::LFM2_CAPABILITIES
        }
    }
    fn systemone(
        &self,
        body: &crate::decision::OrderedJson,
    ) -> std::result::Result<crate::decision::SystemOneResponse, crate::decision::SystemOneError>
    {
        let Some(decisions) = &self.decisions else {
            return Err(crate::decision::SystemOneError::Unsupported(format!(
                "{} has no decision template: it is not a native decision checkpoint and \
                 decision.template selected none",
                self.model_name
            )));
        };
        self.systemone_images_in_scope(body, &decisions.policy, SERVED_DECISION_IMAGES)
    }
    fn systemone_batch(
        &self,
        bodies: &[&crate::decision::OrderedJson],
    ) -> Vec<std::result::Result<crate::decision::SystemOneResponse, crate::decision::SystemOneError>>
    {
        match &self.decisions {
            Some(decisions) => {
                self.systemone_together(bodies, &decisions.policy, SERVED_DECISION_IMAGES)
            }
            None => bodies.iter().map(|body| self.systemone(body)).collect(),
        }
    }
    fn decision_binding(&self) -> Option<(Option<crate::decision::binding::DecisionType>, String)> {
        self.decisions
            .as_ref()
            .map(|decisions| (decisions.native, decisions.origin.clone()))
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        let mut w = self.load_warnings.clone();
        w.extend(lloom::load_warning::emulated(&self.core.device));
        w
    }
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>) {
        self.load_warnings.splice(0..0, warnings);
    }
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        Engine::encode(self, text, add_special)
    }
    fn token_str(&self, id: u32) -> String {
        Engine::token_str(self, id)
    }
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        Engine::render_checked(self, messages, kwargs, checked)
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
        } else if self.dspark_k.is_some() {
            lloom::nn::Body::Window
        } else {
            lloom::nn::Body::Step {
                ring: matches!(self.readback, ReadbackArm::Ring(_)),
            }
        }
    }
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32> {
        Engine::seed_prompt_row(self, n, row)
    }
    fn complete_until(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        Engine::complete_until(self, prompt_ids, params, |frag| on_fragment(frag))
    }
    fn image_placeholder(&self, image: &DecodedImage) -> Result<String> {
        Engine::image_placeholder(self, image)
    }
    fn complete_images_until(
        &self,
        prompt_ids: &[u32],
        images: &[DecodedImage],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        Engine::complete_images_until(self, prompt_ids, images, params, |frag| on_fragment(frag))
    }
    fn vision_report(&self) -> Value {
        lock(&self.last_vision).clone()
    }
    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        Engine::prefill_throughput(self, n, batch)
    }
    fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming> {
        Engine::decode_throughput(self, prompt_tokens, decode_tokens, batch)
    }
    fn next_token_logprobs(&self, prompt_ids: &[u32], k: usize) -> Result<(u32, Vec<(u32, f32)>)> {
        Engine::next_token_logprobs(self, prompt_ids, k)
    }
    fn check_batch(&self, batch: usize) -> Result<()> {
        Engine::check_batch(self, batch)
    }
    fn shutdown(&self) {
        Engine::shutdown(self)
    }
    fn installed(&self) -> Option<&super::report::Installed> {
        Some(&self.installed)
    }
    fn diagnostics(&self) -> Vec<String> {
        Engine::diagnostics(self)
    }
    fn run_mode(&self, spec: &super::modes::ModeSpec) -> Result<()> {
        Engine::run_mode(self, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom::nn::ops::KvQuant;

    #[test]
    fn conv_carry_reads_configured_tile_and_chunk_without_legacy_fields() {
        let config = Config::default();
        assert!(!config.conv_prefill_ctx);
        for (tile, chunk) in [(16, 0), (0, 8)] {
            let mut decode = crate::DecodeConfig::default();
            decode.prefill.tile = tile;
            decode.prefill.chunk = chunk;
            let resolved = decode.options(1, ReadbackArm::Blocking, 0);
            assert_eq!(resolved.prefill_tile, tile);
            assert_eq!(resolved.prefill_chunk, chunk);
            assert!(conv_carry_needed(&config, 0, 0, &decode, false));
        }
    }

    #[test]
    fn conv_carry_keeps_legacy_tile_and_chunk_independently() {
        let config = Config::default();
        let mut decode = crate::DecodeConfig::default();
        decode.prefill.tile = 0;
        for (tile, chunk) in [(16, 0), (0, 8)] {
            assert!(conv_carry_needed(&config, tile, chunk, &decode, false));
        }
        assert!(!conv_carry_needed(&config, 0, 0, &decode, false));
    }

    #[test]
    fn conv_carry_keeps_spark_reuse_and_explicit_axis_but_not_zero_split() {
        let mut config = Config::default();
        let mut decode = crate::DecodeConfig::default();
        decode.prefill.tile = 0;
        assert!(!conv_carry_needed(&config, 0, 0, &decode, false));

        decode.prefill.ubatch = 3;
        assert!(conv_carry_needed(&config, 0, 0, &decode, false));
        decode.prefill.ubatch = 0;
        decode.prefill.spark = Some("progressive:1,16/tile:16".into());
        let resolved = decode.options(1, ReadbackArm::Blocking, 0);
        assert_eq!(resolved.prefill_schedule, Some(vec![1, 16]));
        assert_eq!(resolved.prefill_bulk, 16);
        assert!(conv_carry_needed(&config, 0, 0, &decode, false));
        decode.prefill.spark = None;

        assert!(conv_carry_needed(&config, 0, 0, &decode, true));
        assert!(!conv_carry_needed(&config, 0, 0, &decode, false));
        config.conv_prefill_ctx = true;
        assert!(conv_carry_needed(&config, 0, 0, &decode, false));
    }

    #[test]
    fn single_throughput_request_construction_preserves_tile_and_group_fallback() {
        let mut decode = crate::DecodeConfig::default();
        decode.prefill.tile = 11;
        decode.prefill.chunk = 5;
        for readback in [
            ReadbackArm::Auto,
            ReadbackArm::Blocking,
            ReadbackArm::Ring(crate::RingArm::CROWN),
        ] {
            for (tile, effective) in [(4, 4), (0, 11)] {
                for max_tokens in [1, 7] {
                    let params = throughput_params(max_tokens, readback, tile);
                    assert_eq!(params.prefill_tile, tile);
                    assert_eq!(params.max_tokens, max_tokens);
                    assert_eq!(params.readback, readback);
                    assert_eq!(params.eos_ids, NEVER_EOS);
                    assert_eq!(params.seed, 42);
                    assert!(params.temperature.is_none());
                    assert!(params.top_p.is_none());
                    assert!(params.top_k.is_none());
                    assert!(!params.gumbel);
                    assert!(!params.processors.is_active());
                    assert!(params.stop_strings.is_empty());
                    assert!(!params.logits_digest);
                    assert!(params.thinking.is_none());
                    assert_eq!(params.turn_tail, 0);
                    assert!(params.logprobs.is_none());
                    assert!(!params.cold);
                    // The window body resolves with Blocking even when the
                    // captured request arm is Auto.
                    let resolved = decode.options(
                        params.max_tokens,
                        ReadbackArm::Blocking,
                        params.prefill_tile,
                    );
                    assert_eq!(resolved.max_tokens, max_tokens);
                    assert_eq!(resolved.prefill_tile, effective);
                    assert_eq!(resolved.prefill_chunk, 5);
                }
            }
        }
    }

    /// `--dspark-ctx` (the tuning knob) reaches the drafter on every arm;
    /// `auto` is the pool arms' shared pool at N > 1, else per-sequence
    /// pages on Metal, else contiguous.
    #[test]
    fn the_drafter_context_kv_follows_the_knob() {
        let per_seq = CtxKv::Paged {
            page_size: CtxKv::DEFAULT_PAGE_SIZE,
            shared: false,
        };
        let shared = CtxKv::Paged {
            page_size: CtxKv::DEFAULT_PAGE_SIZE,
            shared: true,
        };
        assert_eq!(
            drafter_ctx_kv(Some(CtxKv::Contiguous), true, true, 4),
            CtxKv::Contiguous,
            "the knob naming the contiguous arm is a choice on every arm"
        );
        assert_eq!(
            drafter_ctx_kv(Some(per_seq), true, true, 4),
            per_seq,
            "the knob beats the pool default"
        );
        assert_eq!(
            drafter_ctx_kv(None, true, true, 4),
            shared,
            "auto, pool, N > 1"
        );
        assert_eq!(
            drafter_ctx_kv(None, true, true, 1),
            per_seq,
            "auto, pool, N = 1"
        );
        assert_eq!(
            drafter_ctx_kv(None, true, false, 1),
            per_seq,
            "auto, private, Metal"
        );
        assert_eq!(
            drafter_ctx_kv(None, true, false, 4),
            per_seq,
            "auto, private, N > 1"
        );
        assert_eq!(
            drafter_ctx_kv(None, false, false, 1),
            CtxKv::Contiguous,
            "auto on the CPU"
        );
    }

    fn config(quant: KvQuant, append: KvAppend) -> Config {
        let mut c = Config::default();
        c.kv_quant = quant;
        c.kv_append = append;
        c
    }

    /// The speculative body at one sequence on the paged KV is named on the
    /// load-warning channel; the in-place and cat KVs, no drafter and a
    /// batch stay silent.
    #[test]
    fn a_paged_speculative_body_at_one_sequence_warns() {
        let w = paged_verify_warning(&config(KvQuant::Off, KvAppend::Paged), true, 1)
            .expect("the paged KV under a drafter at one sequence warns");
        assert_eq!(w.kind, LoadWarningKind::Speculation);
        assert!(
            w.message.contains("context.kv.layout = paged")
                && w.message.contains("--kv-append in-place"),
            "{}",
            w.message
        );
        for (append, speculative, batch) in [
            (KvAppend::InPlace, true, 1),
            (KvAppend::Cat, true, 1),
            (KvAppend::Paged, false, 1),
            (KvAppend::Paged, true, 4),
        ] {
            assert_eq!(
                paged_verify_warning(&config(KvQuant::Off, append), speculative, batch),
                None,
                "{append:?} speculative={speculative} batch={batch}"
            );
        }
    }

    #[test]
    fn the_quantized_arms_are_refused_where_they_have_no_path() {
        let err = refuse_kv_quant_shape(&config(KvQuant::Int8, KvAppend::Paged), &Device::Cpu, 4)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("kv-quant = int8 has no arm on the paged KV pool"),
            "{err}"
        );
        assert!(err.contains("`--rows private`"), "{err}");
        let err = refuse_kv_quant_shape(&config(KvQuant::Int4, KvAppend::Paged), &Device::Cpu, 1)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`--cache pool|pool-spec` or `--kv-append paged`"),
            "{err}"
        );
        let err = refuse_kv_quant_shape(&config(KvQuant::Int8, KvAppend::InPlace), &Device::Cpu, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("kv-quant = int8 is Metal-only"), "{err}");
        // The Off arm is every shape's.
        for append in [KvAppend::Cat, KvAppend::InPlace, KvAppend::Paged] {
            refuse_kv_quant_shape(&config(KvQuant::Off, append), &Device::Cpu, 4).unwrap();
        }
    }
}

pub(crate) mod decision_media;

#[cfg(test)]
mod speculative_schedule_tests;

#[cfg(test)]
mod single_speculative_tests;
