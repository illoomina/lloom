//! The loaded model and the operations every endpoint is a thin shape over.
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

use anyhow::{Context, Result};
use hf_chat_template::ChatTemplate;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{QkvLayout, 
    AttnImpl, BatchPool, Cache, Config, ConvState, KvAppend, Lfm2Config, MemoryBudget, Model,
};
use lloom::models::lfm2_dspark::{CtxKv, DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, Sequence, Session, VarBuilder};
use lloom_engine::{
    ring_eos_read, spawn_batch, BatchConfig, BatchEvent, BatchHandle, BatchMode, BatchRequest,
    DecodeParams, DecodeSession, Decoded, Idle, ReadbackArm, RingPool, RowSpec, SlotPool, Stop,
    TextStream,
};
use lloom_tensor::{DType, Device, Tensor, D};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// The EOS id set that never fires: every arm's EOS mechanism still runs and
/// simply never matches, so a run produces exactly `max_tokens` tokens.
pub const NEVER_EOS: [u32; 1] = [u32::MAX];

/// The model and the device: the half every request borrows, shared with
/// the batch thread.
pub struct Core {
    pub model: Model,
    pub device: Device,
}

/// The `--batch 1` state: the one sequence every request drives, taken out
/// for the request's session and put back after it, the token rings kept
/// across requests (`RingPool`, built at load, #702), and the drafter.
struct Single {
    seq: Option<Sequence<Cache>>,
    rings: Option<RingPool>,
    spec: Option<DSparkSpeculator>,
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

impl lloom_engine::MakeDriver<Model> for Speculators {
    fn make_driver<'m>(
        &mut self,
        model: &'m Model,
        opts: &DecodeOptions,
        device: &Device,
    ) -> lloom_tensor::Result<Option<lloom_engine::Driver<'m, Model>>> {
        let _ = model;
        if !self.continuous {
            return Ok(None);
        }
        match self.spec.take() {
            Some(spec) => {
                let dec = lloom::nn::SpecBatchDecoder::<
                    'm,
                    Model,
                    DSparkSpeculator,
                    Session<'m, Model>,
                >::new(spec, opts.clone(), &self.eos_ids, device.clone())?;
                Ok(Some(Box::new(dec)))
            }
            None => Ok(None),
        }
    }
}

pub type Extra = Speculators;

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

pub struct Engine {
    pub core: Arc<Core>,
    pub text: Text,
    pub model_name: String,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    /// Reuse the cached prefix across requests (`Session::prepare`). Off by
    /// default: a benchmark cell measures a cold prefill, and a repeated prompt
    /// would otherwise resume from the previous request's state.
    pub reuse_prefix: bool,
    /// The batch width `N` of `--batch N`; `1` is the single-sequence path.
    pub batch: usize,
    /// The draft block width under `--dspark`.
    pub dspark_k: Option<usize>,
    pub prefill_tile: usize,
    pub prefill_chunk: usize,
    pub memory_fraction: f64,
    pub mem_sample: bool,
    /// How the rows of a batched step hold their state (`--rows`).
    pub rows: RowsArm,
    /// The rows' shared pool on the `pool` arms (`batched-decode-design.md`
    /// §12.2); `None` on `private` and at `--batch 1`.
    pub pool: Option<BatchPool>,
    /// `--share-prefix`.
    pub share_prefix: bool,
    single: Mutex<Single>,
    sched: Option<BatchHandle<Model, Extra>>,
    sched_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

pub struct LoadOptions {
    pub model_dir: PathBuf,
    pub cpu: bool,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
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
            other => anyhow::bail!("--rows must be auto, pool, pool-per-row or private, got {other:?}"),
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
fn profile_reset() {
    #[cfg(feature = "metal")]
    if lloom_tensor::kernels::metal::profile::enabled() {
        lloom_tensor::kernels::metal::profile::reset();
    }
}

/// The recorder's snapshot after a bracket the caller has synchronised
/// (every completion handler has run), or `None` when it is off.
fn profile_take() -> Option<ProfileReport> {
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

impl Single {
    /// Run `f` over a session on the held sequence, and put the sequence back
    /// whatever `f` returned.
    fn with_session<T>(
        &mut self,
        core: &Core,
        f: impl FnOnce(&mut DecodeSession<'_, Model>, &mut Option<DSparkSpeculator>) -> Result<T>,
    ) -> Result<T> {
        let seq = self
            .seq
            .take()
            .context("the sequence is checked out — a previous request panicked mid-decode")?;
        let rings = self.rings.take().unwrap_or_default();
        let mut session = DecodeSession::from_parts(&core.model, seq, core.device.clone(), rings);
        let out = f(&mut session, &mut self.spec);
        let (seq, rings) = session.into_parts();
        self.seq = Some(seq);
        self.rings = Some(rings);
        out
    }
}

impl Engine {
    pub fn load(opts: LoadOptions) -> Result<Self> {
        let model_dir = opts.model_dir;
        let device = if opts.cpu {
            Device::Cpu
        } else {
            open_device()?
        };
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
        let config: Lfm2Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
                .context("parsing config.json")?;
        let mut config = config.into_config(false);

        // The drafter's config first: the target's cache arms depend on it
        // (the verify window sizes the conv ring, the taps feed the drafter).
        let dspark_dir = match &opts.dspark {
            None => None,
            Some(p) if p.as_os_str().is_empty() => Some(
                newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark").context(
                    "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir; the \
                     drafter is trained against THIS target (design/docs/speculation.md §10.2l)",
                )?,
            ),
            Some(p) => Some(p.clone()),
        };
        // The speculative arm's pool budget: 1024 MiB, sized from measurement
        // (#148, the B=1 defaults pass; `lfm2-chat` applies the same default):
        // a round is two models' activations plus the verify's logits,
        // released a few rounds later under the nonblocking readback, and at
        // the 256 MiB ceiling that working set is evicted as it frees — ~10
        // allocations and a residency commit per round; at 1024, none.
        #[cfg(feature = "metal")]
        if dspark_dir.is_some() {
            if let Device::Metal(m) = &device {
                m.set_free_budget(1024 * 1024 * 1024);
            }
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
        match &dcfg {
            Some(dcfg) if pooled => pool_spec_config(&mut config, dcfg),
            Some(dcfg) => rewindable_config(&mut config, dcfg),
            None if pooled => pool_config(&mut config),
            None => crown_config(&mut config),
        }
        // A prefill fed in more than one forward — a tile, a chunk — needs
        // the conv's left-context carry, or the conv zero-pads each piece's
        // left edge and computes something one pass does not (§13.2f
        // defect 1). One forward is identical either way.
        config.conv_prefill_ctx = opts.prefill_tile > 0 || opts.prefill_chunk > 0;
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
            budget = budget.with_prefill(if opts.prefill_tile > 0 {
                opts.prefill_tile.max(opts.prefill_chunk)
            } else {
                opts.kv_capacity
            });
            if opts.batch > 1 {
                // The scheduler's step is ONE ragged forward: the chunk's
                // prompt tokens plus one token per live row, whatever `N` —
                // not `N` forwards of a tile each. KV and conv stay per row.
                budget.max_seq_len += opts.batch;
                budget.activation_rows = Some(1);
            }
            config.memory_budget = Some(budget);
        }

        let files = weight_files(&model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        let model = Model::new(&config, vb).context("constructing LFM2")?;
        let make_cache = |device: &Device| {
            Cache::new_with(true, dtype, &config, device, opts.kv_capacity)
                .context("allocating the KV cache")
        };
        let cache = make_cache(&device)?;

        // Stop on the checkpoint's declared EOS plus `<|im_end|>` (the chat
        // turn end), as `lfm2-chat` does.
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
            Ok(src) => Some(
                ChatTemplate::from_str(&src)
                    .map_err(|e| anyhow::anyhow!("compiling chat_template.jinja: {e}"))?,
            ),
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
            .unwrap_or_else(|| model_name_from_dir(&model_dir));

        let core = Arc::new(Core { model, device });
        let mut engine = Engine {
            core: Arc::clone(&core),
            text: Text {
                tokenizer,
                template,
                bos_token,
                bos_id,
                eos_ids,
                seed_tokens,
            },
            model_name,
            kv_capacity: opts.kv_capacity,
            readback: opts.readback,
            reuse_prefix: opts.reuse_prefix,
            batch: opts.batch,
            dspark_k: dcfg.as_ref().map(|d| d.max_draft()),
            prefill_tile: opts.prefill_tile,
            prefill_chunk: opts.prefill_chunk,
            memory_fraction: opts.memory_fraction,
            mem_sample: opts.mem_sample,
            rows: if opts.batch > 1 { opts.rows } else { RowsArm::Private },
            pool: None,
            share_prefix: opts.share_prefix && opts.batch > 1 && opts.rows.pooled(),
            single: Mutex::new(Single {
                seq: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                spec: None,
            }),
            sched: None,
            sched_thread: Mutex::new(None),
        };

        if opts.warmup {
            let (libs, pipes) = engine.with_single(|s, _| Ok(s.warmup()?))?;
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
            // plus the block and slack (`lfm2-chat`'s sizing).
            let max_pos = opts.kv_capacity + dcfg.block_size + 64;
            let dvb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, &core.device)?
            };
            let drafter = DSparkDrafter::new(dcfg.clone(), max_pos, dvb)
                .context("constructing DSpark drafter")?;
            // The drafter's context KV: on the pool arms at N > 1 one paged
            // pool set shared by every sequence — the shape the batched
            // draft's block attention reads in ONE dispatch (§12.9); a
            // per-sequence pool or the contiguous arm attends per row. The
            // static pipelined driver (private) keeps the contiguous default
            // (§10.2z: a shared pool costs THAT driver its overlap). An
            // explicit `LLOOM_DSPARK_CTX_KV` wins on every arm.
            let ctx_kv = match std::env::var("LLOOM_DSPARK_CTX_KV") {
                Ok(v) => CtxKv::parse(&v)?,
                Err(_) if pooled && n_seq > 1 => CtxKv::Paged {
                    page_size: CtxKv::DEFAULT_PAGE_SIZE,
                    shared: true,
                },
                Err(_) => CtxKv::Contiguous,
            };
            let mut spec = DSparkSpeculator::new_with_ctx(
                drafter,
                core.model.draft_embedding().clone(),
                core.model.lm_head_linear().clone(),
                n_seq,
                ctx_kv,
            )
            .context("constructing DSpark speculator")?;
            eprintln!(
                "[dspark: drafter loaded from {}, {} layers, K={} (verify window {} positions), \
                 {n_seq} slots, context kv {}]",
                dir.display(),
                dcfg.num_hidden_layers,
                dcfg.max_draft(),
                dcfg.max_draft() + 1,
                spec.ctx_kv().name()
            );
            if opts.warmup {
                // The batched draft's pipelines too (`m = n_seq·γ`, the
                // ragged paged attention, the batched Markov chain).
                spec.warmup_rows(n_seq, &core.device)
                    .context("warming DSpark drafter pipelines")?;
            }
            Ok(spec)
        };

        if opts.batch == 1 {
            if dcfg.is_some() {
                let spec = build_spec(1)?;
                lock(&engine.single).spec = Some(spec);
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
                    Cache::for_batch(true, dtype, &config, &core.device, &pool)
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
        let batch_opts = DecodeOptions {
            lookahead: 1,
            prefill_tile: opts.prefill_tile,
            prefill_chunk: opts.prefill_chunk,
            ..DecodeOptions::new(0)
        };
        // `--dspark` on private caches runs the static pipelined driver; on
        // the pool arms the continuous speculative driver (§12.4).
        let mode: BatchMode<Model, Extra> = if dcfg.is_some() && !pooled {
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
            mode,
            extra,
            on_counters: None,
        };
        let (handle, thread) = spawn_batch(Arc::clone(&core), |c: &Core| &c.model, cfg)
            .map_err(|e| anyhow::anyhow!("starting the batch scheduler: {e}"))?;
        eprintln!(
            "[batch: scheduler up, {} sequences, {} rows per step, tile {} chunk {}, {}]",
            opts.batch,
            opts.batch,
            opts.prefill_tile,
            opts.prefill_chunk,
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

    fn with_single<T>(
        &self,
        f: impl FnOnce(&mut DecodeSession<'_, Model>, &mut Option<DSparkSpeculator>) -> Result<T>,
    ) -> Result<T> {
        lock(&self.single).with_session(&self.core, f)
    }

    /// Run a closed job on the batch thread and wait for it.
    pub fn closed<T: Send + 'static>(
        &self,
        job: impl for<'a, 'm> FnOnce(&mut Idle<'a, 'm, Model, Extra>) -> T + Send + 'static,
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

    /// Render `messages` through the checkpoint's own chat template with the
    /// generation prompt appended. `kwargs` are extra template variables
    /// (`chat_template_kwargs`, e.g. `enable_thinking`).
    pub fn render_chat(&self, messages: &Value, kwargs: Option<&Value>) -> Result<String> {
        let template =
            self.text.template.as_ref().context(
                "the checkpoint has no chat_template.jinja; /v1/chat/completions needs one",
            )?;
        let mut ctx = json!({
            "messages": messages,
            "bos_token": self.text.bos_token,
            "add_generation_prompt": true,
        });
        if let Some(Value::Object(extra)) = kwargs {
            for (k, v) in extra {
                ctx[k] = v.clone();
            }
        }
        template
            .render_context(&ctx)
            .map_err(|e| anyhow::anyhow!("rendering the chat template: {e}"))
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
        if let Some(sched) = &self.sched {
            return self.complete_batched(sched, prompt_ids, params, on_fragment);
        }
        let reuse = self.reuse_prefix;
        let detok = self.detok();
        self.with_single(|s, spec| {
            if !reuse {
                s.reset();
            }
            match spec {
                Some(spec) => s
                    .decode_speculative_until(spec, prompt_ids, detok, params, on_fragment)
                    .map_err(anyhow::Error::from),
                None => s
                    .decode_until(prompt_ids, detok, params, on_fragment)
                    .map_err(anyhow::Error::from),
            }
        })
    }

    /// A request through the batch scheduler: submitted, its tokens read off
    /// the channel and streamed as text, its outcome the row's.
    fn complete_batched(
        &self,
        sched: &BatchHandle<Model, Extra>,
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
                    if let Some(chunk) = stream.push(&ids, &mut detok)? {
                        if !on_fragment(&chunk) {
                            cancelled = true;
                            cancel.store(true, Ordering::Release);
                            continue;
                        }
                    }
                    if stream.hit_stop(&params.stop_strings) {
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
        let (text, _) = stream.finish(&params.stop_strings);
        Ok(Decoded {
            text,
            tokens: done.tokens,
            stop,
            reuse: done.reuse,
            reused: done.reused,
            prefilled: done.prefilled,
            readback: match self.dspark_k {
                Some(k) => ReadbackArm::Speculative { k },
                None => ReadbackArm::Batched {
                    rows: done.joined_beside,
                },
            },
            stats: done.stats,
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
        self.check_fits(prompt_tokens, decode_tokens + 1)?;
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
                let out =
                    if let Some(k) = dspark_k {
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
                            o.stats.rounds.iter().map(|r| r.accepted as f64).sum::<f64>() / n
                        })
                        .collect(),
                    row_draft_ms: outs
                        .iter()
                        .map(|o| o.stats.rounds.iter().map(|r| r.timing.draft_ns).sum::<u64>() as f64 / 1e6)
                        .collect(),
                    row_verify_ms: outs
                        .iter()
                        .map(|o| o.stats.rounds.iter().map(|r| r.timing.verify_ns).sum::<u64>() as f64 / 1e6)
                        .collect(),
                    row_observe_ms: outs
                        .iter()
                        .map(|o| o.stats.rounds.iter().map(|r| r.timing.observe_ns).sum::<u64>() as f64 / 1e6)
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
                    bracket: "D batched steps after every row's context, resumed from held logits",
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
        let mut timing = self.with_single(|s, spec| {
            s.reset();
            if let Some(spec) = spec {
                // One run of `D` tokens; the prompt's submit time subtracted
                // (the prefill's GPU tail is inside the bracket).
                let params = DecodeParams {
                    max_tokens: decode_tokens,
                    eos_ids: NEVER_EOS.to_vec(),
                    readback,
                    ..DecodeParams::default()
                };
                let t = Instant::now();
                let run = s.decode_speculative_until(spec, &prompt, detok, &params, |_| true)?;
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
            // The whole sequence so far; `Session::prepare` resumes from the
            // logits the context run held rather than re-feeding anything.
            let mut all: Vec<u32> = prompt.clone();
            all.extend_from_slice(&first.tokens);
            let params = DecodeParams {
                max_tokens: decode_tokens,
                eos_ids: NEVER_EOS.to_vec(),
                readback,
                prefill_tile,
                ..DecodeParams::default()
            };
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

    fn check_fits(&self, prompt: usize, gen: usize) -> Result<()> {
        let need = prompt + gen;
        if need > self.kv_capacity {
            anyhow::bail!(
                "prompt {prompt} + generation {gen} = {need} positions exceeds the KV capacity \
                 {} (start lloom-serve with --kv-capacity >= {need})",
                self.kv_capacity
            );
        }
        Ok(())
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
struct TakenSessions<'m> {
    sessions: Vec<Session<'m, Model>>,
    slots: Vec<usize>,
}

impl<'m> TakenSessions<'m> {
    fn give_back(self, idle: &mut Idle<'_, 'm, Model, Extra>) {
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
fn with_sessions<'m, T>(
    idle: &mut Idle<'_, 'm, Model, Extra>,
    n: usize,
    f: impl FnOnce(&mut Idle<'_, 'm, Model, Extra>, &mut Vec<Session<'m, Model>>) -> Result<T>,
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

fn take_sessions<'m>(idle: &mut Idle<'_, 'm, Model, Extra>, n: usize) -> Result<TakenSessions<'m>> {
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
fn run_spec_rows<'m>(
    pooled: bool,
    idle: &mut Idle<'_, 'm, Model, Extra>,
    sessions: &mut Vec<Session<'m, Model>>,
    prompts: &[&[u32]],
    opts: &DecodeOptions,
    eos_ids: &[u32],
    on_tokens: &mut dyn FnMut(usize, &[u32]) -> lloom_tensor::Result<()>,
) -> Result<Vec<lloom::nn::DecodeOut>> {
    if pooled {
        let owned = std::mem::take(sessions);
        let (back, outs) = lloom_engine::drive_closed_rows(
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
        spec,
        prompts,
        opts,
        eos_ids,
        idle.device,
        on_tokens,
    )?)
}

/// The static body under `--dspark --batch N --rows private`: every row's
/// round pipelined through the one drafter
/// (`Session::decode_speculative_pipelined`, §10.2y). Greedy, one EOS set
/// per batch — mixed sets are refused rather than served under the first
/// row's.

fn static_dspark_body<'a, 'm>(
    idle: &mut Idle<'a, 'm, Model, Extra>,
    sessions: &mut [Session<'m, Model>],
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
        if s.sampling != Sampling::ArgMax {
            lloom_tensor::bail!(
                "row {i}: the speculative body decodes greedily; a temperature is refused"
            );
        }
        if s.eos_ids != first.eos_ids {
            lloom_tensor::bail!(
                "row {i}: a static speculative batch runs one EOS set; this row's differs from \
                 row 0's"
            );
        }
    }
    let max_tokens = specs.iter().map(|s| s.max_tokens).max().unwrap_or(0);
    let opts = DecodeOptions::new(max_tokens);
    let caps: Vec<usize> = specs.iter().map(|s| s.max_tokens).collect();
    let mut outs = Session::decode_speculative_pipelined(
        sessions,
        spec,
        prompts,
        &opts,
        &first.eos_ids,
        idle.device,
        &mut |i, ids| {
            // Every row runs to the batch's widest cap; a row past its own
            // is cut here and below.
            on_tokens(i, ids)
        },
    )?;
    for (o, cap) in outs.iter_mut().zip(caps) {
        if cap > 0 && o.ids.len() > cap {
            o.ids.truncate(cap);
            o.stats.produced = cap;
        }
    }
    Ok(outs)
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

/// The crown decode configuration `lfm2-chat --cache crown` bakes in: the
/// `Sdpa × InPlace × Shuffle` arms and the GEMV/conv fusions, every one
/// token-identical to the generic path (§13.5, §12.2d, §12.2h). The library's
/// `Config` ships the conservative arms (§7.1a); a serving binary turns the
/// measured ones on.
/// The rows' shared-pool configuration (`--rows pool`, `batched-decode-design.md`
/// §12.2): the paged pool the slot-indexed kernels read, the in-place ring
/// they write, the crown's fused epilogues.
fn pool_config(config: &mut Config) {
    // The fused q/k/v layout (`lfm2-chat --qkv-layout=fused`): no `.contiguous()`
    // copies of the three views at decode — the norms read the transposed view
    // strided and V lands in the slab directly; bit for bit the contiguous path.
    // Measured on the serving line (#148, the B=1 defaults pass, record §8):
    // lfm2-chat 64.0 / 63.7 vs 63.6 / 63.5, the server's B=1 cell 63.1 / 63.1 /
    // 63.2 vs 62.8 / 63.0 / 63.0 (six alternations) — +0.2–0.3 tok/s, every
    // pair the same way; a wash at B=8 (235.3 / 232.2 vs 235.6 / 234.6).
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::FlashDecoding;
    config.kv_append = KvAppend::Paged;
    config.conv_state = ConvState::RotatingRing { k: 1 };
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
}

/// The pool arms under `--dspark` (§12.4): the paged pool and the flash
/// read, a ring wide enough for the verify window, the drafter's taps.
fn pool_spec_config(config: &mut Config, dcfg: &DSparkConfig) {
    let k = dcfg.max_draft();
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::FlashDecoding;
    config.kv_append = KvAppend::Paged;
    config.conv_state = ConvState::RotatingRing { k: k + 1 };
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
    config.tap_dtype = DType::F32;
    config.tap_layers = dcfg.tap_layers();
}

fn crown_config(config: &mut Config) {
    // The fused q/k/v layout (`lfm2-chat --qkv-layout=fused`): no `.contiguous()`
    // copies of the three views at decode — the norms read the transposed view
    // strided and V lands in the slab directly; bit for bit the contiguous path.
    // Measured on the serving line (#148, the B=1 defaults pass, record §8):
    // lfm2-chat 64.0 / 63.7 vs 63.6 / 63.5, the server's B=1 cell 63.1 / 63.1 /
    // 63.2 vs 62.8 / 63.0 / 63.0 (six alternations) — +0.2–0.3 tok/s, every
    // pair the same way; a wash at B=8 (235.3 / 232.2 vs 235.6 / 234.6).
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::Sdpa;
    config.kv_append = KvAppend::InPlace;
    config.conv_state = ConvState::Shuffle;
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
}

/// The REWINDABLE configuration the speculative body needs (`lfm2-chat
/// --cache rewindable`): `Cache::advance` rewinds only under a rotating ring
/// wide enough for the verify window (the shuffle is destructive) and
/// `Cache::resolve` only under `InPlace` (the `Cat` arm has no length to
/// rewind); the taps feed the drafter's projection at F32 (§10.2m).
fn rewindable_config(config: &mut Config, dcfg: &DSparkConfig) {
    let k = dcfg.max_draft();
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::Sdpa;
    config.kv_append = KvAppend::InPlace;
    config.conv_state = ConvState::RotatingRing { k: k + 1 };
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
    config.tap_dtype = DType::F32;
    config.tap_layers = dcfg.tap_layers();
}

fn open_device() -> Result<Device> {
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

/// The newest snapshot of `repo` in the HF cache, if any.
fn newest_snapshot(repo: &str) -> Option<PathBuf> {
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

/// The LFM2.5-2.6B snapshot in the HF cache, if it is there
/// (`LFM2_MODEL_DIR` first).
pub fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    newest_snapshot("models--LiquidAI--LFM2.5-2.6B")
}

/// `models--Org--Name/snapshots/<hash>` → `Org/Name`; a plain directory → its
/// name.
fn model_name_from_dir(dir: &Path) -> String {
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
