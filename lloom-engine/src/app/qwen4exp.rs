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
//! target, byte-identical to greedy (issue #749).
//!
//! Not built for this family yet: `--dspark` (LFM2's drafter). `--batch N`
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
use lloom::models::qwen4exp::loader::{
    LoadOptions as ModelLoad, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::{Cache, ExpertResidency, MtpDrafter, Qwen4Exp, Resident, Streamed};
use lloom::nn::expert_stream::HasStreamStats;
use lloom::nn::model::CausalLM;
use lloom::nn::Sequence;
use lloom::nn::{DecodeOptions, Session};
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

/// The model, shared with the batch thread.
pub struct QwenCore<R: ExpertResidency + HasStreamStats> {
    pub model: Qwen4Exp<R>,
}

/// The one sequence every request drives, and the rings kept across
/// requests.
struct Single {
    seq: Option<Sequence<Cache>>,
    rings: Option<RingPool>,
    /// The loop's options, handed to every request's session.
    decode: crate::DecodeConfig,
}

pub struct Engine<R: ExpertResidency + HasStreamStats> {
    core: Arc<QwenCore<R>>,
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
    /// The hotlist to write at shutdown (the usage since load merged in).
    hotlist: Option<PathBuf>,
    /// The tier's load-time description (`[loaded …]`, `/health`).
    tier: Value,
    single: Mutex<Single>,
    /// `--batch N`: the scheduler and its thread.
    sched: Option<BatchHandle<Qwen4Exp<R>, ()>>,
    sched_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Why a ring arm on this engine cannot take the PLE's host fill
/// (`CausalLM::host_feed`), or `None` when it can. Without the fill a ring
/// arm's forward reads its fed id back.
fn host_fill_absent<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
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
            let why = no_fill.or((!crate::options::stamps_each_token(ring)).then_some(
                "a deferred EOS read on the fixed cadence stamps once per window, and the fill \
                 waits on each token's stamp",
            ));
            let note = match why {
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

impl Engine<Resident> {
    /// Load the family `opts` names, on the tier `opts.experts` names.
    pub fn load(opts: LoadOptions) -> Result<Box<dyn Served>> {
        let device = if opts.cpu {
            Device::Cpu
        } else {
            open_device()?
        };
        let mut shards = discover_shards(&opts.first_shard);
        if let Some(p) = &opts.mtp {
            if !p.is_file() {
                anyhow::bail!("--mtp {}: no such file", p.display());
            }
            shards.push(p.clone());
        }
        let moe_act = match opts.moe_act.as_str() {
            "f32" => None,
            "f16" => Some(DType::F16),
            other => anyhow::bail!("--moe-act {other}: f32 | f16"),
        };
        let mload = ModelLoad {
            ple_shard: None,
            ple: opts.ple,
            moe_act,
            ple_cache_rows: (opts.ple_cache_rows > 0).then_some(opts.ple_cache_rows),
            mtp: opts.mtp.is_some(),
            memory_fraction: Some(opts.memory_fraction),
            kv_precision: opts.kv_precision,
        };
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
                let mut model: Qwen4Exp<Resident> =
                    Qwen4Exp::from_gguf_shards(&shards, &device, &mload)
                        .context("loading Qwen4-exp (resident experts)")?;
                model.set_prefill_mode(prefill_mode);
                model.set_memory_fraction(opts.memory_fraction);
                let tier = json!({ "experts": "resident", "shards": shards.len(), "load_ms": ms(t.elapsed()), "prefill_mode": opts.prefill_mode });
                Engine::<Resident>::finish(model, device, opts, tier, installed)
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
                };
                let mut model: Qwen4Exp<Streamed> =
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
                Engine::<Streamed>::finish(model, device, opts, tier, installed)
            }
            other => anyhow::bail!("--experts {other}: resident | stream"),
        }
    }
}

impl<R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Engine<R> {
    fn finish(
        model: Qwen4Exp<R>,
        device: Device,
        opts: LoadOptions,
        mut tier: Value,
        installed: super::report::Installed,
    ) -> Result<Box<dyn Served>> {
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
        let (readback, note) = resolve_readback(opts.readback, host_fill_absent(&model, &device));
        if let Some(note) = note {
            eprintln!("[readback {note}]");
        }
        let mtp = opts.mtp.is_some();
        if mtp && model.mtp_head().is_none() {
            anyhow::bail!("--mtp was given but the loader found no MTP head in the shards");
        }
        let mut cache = if mtp {
            // The verify window's states in phase rings: an accept by phase.
            let pools = model.spec_pools(1, opts.mtp_k.max(1), &device);
            Cache::in_pools(model.config(), &pools)?
        } else {
            model.new_cache(model.config(), &device)?
        };
        cache.want_nextn = mtp;
        tier["mtp"] = json!(mtp);
        tier["ple"] = json!(opts.ple);
        tier["ple_cache_rows"] = json!(opts.ple_cache_rows);
        tier["moe_act"] = json!(opts.moe_act);
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
            hotlist: opts.hotlist.clone(),
            tier,
            single: Mutex::new(Single {
                seq: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                decode: opts.decode.clone(),
            }),
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
            let cfg_model = engine.core.model.config().clone();
            let pools = engine.core.model.state_pools(opts.batch, &engine.device);
            let mut caches: Vec<Cache> = (0..opts.batch)
                .map(|_| Cache::in_pools(&cfg_model, &pools))
                .collect::<lloom_tensor::Result<_>>()?;
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
                spawn_batch(Arc::clone(&engine.core), |c: &QwenCore<R>| &c.model, cfg)
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
        Ok(Box::new(engine))
    }

    /// Run `f` over a session on the held sequence, and put the sequence
    /// back whatever `f` returned.
    fn with_single<T>(
        &self,
        f: impl FnOnce(&mut DecodeSession<'_, Qwen4Exp<R>>) -> Result<T>,
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

    fn detok(&self) -> impl FnMut(&[u32]) -> lloom_tensor::Result<String> + '_ {
        |ids: &[u32]| -> lloom_tensor::Result<String> {
            self.text
                .tokenizer
                .decode(ids, false)
                .map_err(|e| lloom_tensor::Error::Msg(format!("detokenize: {e}")))
        }
    }

    fn check_fits(&self, prompt: usize, gen: usize) -> Result<()> {
        let need = prompt + gen;
        if need > self.kv_capacity {
            anyhow::bail!(
                "prompt {prompt} + generation {gen} = {need} positions exceeds the KV capacity \
                 {} (start with --kv-capacity >= {need})",
                self.kv_capacity
            );
        }
        Ok(())
    }

    /// The MTP drafter for one run: the head, the shared embedding and
    /// `lm_head`, a one-layer cache of its own. Cheap to build per request.
    fn drafter(&self) -> Result<MtpDrafter<'_, R>> {
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
        .with_k(self.mtp_k))
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
struct TakenSessions<'m, R: ExpertResidency + HasStreamStats> {
    sessions: Vec<Session<'m, Qwen4Exp<R>>>,
    slots: Vec<usize>,
}

impl<'m, R: ExpertResidency + HasStreamStats> TakenSessions<'m, R> {
    fn give_back(self, idle: &mut Idle<'_, 'm, Qwen4Exp<R>, ()>) {
        for (slot, mut s) in self.slots.into_iter().zip(self.sessions) {
            s.reset();
            idle.pool.release(slot, s.into_sequence());
        }
    }
}

/// Run `f` over `n` sequences checked out of the pool and reset, and put
/// them back — reset again — whatever `f` returned.
fn with_sessions<'m, R: ExpertResidency + HasStreamStats, T>(
    idle: &mut Idle<'_, 'm, Qwen4Exp<R>, ()>,
    n: usize,
    f: impl FnOnce(&mut Idle<'_, 'm, Qwen4Exp<R>, ()>, &mut Vec<Session<'m, Qwen4Exp<R>>>) -> Result<T>,
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

impl<R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Engine<R> {
    /// Run a closed job on the scheduler's thread, the rows idle.
    fn closed<T: Send + 'static>(
        &self,
        job: impl for<'a, 'm> FnOnce(&mut Idle<'a, 'm, Qwen4Exp<R>, ()>) -> T + Send + 'static,
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
        sched: &BatchHandle<Qwen4Exp<R>, ()>,
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
            readback: ReadbackArm::Batched {
                rows: done.joined_beside,
            },
            stats: done.stats,
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

impl<R: ExpertResidency + HasStreamStats + Send + Sync + 'static> Served for Engine<R> {
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
            "family=qwen4exp batch={} experts={} slots={} loader={} mtp={} ple={} ple_cache_rows={} moe_act={} config=[{}] prefill_tile={} mem_sample={} {}",
            self.batch,
            self.tier["experts"].as_str().unwrap_or("?"),
            self.tier["slots_per_layer"].as_u64().unwrap_or(0),
            self.tier["loader"].as_str().unwrap_or("-"),
            self.mtp,
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
        let detok = self.detok();
        self.with_single(|s| {
            if !reuse {
                s.reset();
            }
            if self.mtp {
                let mut drafter = self.drafter()?;
                s.decode_speculative_until(&mut drafter, prompt_ids, detok, params, |f| {
                    on_fragment(f)
                })
                .map_err(anyhow::Error::from)
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
        self.check_fits(prompt_tokens, decode_tokens + 1)?;
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
            if self.mtp {
                // One run of `D` tokens through the speculative body; the
                // prompt's submit time subtracted (the prefill's GPU tail is
                // inside the bracket), as the LFM2 engine brackets DSpark.
                let params = DecodeParams {
                    max_tokens: decode_tokens,
                    eos_ids: NEVER_EOS.to_vec(),
                    readback,
                    ..DecodeParams::default()
                };
                let mut drafter = self.drafter()?;
                super::common::profile_reset();
                let t = Instant::now();
                let run =
                    s.decode_speculative_until(&mut drafter, &prompt, detok, &params, |_| true)?;
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
            assert!(note.contains("host fill") && note.contains("misses"), "{note}");
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
