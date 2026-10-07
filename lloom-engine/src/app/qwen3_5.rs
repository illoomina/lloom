//! The Qwen3.5 engine: one sequence over the dense (`qwen35`) or the
//! resident-MoE (`qwen35moe`) model on the generic decoder
//! (`lloom::models::qwen3_5`), driven by the same `DecodeSession` the other
//! families' engines drive. The `No × One` shapes only: no speculative body
//! (the MTP head is not built), no rows (`--batch N`), no streamed expert
//! tier — each refused by name in `family.rs`.

use super::common::{
    lock, model_name_from_dir, ms, newest_snapshot, open_device, DecodeTiming, MemGauge, MemReport,
    PrefillTiming, Text, NEVER_EOS,
};
use super::served::Served;
use super::template::{Rendered, Template};
use crate::{DecodeParams, DecodeSession, Decoded, ReadbackArm, RingPool};
use anyhow::{Context, Result};
use lloom::load_warning::LoadWarning;
use lloom::models::qwen3_5::{arch_of, Config, LoadOptions as ModelLoad, Qwen3_5, Qwen3_5Moe};
use lloom::nn::model::{CacheReset, CacheRewind, CausalLM};
use lloom::nn::Sequence;
use lloom::profile::{Profile, ProfileKey};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::{DType, Device, Tensor, D};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
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
}

/// The one sequence every request drives, and the rings kept across
/// requests.
struct Single<C> {
    seq: Option<Sequence<C>>,
    rings: Option<RingPool>,
    decode: crate::DecodeConfig,
}

/// The engine over a model `M` of this family (the dense or the MoE alias).
pub struct Engine<M: CausalLM<Config = Config>> {
    model: Arc<M>,
    device: Device,
    text: Text,
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
    single: Mutex<Single<M::Cache>>,
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

/// Detect, hold the regime, load.
pub fn load(opts: LoadOptions) -> Result<Box<dyn Served>> {
    use lloom::profile::{NoRewind, One, Rewind as _, Rows as _};
    let key = ProfileKey::at_contract(NoRewind::NAME, One::NAME);
    lloom::profile::dispatch!(key, lloom::profile::QWEN3_5, P => load_as::<P>(opts))
}

fn load_as<P: Profile>(opts: LoadOptions) -> Result<Box<dyn Served>> {
    let device = if opts.cpu {
        Device::Cpu
    } else {
        open_device()?
    };
    let shards = lloom::hub::gguf_shards(&opts.first_shard);
    let mut f = std::fs::File::open(&opts.first_shard)
        .with_context(|| format!("opening {}", opts.first_shard.display()))?;
    let content = Content::read(&mut f)
        .with_context(|| format!("reading the GGUF header of {}", opts.first_shard.display()))?;
    let arch = arch_of(&content).context("not a qwen35 / qwen35moe GGUF")?;
    drop(f);
    let installed = super::common::install_executor(&device, &opts.executor, false)?;
    #[cfg(feature = "metal")]
    if let Some(mb) = opts.memory.free_budget_mb {
        if let Device::Metal(m) = &device {
            m.set_free_budget(mb * 1024 * 1024);
        }
    }
    let mload = ModelLoad {
        kv_capacity: opts.kv_capacity,
        kv_precision: opts.kv_precision,
        memory_fraction: Some(opts.memory_fraction),
        dense_f32: false,
    };
    let t = Instant::now();
    let mut warnings = Vec::new();
    match arch {
        lloom::models::qwen3_5::ARCH_DENSE => {
            let model: Qwen3_5<P> = Qwen3_5::from_gguf_shards(&shards, &device, &mload)
                .context("loading Qwen3.5 (dense)")?;
            let tier = json!({ "arch": arch, "shards": shards.len(), "load_ms": ms(t.elapsed()) });
            Engine::finish(
                model, device, opts, &content, arch, tier, installed, warnings,
            )
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
            let model: Qwen3_5Moe<P> = Qwen3_5Moe::from_gguf_shards(&shards, &device, &mload)
                .context("loading Qwen3.5 (MoE, resident)")?;
            let tier = json!({ "arch": arch, "experts": "resident", "shards": shards.len(), "load_ms": ms(t.elapsed()) });
            Engine::finish(
                model, device, opts, &content, arch, tier, installed, warnings,
            )
        }
    }
}

impl<M> Engine<M>
where
    M: CausalLM<Config = Config> + Send + Sync + 'static,
    M::Cache: CacheReset + CacheRewind + Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn finish(
        model: M,
        device: Device,
        opts: LoadOptions,
        content: &Content,
        arch: &'static str,
        mut tier: Value,
        installed: super::report::Installed,
        load_warnings: Vec<LoadWarning>,
    ) -> Result<Box<dyn Served>> {
        let (tokenizer, template, text_source) = resolve_text(&opts, content)?;
        // The turn end and the checkpoint's own EOS (`generation_config.json`
        // lists both 248046 and 248044).
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
            .unwrap_or_else(|| match base_model_snapshot(content) {
                Some(dir) => model_name_from_dir(&dir),
                None => opts
                    .first_shard
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| arch.to_string()),
            });
        tier["tokenizer"] = json!(text_source);
        tier["first_shard"] = json!(opts.first_shard.display().to_string());
        let cache = model.new_cache(model_config(&model), &device)?;
        let mut engine = Engine {
            model: Arc::new(model),
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
            },
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
            single: Mutex::new(Single {
                seq: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                decode: opts.decode.clone(),
            }),
        };
        if opts.warmup {
            let t = Instant::now();
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
        let _ = &mut engine;
        Ok(Box::new(engine))
    }

    fn with_single<T>(&self, f: impl FnOnce(&mut DecodeSession<'_, M>) -> Result<T>) -> Result<T> {
        let mut single = lock(&self.single);
        let seq = single
            .seq
            .take()
            .context("the sequence is checked out — a previous request panicked mid-decode")?;
        let rings = single.rings.take().unwrap_or_default();
        let mut session = DecodeSession::from_parts(&*self.model, seq, self.device.clone(), rings)
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
}

/// The family config a decoder of either alias carries.
fn model_config<M: CausalLM<Config = Config>>(_m: &M) -> &'static Config {
    // `new_cache` ignores its config argument on a `Decoder` (the capacity
    // is the model's); a static placeholder keeps the call shape.
    static CFG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    CFG.get_or_init(Config::tiny)
}

impl<M> Served for Engine<M>
where
    M: CausalLM<Config = Config> + Send + Sync + 'static,
    M::Cache: CacheReset + CacheRewind + Send + Sync + 'static,
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
            "family=qwen3_5 arch={} batch=1 experts={} prefill_tile={} mem_sample={} {}",
            self.arch,
            self.tier["experts"].as_str().unwrap_or("-"),
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
    fn thinking_budget_refused(&self) -> Option<&'static str> {
        None
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
        self.with_single(|s| {
            if !reuse {
                s.reset();
            }
            s.decode_until(prompt_ids, detok, params, |f| on_fragment(f))
                .map_err(anyhow::Error::from)
        })
    }
    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        self.check_batch(batch)?;
        self.check_fits(n, 1)?;
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
        let prompt = self.seed_prompt_row(prompt_tokens, 0);
        let device = self.device.clone();
        let readback = self.readback;
        let prefill_tile = self.prefill_tile;
        let detok = self.detok();
        let gauge = MemGauge::start(&device, self.mem_sample);
        let mut timing = self.with_single(|s| {
            s.reset();
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
