//! The loaded model and the operations every endpoint is a thin shape over.
//!
//! One model, one device-side sequence, one request at a time (the caller
//! holds the mutex). Every operation here is expressed over `lloom`'s tier-1
//! API and `lloom_engine::DecodeSession` — nothing reaches into a model's
//! internals (RESTRUCTURING §7.1b).

use anyhow::{Context, Result};
use hf_chat_template::ChatTemplate;
use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, Lfm2Config, Model};
use lloom::nn::{Sequence, VarBuilder};
use lloom_engine::{DecodeParams, DecodeSession, Decoded, ReadbackArm};
use lloom_tensor::{DType, Device, Tensor, D};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// The EOS id set that never fires: every arm's EOS mechanism still runs and
/// simply never matches, so a run produces exactly `max_tokens` tokens.
pub const NEVER_EOS: [u32; 1] = [u32::MAX];

/// The model, its cache and the device: the half a decode borrows.
pub struct Core {
    pub model: Model,
    pub device: Device,
    /// The one sequence every request drives, taken out for the request's
    /// session and put back after it.
    seq: Option<Sequence<Cache>>,
}

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
    pub core: Core,
    pub text: Text,
    pub model_name: String,
    pub kv_capacity: usize,
    pub readback: ReadbackArm,
    /// Reuse the cached prefix across requests (`Session::prepare`). Off by
    /// default: a benchmark cell measures a cold prefill, and a repeated prompt
    /// would otherwise resume from the previous request's state.
    pub reuse_prefix: bool,
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
}

/// What one throughput cell measured, in the sidecar's units.
pub struct PrefillTiming {
    pub prompt_tokens: usize,
    pub prefill_ms: f64,
}

pub struct DecodeTiming {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub readback: ReadbackArm,
}

impl Core {
    /// Run `f` over a session on the held sequence, and put the sequence back
    /// whatever `f` returned.
    pub fn with_session<T>(
        &mut self,
        f: impl FnOnce(&mut DecodeSession<'_, Model>) -> Result<T>,
    ) -> Result<T> {
        let seq = self
            .seq
            .take()
            .context("the sequence is checked out — a previous request panicked mid-decode")?;
        let mut session = DecodeSession::from_sequence(&self.model, seq, self.device.clone());
        let out = f(&mut session);
        self.seq = Some(session.into_sequence());
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

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let config: Lfm2Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
                .context("parsing config.json")?;
        let mut config = config.into_config(false);
        crown_config(&mut config);

        let files = weight_files(&model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        let model = Model::new(&config, vb).context("constructing LFM2")?;
        let cache = Cache::new_with(true, dtype, &config, &device, opts.kv_capacity)
            .context("allocating the KV cache")?;

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

        let mut engine = Engine {
            core: Core {
                model,
                device,
                seq: Some(Sequence::new(cache)),
            },
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
        };

        if opts.warmup {
            let (libs, pipes) = engine.core.with_session(|s| Ok(s.warmup()?))?;
            eprintln!("[warmup: {libs} libraries, {pipes} pipelines compiled]");
        }
        Ok(engine)
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

    // ---- decode ---------------------------------------------------------

    /// One decode over `prompt_ids` from a fresh sequence (unless prefix reuse
    /// is on), streaming text fragments to `on_fragment`.
    pub fn complete(
        &mut self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str),
    ) -> Result<Decoded> {
        self.complete_until(prompt_ids, params, |frag| {
            on_fragment(frag);
            true
        })
    }

    /// [`Self::complete`] whose hook can end the run early (`false`).
    pub fn complete_until(
        &mut self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: impl FnMut(&str) -> bool,
    ) -> Result<Decoded> {
        self.check_fits(prompt_ids.len(), params.max_tokens)?;
        let text = &self.text;
        let reuse = self.reuse_prefix;
        self.core.with_session(|s| {
            if !reuse {
                s.reset();
            }
            let detok = |ids: &[u32]| -> lloom_tensor::Result<String> {
                text.tokenizer
                    .decode(ids, false)
                    .map_err(|e| lloom_tensor::Error::Msg(format!("detokenize: {e}")))
            };
            s.decode_until(prompt_ids, detok, params, on_fragment)
                .map_err(anyhow::Error::from)
        })
    }

    /// Time one prefill of `n` seed tokens on a fresh cache: the forward
    /// submitted and the device drained, nothing else in the bracket.
    pub fn prefill_throughput(&mut self, n: usize) -> Result<PrefillTiming> {
        self.check_fits(n, 1)?;
        let ids = self.seed_prompt(n);
        let device = self.core.device.clone();
        let prefill = self.core.with_session(|s| {
            s.reset();
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let t = Instant::now();
            let _ = s.model().forward(&input, 0, s.cache_mut())?;
            device.synchronize()?;
            let elapsed = t.elapsed();
            // The forward wrote into the cache behind the session's token
            // log; nothing downstream may read that state.
            s.reset();
            Ok(elapsed)
        })?;
        Ok(PrefillTiming {
            prompt_tokens: n,
            prefill_ms: ms(prefill),
        })
    }

    /// Time `decode_tokens` decode steps after a `prompt_tokens`-deep context
    /// (`llama-bench --n-depth P --n-gen D`'s shape): the prompt is prefilled
    /// and its first token sampled outside the bracket, the device drained,
    /// then `D` tokens — `D` forwards and `D` samples — are timed to the last
    /// one landing. EOS never fires.
    pub fn decode_throughput(
        &mut self,
        prompt_tokens: usize,
        decode_tokens: usize,
    ) -> Result<DecodeTiming> {
        self.check_fits(prompt_tokens, decode_tokens + 1)?;
        let prompt = self.seed_prompt(prompt_tokens);
        let device = self.core.device.clone();
        let text = &self.text;
        let readback = self.readback;
        self.core.with_session(|s| {
            s.reset();
            let detok = |ids: &[u32]| -> lloom_tensor::Result<String> {
                text.tokenizer
                    .decode(ids, false)
                    .map_err(|e| lloom_tensor::Error::Msg(format!("detokenize: {e}")))
            };
            let params = DecodeParams {
                max_tokens: 1,
                eos_ids: NEVER_EOS.to_vec(),
                readback,
                ..DecodeParams::default()
            };
            let t = Instant::now();
            let first = s.decode(&prompt, detok, &params, |_| {})?;
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
                ..DecodeParams::default()
            };
            let t = Instant::now();
            let run = s.decode(&all, detok, &params, |_| {})?;
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
                prefill_ms,
                decode_ms,
                readback: run.readback,
            })
        })
    }

    /// The next-token log-probabilities after `prompt_ids`: the `k` most
    /// likely ids with their log-probs, and the argmax. One forward on a fresh
    /// cache; the logits come to the host once.
    pub fn next_token_logprobs(
        &mut self,
        prompt_ids: &[u32],
        k: usize,
    ) -> Result<(u32, Vec<(u32, f32)>)> {
        self.check_fits(prompt_ids.len(), 1)?;
        let device = self.core.device.clone();
        self.core.with_session(|s| {
            s.reset();
            let input = Tensor::new(prompt_ids, &device)?.unsqueeze(0)?;
            let logits = s.model().forward(&input, 0, s.cache_mut())?;
            s.reset();
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
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
        })
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
}

/// The crown decode configuration `lfm2-chat --cache crown` bakes in: the
/// `Sdpa × InPlace × Shuffle` arms and the GEMV/conv fusions, every one
/// token-identical to the generic path (§13.5, §12.2d, §12.2h). The library's
/// `Config` ships the conservative arms (§7.1a); a serving binary turns the
/// measured ones on.
fn crown_config(config: &mut Config) {
    config.attn_impl = AttnImpl::Sdpa;
    config.kv_append = KvAppend::InPlace;
    config.conv_state = ConvState::Shuffle;
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
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

/// The LFM2.5-2.6B snapshot in the HF cache, if it is there
/// (`LFM2_MODEL_DIR` first).
pub fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    entries.sort();
    entries.pop()
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

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
