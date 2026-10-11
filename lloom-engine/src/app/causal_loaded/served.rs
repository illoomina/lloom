//! Actual load-once `Served` caller over the same retained owner as decisions.
//! Prediction is an explicit caller coverage contract, not verified admission.
//! Default registration, enrolled host metadata and worker custody remain open.
use super::*;
use crate::app::common::{DecodeTiming, MemGauge, PrefillTiming};
use crate::app::family::Capabilities;
use crate::app::served::Served;
use crate::app::template::{Rendered, Template};
use lloom::load_warning::LoadWarning;
use lloom::nn::ops::KvQuant;
use lloom::profile::Profile;
use lloom_tensor::{DType, Tensor, D};
use serde_json::{json, Map, Value};
use std::time::Instant;

type Model = lloom::models::quantized_lfm2::Model;

/// Inputs available before allocating any request producer. The predictor must
/// cover host preparation, caches, retained frames, readout and async retention
/// through callback completion, not just a token tile. No default zero estimate.
pub enum Request<'a> {
    Encode {
        text: &'a str,
        add_special: bool,
    },
    Token(u32),
    Render {
        messages: &'a Value,
        kwargs: &'a Map<String, Value>,
        checked: &'a [&'a str],
    },
    Generation {
        prompt: &'a [u32],
        params: &'a DecodeParams,
    },
    Decisions(&'a OrderedJson),
    PreparedDecisions(&'a CausalRequest),
    Seed {
        positions: usize,
        row: usize,
    },
    Prefill {
        positions: usize,
    },
    Decode {
        prompt: usize,
        generated: usize,
    },
    Logprobs {
        prompt: &'a [u32],
        k: usize,
    },
}
pub trait Predictor: Send + Sync {
    fn predicted_extra(&self, request: Request<'_>) -> Result<usize>;
}

/// Serving facts are explicit checkpoint/caller inputs, not guessed siblings.
/// They and compiled template/seed are produced within the supplied load scope;
/// host tickets for these and predictor storage are not implicitly enrolled.
pub struct Config {
    pub model_name: String,
    pub bos_token: String,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    pub seed_text: String,
    pub chat_template: Option<String>,
}
struct Metadata {
    config: Config,
    device: Device,
    capacity: usize,
    decode: DecodeConfig,
    template: Option<Template>,
    seed: Vec<u32>,
    decisions: bool,
}
/// No model/tokenizer/quota copy and no borrowed unscoped Loaded escape.
pub struct Engine<P> {
    loaded: Loaded<CausalEngine<Model>>,
    metadata: Metadata,
    predictor: P,
    warnings: Vec<LoadWarning>,
}

/// Explicit family embedding factory returning a real HTTP/REPL implementor.
/// The same one load promise covers weights, initial cache and serving facts.
/// No default registry/CLI option is activated before built-in prediction and
/// producer enrollment. CPU/CUDA activation retains Loaded's existing guard.
pub struct Files<'a> {
    pub checkpoint: &'a Path,
    pub tokenizer: &'a Path,
}
pub fn load_lfm2_gguf<P: Predictor>(
    files: Files<'_>,
    policy: Option<CausalPolicySource>,
    device: &Device,
    load: LoadContext,
    decode: DecodeConfig,
    config: Config,
    predictor: P,
) -> Result<Engine<P>> {
    let mut metadata = None;
    let serving_decode = decode.clone();
    let loaded = Loaded::new(device, load, |device, _context| {
        let (engine, serving) = produce_lfm2_gguf_with(
            files.checkpoint,
            files.tokenizer,
            policy,
            device,
            decode,
            |resolved, tokenizer, policy| {
                if config.eos_ids.is_empty()
                    || config
                        .eos_ids
                        .iter()
                        .chain(config.bos_id.iter())
                        .any(|&id| id as usize >= resolved.vocab_size)
                {
                    lloom_tensor::bail!("serving BOS/EOS ids must fit the loaded vocabulary and EOS must be nonempty")
                }
                let template = config
                    .chat_template
                    .as_deref()
                    .map(Template::compile)
                    .transpose()
                    .map_err(|e| Error::msg(format!("{e:#}")))?;
                let seed = tokenizer
                    .encode(config.seed_text.as_str(), false)
                    .map_err(|e| Error::msg(e.to_string()))?
                    .get_ids()
                    .to_vec();
                if seed.is_empty() {
                    lloom_tensor::bail!("serving seed must encode at least one token")
                }
                Ok(Metadata {
                    config,
                    device: device.clone(),
                    capacity: resolved.context_length,
                    decode: serving_decode,
                    template,
                    seed,
                    decisions: policy.is_some(),
                })
            },
        )?;
        metadata = Some(serving);
        Ok(engine)
    })?;
    Ok(Engine {
        loaded,
        metadata: metadata.expect("successful producer installed serving facts"),
        predictor,
        warnings: Vec::new(),
    })
}

impl<P: Predictor> Engine<P> {
    /// Preserve explicit generic prepared decisions when named raw rendering is
    /// absent or uncertified. The adapter never exposes an unscoped inner owner.
    pub fn decide_prepared(&self, request: &CausalRequest) -> Result<SystemOneResponse> {
        let predicted = self
            .predictor
            .predicted_extra(Request::PreparedDecisions(request))?;
        CausalEngine::decide_admitted(&self.loaded, predicted, request)
    }
    fn pass<T>(
        &self,
        request: Request<'_>,
        run: impl FnOnce(&CausalEngine<Model>) -> Result<T>,
    ) -> Result<T> {
        let predicted = self.predictor.predicted_extra(request)?;
        self.loaded
            .with_pass(predicted, |engine, _context| run(engine))
    }
    pub fn seed_prompt_row_checked(&self, n: usize, row: usize) -> Result<Vec<u32>> {
        check_context_capacity(self.metadata.capacity, n, 0, 0).map_err(Error::wrap)?;
        self.pass(Request::Seed { positions: n, row }, |_| {
            Ok(self.seed(n, row))
        })
    }
    fn seed(&self, n: usize, row: usize) -> Vec<u32> {
        (0..n)
            .map(|i| self.metadata.seed[(i.wrapping_add(row)) % self.metadata.seed.len()])
            .collect()
    }
    fn benchmark<'a>(&self, engine: &'a CausalEngine<Model>) -> Result<RequestLease<'a, Model>> {
        let mut lease = engine.checkout()?;
        lease.start_decode(&engine.model, &engine.device, engine.decode.clone());
        lease.session().reset();
        Ok(lease)
    }
}
impl<P: Predictor> Served for Engine<P> {
    fn device(&self) -> &Device {
        &self.metadata.device
    }
    fn model_name(&self) -> &str {
        &self.metadata.config.model_name
    }
    fn kv_capacity(&self) -> usize {
        self.metadata.capacity
    }
    fn readback(&self) -> ReadbackArm {
        self.metadata.decode.readback.arm()
    }
    fn prefill_tile(&self) -> usize {
        self.metadata.decode.prefill.tile
    }
    fn batch(&self) -> usize {
        1
    }
    fn eos_ids(&self) -> &[u32] {
        &self.metadata.config.eos_ids
    }
    fn bos_id(&self) -> Option<u32> {
        self.metadata.config.bos_id
    }
    fn health(&self) -> Value {
        json!({"capacity": self.kv_capacity(), "capabilities": self.capabilities().json(), "admission": "caller-supplied coverage; producer enrollment incomplete"})
    }
    fn describe(&self) -> String {
        format!(
            "load-once GGUF causal owner; {}",
            self.capabilities().line()
        )
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            family: "lfm2-gguf",
            generation: true,
            decisions: self.metadata.decisions,
            profiles: lloom::profile::Bound {
                family: "lfm2-gguf",
                rewind: &["No"],
                rows: &["One"],
                except: &[lloom::profile::CrownExact::KEY],
            },
            lookahead_max: 1,
            ring: "by-name: zero rewind horizon",
            batch: false,
            prefix_share: false,
            speculation: &[],
            model_axes: false,
            rewind: "none",
            executor_arms: false,
            kv_quant: &[KvQuant::Off],
            kv_dtype: "f32",
            kv_dense: &["f32"],
            kv_quant_missing: "quantized GGUF weights do not install quantized KV stores",
            vision: false,
            vision_missing:
                "this text owner has no media encoder; other family media paths remain available",
            eos_gate: false,
            eos_gate_missing: "generic GGUF forward has no gated kernels",
        }
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        self.warnings.clone()
    }
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>) {
        self.warnings.splice(0..0, warnings);
    }
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let predicted = self
            .predictor
            .predicted_extra(Request::Decisions(body))
            .map_err(|e| SystemOneError::Execution(e.to_string()))?;
        CausalEngine::systemone_admitted(&self.loaded, predicted, body)
    }
    fn encode(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self.pass(Request::Encode { text, add_special }, |engine| {
            engine.encode(text, add_special)
        })?)
    }
    fn token_str(&self, id: u32) -> String {
        self.token_str_checked(id).expect("legacy infallible token display cannot report admission refusal; use token_str_checked")
    }
    fn token_str_checked(&self, id: u32) -> anyhow::Result<String> {
        Ok(self.pass(Request::Token(id), |engine| {
            if id as usize >= engine.model.vocab_size() {
                lloom_tensor::bail!("token display id outside loaded vocabulary")
            }
            engine
                .tokenizer
                .decode(&[id], false)
                .map_err(|e| Error::msg(e.to_string()))
        })?)
    }

    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> anyhow::Result<Rendered> {
        // Preserve the incumbent typed template errors (Rejected/Unrenderable)
        // inside a nested result, instead of flattening them into tensor errors.
        self.pass(
            Request::Render {
                messages,
                kwargs,
                checked,
            },
            |_| {
                Ok(match self.metadata.template.as_ref() {
                    Some(template) => template.render_checked(
                        messages,
                        &self.metadata.config.bos_token,
                        kwargs,
                        checked,
                    ),
                    None => Err(anyhow::anyhow!(
                        "checkpoint/caller supplied no chat template"
                    )),
                })
            },
        )?
    }
    fn template_reads(&self, name: &str) -> Option<bool> {
        self.metadata.template.as_ref()?.reads(name)
    }
    fn template_renders(&self) -> u64 {
        self.metadata.template.as_ref().map_or(0, Template::renders)
    }
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32> {
        self.seed_prompt_row_checked(n, row).expect(
            "legacy infallible seed cannot report admission refusal; use seed_prompt_row_checked",
        )
    }
    fn seed_prompt_row_checked(&self, n: usize, row: usize) -> anyhow::Result<Vec<u32>> {
        Ok(Engine::seed_prompt_row_checked(self, n, row)?)
    }

    fn complete_until(
        &self,
        prompt: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> anyhow::Result<Decoded> {
        Ok(self.pass(Request::Generation { prompt, params }, |engine| {
            engine.complete(prompt, params, &mut |text| on_fragment(text))
        })?)
    }
    fn prefill_throughput(&self, n: usize, batch: usize) -> anyhow::Result<PrefillTiming> {
        self.check_batch(batch)?;
        check_context_capacity(self.kv_capacity(), n, 1, 0)?;
        Ok(self.pass(Request::Prefill { positions: n }, |engine| {
            let prompt = self.seed(n, 0);
            let gauge = MemGauge::start(&engine.device, false);
            let mut lease = self.benchmark(engine)?;
            let params = DecodeParams {
                max_tokens: 1,
                eos_ids: NEVER_EOS.to_vec(),
                prefill_tile: self.prefill_tile(),
                ..DecodeParams::default()
            };
            let start = Instant::now();
            let out = lease.session().prefill(&prompt, &params)?;
            engine.device.synchronize()?;
            let prefill_ms = 1000. * start.elapsed().as_secs_f64();
            lease.session().reset();
            Ok(PrefillTiming {
                prompt_tokens: n,
                batch: 1,
                prefill_ms,
                forwards: out.units,
                mem: gauge.finish(&engine.device),
            })
        })?)
    }
    fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> anyhow::Result<DecodeTiming> {
        self.check_batch(batch)?;
        check_context_capacity(self.kv_capacity(), prompt_tokens, decode_tokens, 1)?;
        Ok(self.pass(
            Request::Decode {
                prompt: prompt_tokens,
                generated: decode_tokens,
            },
            |engine| {
                let mut prompt = self.seed(prompt_tokens, 0);
                let gauge = MemGauge::start(&engine.device, false);
                let mut lease = self.benchmark(engine)?;
                let mut params = DecodeParams {
                    max_tokens: 1,
                    eos_ids: NEVER_EOS.to_vec(),
                    readback: self.readback(),
                    prefill_tile: self.prefill_tile(),
                    ..DecodeParams::default()
                };
                let start = Instant::now();
                let first =
                    lease
                        .session()
                        .decode(&prompt, |_| Ok(String::new()), &params, |_| {})?;
                engine.device.synchronize()?;
                let prefill_ms = 1000. * start.elapsed().as_secs_f64();
                if first.tokens.len() != 1 {
                    lloom_tensor::bail!("context run did not produce one token")
                }
                prompt.extend_from_slice(&first.tokens);
                params.max_tokens = decode_tokens;
                let start = Instant::now();
                let run =
                    lease
                        .session()
                        .decode(&prompt, |_| Ok(String::new()), &params, |_| {})?;
                engine.device.synchronize()?;
                let decode_ms = 1000. * start.elapsed().as_secs_f64();
                if run.tokens.len() != decode_tokens || run.prefilled != 0 {
                    lloom_tensor::bail!("decode throughput did not resume the held context")
                }
                lease.session().reset();
                Ok(DecodeTiming {
                    prompt_tokens,
                    decode_tokens,
                    batch: 1,
                    prefill_ms,
                    decode_ms,
                    readback: run.readback,
                    bracket: "D steps after context, resumed from held logits",
                    row_tokens: vec![run.tokens.len()],
                    row_rounds: Vec::new(),
                    row_accepted: Vec::new(),
                    row_draft_ms: Vec::new(),
                    row_verify_ms: Vec::new(),
                    row_observe_ms: Vec::new(),
                    profile: None,
                    mem: gauge.finish(&engine.device),
                })
            },
        )?)
    }
    fn next_token_logprobs(
        &self,
        prompt: &[u32],
        k: usize,
    ) -> anyhow::Result<(u32, Vec<(u32, f32)>)> {
        check_context_capacity(self.kv_capacity(), prompt.len(), 1, 0)?;
        Ok(self.pass(Request::Logprobs { prompt, k }, |engine| {
            if prompt.is_empty()
                || k == 0
                || prompt
                    .iter()
                    .any(|&id| id as usize >= engine.model.vocab_size())
            {
                lloom_tensor::bail!("logprobs requires a valid nonempty prompt and positive k")
            }
            let _lease = engine.checkout()?;
            let mut cache = engine.model.new_cache()?;
            let ids = Tensor::new(prompt, &engine.device)?.unsqueeze(0)?;
            let logits = engine
                .model
                .forward(&ids, 0, &mut cache)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let values = lloom::nn::ops::log_softmax(&logits, D::Minus1)?.to_vec1::<f32>()?;
            let mut top: Vec<_> = values
                .into_iter()
                .enumerate()
                .map(|(i, value)| (i as u32, value))
                .collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let argmax = top
                .first()
                .ok_or_else(|| Error::msg("empty loaded vocabulary"))?
                .0;
            top.truncate(k);
            Ok((argmax, top))
        })?)
    }
    fn check_batch(&self, batch: usize) -> anyhow::Result<()> {
        if batch != 1 {
            anyhow::bail!("lfm2-gguf text owner serves exactly one sequence, requested {batch}")
        }
        Ok(())
    }
    fn shutdown(&self) {} // No owner worker threads are launched by this adapter.
}
