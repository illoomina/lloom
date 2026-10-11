//! App-owned load-once causal execution: named/raw decisions and ordinary
//! generation use the same model and supplied allocation namespace.
//!
//! Additive embedding factory, not default served activation. Header/tokenizer,
//! request metadata and async native producers still require complete enrollment
//! and aggregate prediction. Existing app families/legacy loaders stay available.
use super::common::{check_context_capacity, NEVER_EOS};
use crate::decision::{
    parse_systemone_questions, parse_systemone_state, CausalRequest, NoulAliases, OrderedJson,
    StateNullability, SystemOneError, SystemOneResponse, SystemOneTemplate, SystemOneTemplateKind,
    Temperature,
};
use crate::{DecodeConfig, DecodeParams, Decoded, ReadbackArm, RingPool};
use lloom::nn::branch::RetainState;
use lloom::nn::memory::{LoadContext, Loaded};
use lloom::nn::{CacheReset, CacheRewind, CausalLM, Sequence};
use lloom_tensor::{Device, Error, Result};
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer;

/// Explicit caller policy remains available for generic causal models. No
/// guessed systemone template, implicit calibration or unknown-source state
/// certificate is installed. Arbitrary sources retain normal template rendering
/// via SystemOneTemplate; only certified sources enter the shared-trunk renderer.
pub struct CausalPolicy {
    pub template: SystemOneTemplate,
    pub temperature: Temperature,
    pub max_options: usize,
    pub noul_aliases: NoulAliases,
    pub state_nullability: StateNullability,
}
/// Calibration/limits remain explicit even for a named checkpoint template.
/// Selecting a named source never invents its option temperature or API policy.
pub enum CausalPolicySource {
    Explicit(CausalPolicy),
    Named {
        temperature: Temperature,
        max_options: usize,
        noul_aliases: NoulAliases,
        state_nullability: StateNullability,
    },
}
use super::request_slot::{Lease as RequestLease, State as Chat};

/// One model, a held chat sequence and request-level executor serialization.
/// Decisions build one fresh cache, never clone or replace the chat sequence.
/// The generic generation driver owns reset/recovery for zero-horizon caches.
pub struct CausalEngine<M: CausalLM> {
    model: M,
    config: M::Config,
    device: Device,
    capacity: usize,
    tokenizer: Tokenizer,
    policy: Option<CausalPolicy>,
    decode: DecodeConfig,
    chat: Mutex<Option<Chat<M::Cache>>>,
}
impl<M: CausalLM> CausalEngine<M>
where
    M::Cache: CacheReset + CacheRewind,
{
    /// All producers run inside the existing supplied load scope. This permits
    /// another causal app family to reuse this owner without a second model
    /// budget or a decision-only model copy. CPU/CUDA enrollment is unchanged.
    pub fn load_admitted(
        device: &Device,
        load: LoadContext,
        capacity: usize,
        decode: DecodeConfig,
        produce: impl FnOnce(&Device) -> Result<(M, M::Config, Tokenizer, Option<CausalPolicy>)>,
    ) -> Result<Loaded<Self>> {
        if capacity == 0 {
            lloom_tensor::bail!("causal app context capacity must be positive")
        }
        Loaded::new(device, load, |device, _context| {
            let (model, config, tokenizer, policy) = produce(device)?;
            let cache = model.new_cache(&config, device)?;
            Ok(Self {
                model,
                config,
                device: device.clone(),
                capacity,
                tokenizer,
                policy,
                decode,
                chat: Mutex::new(Some(Chat {
                    sequence: Some(Sequence::new(cache)),
                    rings: Some(RingPool::new()),
                    extra: (),
                    needs_recovery: false,
                })),
            })
        })
    }

    fn checkout(&self) -> Result<RequestLease<'_, M>> {
        RequestLease::checkout(&self.chat, false)
    }

    /// Tokenization is a request producer and runs in that owner's supplied
    /// whole-pass promise. This is not an independent quota or byte predictor.
    pub fn encode_admitted(
        loaded: &Loaded<Self>,
        predicted_extra: usize,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<u32>> {
        loaded.with_pass(predicted_extra, |engine, _context| {
            engine.encode(text, add_special)
        })
    }
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special)
            .map(|encoded| encoded.get_ids().to_vec())
            .map_err(|error| Error::msg(format!("tokenizing causal input: {error}")))
    }

    /// Ordinary rendering remains available even when a template has no shared
    /// state certificate. A caller may supply its explicit token prefix through
    /// CausalRequest::new and then use decide_admitted; no implicit certificate
    /// or shared-trunk replay fallback is fabricated by this owner.
    pub fn render_systemone_admitted(
        loaded: &Loaded<Self>,
        predicted_extra: usize,
        id: &str,
        state: &OrderedJson,
        question: &crate::decision::Question<OrderedJson>,
        labels: &[String],
    ) -> Result<String> {
        loaded.with_pass(predicted_extra, |engine, _context| {
            let policy = engine.policy.as_ref().ok_or_else(|| {
                Error::msg("no named or explicit causal systemone policy was installed")
            })?;
            policy.template.render(
                id,
                state,
                question,
                labels,
                crate::decision::SystemOneMedia::Text,
            )
        })
    }

    /// Ordinary generation remains a first-class caller on the same weights.
    /// No decision rendering/readout is added to its token loop. Prompt/callback
    /// origin and complete asynchronous prediction remain caller obligations.
    pub fn complete_admitted(
        loaded: &Loaded<Self>,
        predicted_extra: usize,
        prompt: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        loaded.with_pass(predicted_extra, |engine, _context| {
            engine.complete(prompt, params, &mut on_fragment)
        })
    }
    fn complete(
        &self,
        prompt: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (impl FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        check_context_capacity(self.capacity, prompt.len(), params.max_tokens, 0)
            .map_err(Error::wrap)?;
        if prompt
            .iter()
            .any(|&id| id as usize >= self.model.vocab_size())
        {
            lloom_tensor::bail!("chat prompt token is outside loaded vocabulary")
        }
        let mut lease = self.checkout()?;
        lease.start_decode(&self.model, &self.device, self.decode.clone());
        let tokenizer = &self.tokenizer;
        lease.session().decode_until(
            prompt,
            |ids| {
                tokenizer
                    .decode(ids, false)
                    .map_err(|error| Error::msg(format!("detokenizing causal output: {error}")))
            },
            params,
            |fragment| on_fragment(fragment),
        )
    }

    /// Fully prepared programmatic requests preserve explicit renderer/certificate
    /// paths for generic models independently of named checkpoint policy.
    pub fn decide_admitted(
        loaded: &Loaded<Self>,
        predicted_extra: usize,
        request: &CausalRequest,
    ) -> Result<SystemOneResponse>
    where
        M::Cache: RetainState,
    {
        loaded.with_pass(predicted_extra, |engine, _context| engine.decide(request))
    }
    fn decide(&self, request: &CausalRequest) -> Result<SystemOneResponse>
    where
        M::Cache: RetainState,
    {
        check_context_capacity(self.capacity, request.max_prompt_tokens(), 0, 0)
            .map_err(Error::wrap)?;
        // Serialize at request level only; no lock is taken in a decode step.
        let _lease = self.checkout()?;
        let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, &self.device)?;
        let options = self.decode.options(0, ReadbackArm::Blocking, 0);
        request.execute(&self.model, &self.config, &mut readback, &options)
    }

    /// Raw ordered endpoint/programmatic preparation and execution belong to one
    /// whole request promise. Loaded HTTP/CLI adapters still need their predictor
    /// and producer enrollment before default registration; this is not activation.
    pub fn systemone_admitted(
        loaded: &Loaded<Self>,
        predicted_extra: usize,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        M::Cache: RetainState,
    {
        loaded
            .with_pass(predicted_extra, |engine, _context| {
                Ok(engine.systemone(body))
            })
            .map_err(|error| SystemOneError::Execution(error.to_string()))?
    }
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        M::Cache: RetainState,
    {
        let policy = self.policy.as_ref().ok_or_else(|| {
            SystemOneError::Unsupported(
                "no named or explicit causal systemone policy was installed".into(),
            )
        })?;
        let invalid = |error: Error| SystemOneError::InvalidRequest(error.to_string());
        let state = parse_systemone_state(body, policy.state_nullability).map_err(invalid)?;
        if !state.media.is_empty() {
            return Err(SystemOneError::Unsupported(
                "this text causal owner has no installed media encoder; existing media APIs remain separate".into()));
        }
        let questions = parse_systemone_questions(body, policy.max_options, policy.noul_aliases)
            .map_err(invalid)?
            .into_iter()
            .map(|(id, question)| (id, question, policy.temperature))
            .collect();
        let request = CausalRequest::render(
            &policy.template,
            state.state.as_ref(),
            questions,
            self.capacity,
            |label| self.encode(label, false),
            |prompt| self.encode(prompt, false),
        )
        .map_err(invalid)?;
        self.decide(&request)
            .map_err(|error| SystemOneError::Execution(error.to_string()))
    }
}

/// Actual app GGUF LFM2 factory, not a second model-only Loaded wrapper.
/// Tokenizer is explicit (same checkpoint/caller), avoiding a guessed sibling or
/// downloaded vocabulary. Named checkpoint policy or explicit generic policy is
/// optional: ordinary generation and prepared decisions remain available without
/// it. All header/model/tokenizer/initial-chat producers run inside one load scope.
pub fn load_lfm2_gguf_admitted(
    checkpoint: &Path,
    tokenizer_path: &Path,
    policy_source: Option<CausalPolicySource>,
    device: &Device,
    load: LoadContext,
    decode: DecodeConfig,
) -> Result<Loaded<CausalEngine<lloom::models::quantized_lfm2::Model>>> {
    Loaded::new(device, load, |device, _context| {
        produce_lfm2_gguf(checkpoint, tokenizer_path, policy_source, device, decode)
    })
}

fn produce_lfm2_gguf(
    checkpoint: &Path,
    tokenizer_path: &Path,
    policy_source: Option<CausalPolicySource>,
    device: &Device,
    decode: DecodeConfig,
) -> Result<CausalEngine<lloom::models::quantized_lfm2::Model>> {
    produce_lfm2_gguf_with(
        checkpoint,
        tokenizer_path,
        policy_source,
        device,
        decode,
        |_, _, _| Ok(()),
    )
    .map(|(engine, ())| engine)
}

fn produce_lfm2_gguf_with<T>(
    checkpoint: &Path,
    tokenizer_path: &Path,
    policy_source: Option<CausalPolicySource>,
    device: &Device,
    decode: DecodeConfig,
    preflight: impl FnOnce(
        &lloom::models::quantized_lfm2::Config,
        &Tokenizer,
        Option<&CausalPolicy>,
    ) -> Result<T>,
) -> Result<(CausalEngine<lloom::models::quantized_lfm2::Model>, T)> {
    use lloom_tensor::quantized::gguf_file::Content;
    let mut file = std::fs::File::open(checkpoint)?;
    let content = Content::read(&mut file)?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .ok_or_else(|| Error::msg("GGUF is missing general.architecture"))?
        .to_string()?;
    if architecture != "lfm2" {
        lloom_tensor::bail!("causal LFM2 factory received {architecture}")
    }
    let plan = lloom::models::quantized_lfm2::Model::preflight_gguf_content(&content, &mut file)?;
    let config = plan.config();
    let capacity = config.context_length;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| Error::msg(format!("loading causal tokenizer: {error}")))?;
    let policy = match policy_source {
        Some(CausalPolicySource::Explicit(policy)) => Some(policy),
        Some(CausalPolicySource::Named {
            temperature,
            max_options,
            noul_aliases,
            state_nullability,
        }) => {
            let source = content
                .metadata
                .get("tokenizer.chat_template.systemone")
                .ok_or_else(|| Error::msg("checkpoint has no named systemone template"))?;
            Some(CausalPolicy {
                template: SystemOneTemplate::compile(
                    source.to_string()?,
                    SystemOneTemplateKind::D1Causal,
                )?,
                temperature,
                max_options,
                noul_aliases,
                state_nullability,
            })
        }
        None => None,
    };
    // Validate actual tokenizer ids, including sparse/added ids. A count
    // alone is not proof that ids fit the model's output/embedding rows.
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&id| id as usize >= config.vocab_size)
    {
        lloom_tensor::bail!("tokenizer id is outside resolved GGUF vocabulary")
    }
    let prepared = preflight(&config, &tokenizer, policy.as_ref())?;
    let model = plan.materialize(&mut file, device)?;
    let cache = model.new_cache()?;
    Ok((
        CausalEngine {
            model,
            config,
            device: device.clone(),
            capacity,
            tokenizer,
            policy,
            decode,
            chat: Mutex::new(Some(Chat {
                sequence: Some(Sequence::new(cache)),
                rings: Some(RingPool::new()),
                extra: (),
                needs_recovery: false,
            })),
        },
        prepared,
    ))
}

/// Owning adapter for the actual HTTP/REPL Served seam, with caller-supplied
/// complete prediction. Default family registration remains separate.
pub mod served;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod request_slot_tests;
