//! App-owned load-once causal execution: named/raw decisions and ordinary
//! generation use the same model and supplied allocation namespace.
//!
//! Additive embedding factory, not default served activation. Header/tokenizer,
//! request metadata and async native producers still require complete enrollment
//! and aggregate prediction. Existing app families/legacy loaders stay available.
use super::common::{check_context_capacity, NEVER_EOS};
use crate::decision::binding::{self, Calibration, CheckpointDecision, Selection};
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

/// A template binding and its request limits. A native checkpoint's binding
/// carries its own calibration (`CausalPolicySource::Named`); an explicit
/// caller policy states one. Arbitrary sources retain normal template
/// rendering; only certified sources enter the shared-trunk renderer.
pub struct CausalPolicy {
    pub template: SystemOneTemplate,
    pub calibration: Calibration,
    pub max_options: usize,
    pub noul_aliases: NoulAliases,
    pub state_nullability: StateNullability,
}
impl CausalPolicy {
    /// The checkpoint's own binding, when it is a native causal decision model:
    /// a GGUF file's metadata, or a safetensors directory's `config.json` with
    /// the engine-shipped template. `Ok(None)` for a plain checkpoint.
    pub fn native(
        checkpoint: &Path,
        max_options: usize,
        noul_aliases: NoulAliases,
        state_nullability: StateNullability,
    ) -> Result<Option<Self>> {
        let facts = if checkpoint.is_dir() {
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(checkpoint.join("config.json"))?)
                    .map_err(Error::wrap)?;
            CheckpointDecision::from_config(&config)?
        } else {
            let mut file = std::fs::File::open(checkpoint)?;
            let content = lloom_tensor::quantized::gguf_file::Content::read(&mut file)?;
            CheckpointDecision::from_gguf(&content.metadata)?
        };
        Ok(
            match binding::resolve(&Selection::Auto, &facts, SystemOneTemplateKind::D1Causal)? {
                Some(binding::Binding::Template(binding)) => Some(Self {
                    template: binding.template,
                    calibration: binding.calibration,
                    max_options,
                    noul_aliases,
                    state_nullability,
                }),
                _ => None,
            },
        )
    }

    /// Shared ordered text preparation. Media execution remains a distinct family
    /// integration; no encoded payload is decoded or silently replaced here.
    pub(crate) fn prepare_text(
        &self,
        body: &OrderedJson,
        capacity: usize,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
    ) -> std::result::Result<CausalRequest, SystemOneError> {
        let invalid = |error: Error| SystemOneError::InvalidRequest(error.to_string());
        let state = parse_systemone_state(body, self.state_nullability).map_err(invalid)?;
        if !state.media.is_empty() {
            return Err(SystemOneError::Unsupported(
                "this text causal owner has no installed media encoder; existing media APIs remain separate".into()));
        }
        self.prepare_rendered(
            state.state.as_ref(),
            self.parse_questions(body)?,
            capacity,
            crate::decision::SystemOneMedia::Text,
            |text| encode(text),
            |text| encode(text),
        )
    }

    /// One ordered question/readout policy for text and family-owned media.
    /// The family supplies the already filtered state and whole prompt encoder.
    pub(crate) fn parse_questions(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<
        Vec<(String, crate::decision::Question<OrderedJson>, Temperature)>,
        SystemOneError,
    > {
        parse_systemone_questions(body, self.max_options, self.noul_aliases)
            .map(|questions| {
                questions
                    .into_iter()
                    .map(|(id, question)| {
                        let temperature = self.calibration.temperature(&question);
                        (id, question, temperature)
                    })
                    .collect()
            })
            .map_err(|error| SystemOneError::InvalidRequest(error.to_string()))
    }

    pub(crate) fn prepare_rendered(
        &self,
        state: &OrderedJson,
        questions: Vec<(String, crate::decision::Question<OrderedJson>, Temperature)>,
        capacity: usize,
        media: crate::decision::SystemOneMedia<'_>,
        encode_label: impl FnMut(&str) -> Result<Vec<u32>>,
        encode_prompt: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> std::result::Result<CausalRequest, SystemOneError> {
        let invalid = |error: Error| SystemOneError::InvalidRequest(error.to_string());
        CausalRequest::render_with_media(
            &self.template,
            state,
            questions,
            capacity,
            media,
            encode_label,
            encode_prompt,
        )
        .map_err(invalid)
    }
}
/// A causal decision template bound at load: the checkpoint's own (a native
/// d1 checkpoint) or the certified one `decision.template` selects, with
/// llama.cpp's `lfm2-d1` request limits (255 options, `true`/`false` noul
/// keys, a null state allowed), and what the load line and `/v1/models` say
/// of it. `generic` is the front end's strategy, so nothing is bound for it.
pub struct BoundPolicy {
    pub policy: CausalPolicy,
    pub native: Option<binding::DecisionType>,
    pub origin: String,
}
impl BoundPolicy {
    pub fn resolve(facts: &CheckpointDecision, selection: Option<&str>) -> Result<Option<Self>> {
        let selection = Selection::parse(selection)?;
        let resolved = binding::resolve(&selection, facts, SystemOneTemplateKind::D1Causal)
            .map_err(|error| Error::msg(format!("decision.template: {error}")))?;
        Ok(match resolved {
            Some(binding::Binding::Template(binding))
                if !binding.template.has_causal_certificate() =>
            {
                lloom_tensor::bail!(
                    "decision.template {}: the shared-state renderer reads only the audited \
                     lfm2-d1 template (its state is rendered before any question), and this one \
                     is not byte-identical to it",
                    binding.origin
                )
            }
            Some(binding::Binding::Template(binding)) => Some(Self {
                policy: CausalPolicy {
                    template: binding.template,
                    calibration: binding.calibration,
                    max_options: 255,
                    noul_aliases: NoulAliases::TrueFalseOnly,
                    state_nullability: StateNullability::AllowNull,
                },
                native: binding.native,
                origin: binding.origin,
            }),
            Some(binding::Binding::Generic) | None => None,
        })
    }
}

/// `Named` binds the checkpoint's own decision template and calibration
/// (`decision::binding`): the GGUF must be a native decision checkpoint.
/// Request limits stay explicit; no API policy is invented.
pub enum CausalPolicySource {
    Explicit(CausalPolicy),
    Named {
        max_options: usize,
        noul_aliases: NoulAliases,
        state_nullability: StateNullability,
    },
}
use super::request_slot::{Lease as RequestLease, State as Chat};

/// Explicit family-owned recovery boundary before marked model work is reused.
/// The family covers ALL of its retained scratch/queues/host producers; Device
/// identity alone is not that proof. A refusal or unwind leaves full Held custody
/// and NeedsRecovery intact. Hooks run at request checkout, never in Lease Drop.
/// Held decision-only work preserves the marker until an actual decoding reset,
/// so the hook must tolerate repeated successful calls before that reset.
/// Hook storage has its caller origin; prediction/enrollment remain obligations.
pub trait Recovery<M: CausalLM>: Send + Sync {
    fn before_reuse(&self, model: &M, device: &Device) -> Result<()>;
}

/// Explicit opt-in for a family whose pending work uses this Device's registered
/// Commands. This excludes arbitrary custom queues, raw submissions and IO.
/// Prefix DATA readiness is not callback/shutdown or per-binding custody proof.
pub struct AssociatedDeviceRecovery;
impl<M: CausalLM> Recovery<M> for AssociatedDeviceRecovery {
    fn before_reuse(&self, _model: &M, device: &Device) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = device {
            let marker = device.flush_marker()?;
            return device.wait_marker_prefix_data(&marker);
        }
        device.synchronize()
    }
}

/// One model, a held chat sequence and request-level executor serialization.
/// Decisions build one fresh cache, never clone or replace the chat sequence.
/// Sequence prepare handles normal zero-horizon reuse. Incomplete request
/// recovery requires the explicit family strategy; legacy callers retain custody
/// but do not acquire a native/reset guarantee from this owner.
pub struct CausalEngine<M: CausalLM> {
    model: M,
    config: M::Config,
    device: Device,
    capacity: usize,
    tokenizer: Tokenizer,
    policy: Option<CausalPolicy>,
    decode: DecodeConfig,
    // None retains the explicitly caller-managed legacy generic contract.
    recovery: Option<Box<dyn Recovery<M>>>,
    /// The LFM2-VL tower an mmproj supplied, for image decisions.
    vision: Option<MmprojVision>,
    /// The `decision-ragged` arm, installed by a family whose model batches
    /// ragged rows (`install_ragged`); taken under `fast` only.
    ragged: Option<Ragged<M>>,
    chat: Mutex<Option<Chat<M::Cache>>>,
}

/// The ragged decision read (`CausalRequest::execute_ragged`) for this
/// owner's model, and the state the last read kept. Function pointers, as
/// the served adapter's decision dispatch: the generic owner does not take
/// the bounds the read needs.
struct Ragged<M: CausalLM> {
    text: RaggedText<M>,
    images: RaggedImages<M>,
    report: fn(&CausalEngine<M>) -> serde_json::Value,
    /// A `TrunkCache` of the model's retained state.
    trunks: Box<dyn std::any::Any + Send + Sync>,
}
type RaggedText<M> = fn(
    &CausalEngine<M>,
    &CausalRequest,
    &mut lloom::generation::Blocking,
    &lloom::nn::DecodeOptions,
) -> Result<SystemOneResponse>;
type RaggedImages<M> = for<'a> fn(
    &CausalEngine<M>,
    &super::vision::ImageRows<'a, M>,
    &CausalRequest,
    &mut lloom::generation::Blocking,
    &lloom::nn::DecodeOptions,
) -> Result<SystemOneResponse>;

impl<M> CausalEngine<M>
where
    M: lloom::nn::BatchedCausalLM + 'static,
    M::Cache: lloom::nn::branch::AdoptRetained,
    <M::Cache as RetainState>::Retained: Send + Sync + 'static,
{
    /// Take the `decision-ragged` arm (`scheduler.decision_ragged`).
    pub(crate) fn install_ragged(&mut self) {
        self.ragged = Some(Ragged {
            text: Self::ragged_text,
            images: Self::ragged_images,
            report: |engine| {
                engine
                    .trunks()
                    .map_or(serde_json::Value::Null, crate::decision::TrunkCache::report)
            },
            trunks: Box::new(crate::decision::TrunkCache::<
                <M::Cache as RetainState>::Retained,
            >::default()),
        });
    }

    fn trunks(&self) -> Option<&crate::decision::TrunkCache<<M::Cache as RetainState>::Retained>> {
        self.ragged.as_ref()?.trunks.downcast_ref()
    }

    fn ragged_text(
        &self,
        request: &CausalRequest,
        readback: &mut lloom::generation::Blocking,
        options: &lloom::nn::DecodeOptions,
    ) -> Result<SystemOneResponse> {
        request.execute_ragged(
            &self.model,
            &self.model,
            readback,
            options,
            |model, device| model.new_cache(&self.config, device),
            self.trunks(),
            true,
        )
    }

    /// A trunk with image rows is counted, not kept: its tokens do not name
    /// its pixels.
    fn ragged_images(
        &self,
        rows: &super::vision::ImageRows<'_, M>,
        request: &CausalRequest,
        readback: &mut lloom::generation::Blocking,
        options: &lloom::nn::DecodeOptions,
    ) -> Result<SystemOneResponse> {
        request.execute_ragged(
            rows,
            &self.model,
            readback,
            options,
            |model, device| model.new_cache(&self.config, device),
            self.trunks(),
            false,
        )
    }
}

impl<M: CausalLM> CausalEngine<M> {
    /// The ragged arm's engagement for `/health`; `None` when not installed.
    pub(crate) fn ragged_report(&self) -> Option<serde_json::Value> {
        self.ragged.as_ref().map(|ragged| (ragged.report)(self))
    }

    /// The ragged arm when it is installed and the contract is `fast`.
    fn ragged_arm(&self) -> Option<&Ragged<M>> {
        self.ragged
            .as_ref()
            .filter(|_| lloom_tensor::tuning::contract().is_fast())
    }
}

/// An mmproj's LFM2-VL tower and projector, loaded with the model: what a
/// decision request's images are encoded with before their rows splice into
/// the shared trunk (`lfm2::decision_media`).
pub struct MmprojVision {
    encoder: lloom::models::lfm2_vl::VisionEncoder,
    config: lloom::models::lfm2_vl::Lfm2VlConfig,
    processor: lloom::models::lfm2_vl::ImageProcessor,
    dtype: lloom_tensor::DType,
}
impl MmprojVision {
    /// Load `mmproj` for a text model of width `text_hidden` whose tokenizer
    /// names `<image>`. The image preprocessing is LFM2-VL's
    /// (`ImageProcessor::lfm2_vl_3b`, d1-3B's `processor_config.json`): the
    /// mmproj does not carry it.
    pub fn load(
        mmproj: &Path,
        tokenizer: &Tokenizer,
        text_hidden: usize,
        device: &Device,
    ) -> Result<Self> {
        use lloom_tensor::quantized::gguf_file::Content;
        let image_token_id = tokenizer.token_to_id("<image>").ok_or_else(|| {
            Error::msg("the tokenizer has no <image> token for the mmproj's image rows")
        })?;
        let mut file = std::fs::File::open(mmproj)?;
        let content = Content::read(&mut file)?;
        let config = lloom::models::lfm2_vl::Lfm2VlConfig::from_mmproj(&content, image_token_id)?;
        let dtype = if device.is_cpu() {
            lloom_tensor::DType::F32
        } else {
            lloom_tensor::DType::F16
        };
        let encoder = lloom::models::lfm2_vl::VisionEncoder::from_mmproj(
            &content,
            &mut file,
            &config,
            text_hidden,
            dtype,
            device,
        )?;
        Ok(Self {
            encoder,
            config,
            processor: lloom::models::lfm2_vl::ImageProcessor::lfm2_vl_3b(),
            dtype,
        })
    }
}
impl<M: CausalLM> CausalEngine<M>
where
    M::Cache: CacheReset + CacheRewind,
{
    /// All producers run inside the existing supplied load scope. This permits
    /// another causal app family to reuse this owner without a second model
    /// budget or a decision-only model copy. CPU/CUDA enrollment is unchanged.
    /// This original entry retains caller-managed recovery: it cannot infer
    /// arbitrary M scratch/topology. Families install their explicit strategy
    /// with load_admitted_with_recovery; default activation remains pending.
    pub fn load_admitted(
        device: &Device,
        load: LoadContext,
        capacity: usize,
        decode: DecodeConfig,
        produce: impl FnOnce(&Device) -> Result<(M, M::Config, Tokenizer, Option<CausalPolicy>)>,
    ) -> Result<Loaded<Self>> {
        Self::load_admitted_inner(device, load, capacity, decode, None, produce)
    }

    /// Additive family-aware constructor. Do not use the associated-device
    /// strategy for a custom model unless its complete producer topology fits.
    pub fn load_admitted_with_recovery(
        device: &Device,
        load: LoadContext,
        capacity: usize,
        decode: DecodeConfig,
        recovery: Box<dyn Recovery<M>>,
        produce: impl FnOnce(&Device) -> Result<(M, M::Config, Tokenizer, Option<CausalPolicy>)>,
    ) -> Result<Loaded<Self>> {
        Self::load_admitted_inner(device, load, capacity, decode, Some(recovery), produce)
    }

    fn load_admitted_inner(
        device: &Device,
        load: LoadContext,
        capacity: usize,
        decode: DecodeConfig,
        recovery: Option<Box<dyn Recovery<M>>>,
        produce: impl FnOnce(&Device) -> Result<(M, M::Config, Tokenizer, Option<CausalPolicy>)>,
    ) -> Result<Loaded<Self>> {
        lloom_tensor::memory_admission::begin_model_load();
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
                recovery,
                vision: None,
                ragged: None,
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
        let lease = RequestLease::checkout(&self.chat, self.recovery.is_some())?;
        if let Some(recovery) = &self.recovery {
            lease.before_recovery(|| recovery.before_reuse(&self.model, &self.device))?;
        }
        Ok(lease)
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
            engine.render_systemone(id, state, question, labels)
        })
    }
    fn render_systemone(
        &self,
        id: &str,
        state: &OrderedJson,
        question: &crate::decision::Question<OrderedJson>,
        labels: &[String],
    ) -> Result<String> {
        let policy = self.policy.as_ref().ok_or_else(|| {
            Error::msg("no named or explicit causal systemone policy was installed")
        })?;
        policy.template.render(
            id,
            state,
            question,
            labels,
            crate::decision::SystemOneMedia::Text,
        )
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
        // A text prompt must not reach the image positions a previous image
        // request left in the held sequence.
        if let Some(vision) = &self.vision {
            let n = prompt
                .iter()
                .filter(|&&id| id == vision.config.image_token_id)
                .count();
            if n > 0 {
                lloom_tensor::bail!(
                    "the prompt holds {n} image position(s) (`<image>`) and no image is attached"
                )
            }
        }
        let mut lease = self.checkout()?;
        lease.start_decode(&self.model, &self.device, self.decode.clone());
        let tokenizer = &self.tokenizer;
        let result = lease.session().decode_until(
            prompt,
            |ids| {
                tokenizer
                    .decode(ids, false)
                    .map_err(|error| Error::msg(format!("detokenizing causal output: {error}")))
            },
            params,
            |fragment| on_fragment(fragment),
        );
        if result.is_ok() {
            lease.mark_complete();
        }
        result
    }

    /// The expansion of one image's `<image>` through the mmproj tower's
    /// processor: its layout's delimiters and token runs.
    pub(crate) fn image_placeholder(&self, image: &super::vision::DecodedImage) -> Result<String> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::msg("this causal owner has no vision tower"))?;
        let layout = vision.processor.layout(image.height, image.width);
        Ok(vision
            .processor
            .placeholder(&layout, vision.config.use_image_special_tokens))
    }

    /// Chat whose prompt's image positions take `images`' rows, in order:
    /// every image encoded by the mmproj tower, then the whole prompt
    /// prefilled through [`super::vision::ImageRows`] on the held sequence
    /// (reset first: a kept prefix would hold another request's image rows)
    /// and decoded from there, as the safetensors family does.
    pub(crate) fn complete_images(
        &self,
        prompt: &[u32],
        images: &[super::vision::DecodedImage],
        params: &DecodeParams,
        on_fragment: &mut (impl FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        if images.is_empty() {
            return self.complete(prompt, params, on_fragment);
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::msg("this causal owner has no vision tower"))?;
        check_context_capacity(self.capacity, prompt.len(), params.max_tokens, 0)
            .map_err(Error::wrap)?;
        if prompt
            .iter()
            .any(|&id| id as usize >= self.model.vocab_size())
        {
            lloom_tensor::bail!("chat prompt token is outside loaded vocabulary")
        }
        let image_token = vision.config.image_token_id;
        let total = images.iter().try_fold(0usize, |sum, image| {
            let layout = vision.processor.layout(image.height, image.width);
            sum.checked_add(vision.processor.image_tokens(&layout))
        });
        let total = total.ok_or_else(|| Error::msg("image position count overflow"))?;
        let reserved = prompt.iter().filter(|&&id| id == image_token).count();
        if reserved != total {
            lloom_tensor::bail!(
                "the prompt reserves {reserved} image position(s) and its {} image(s) take {total}",
                images.len()
            )
        }
        let mut lease = self.checkout()?;
        lease.start_external_work();
        let mut parts = Vec::with_capacity(images.len());
        for image in images {
            let prepared = vision.processor.prepare_rgb8(
                &image.rgb,
                image.height,
                image.width,
                &self.device,
                vision.dtype,
            )?;
            parts.push(vision.encoder.encode_image(&prepared)?);
        }
        let features = if parts.len() == 1 {
            parts.pop().expect("one encoded image")
        } else {
            lloom_tensor::Tensor::cat(&parts, 0)?
        };
        let rows = super::vision::ImageRows::try_new(
            &self.model,
            prompt.to_vec(),
            image_token,
            features,
            0,
        )?;
        let tokenizer = &self.tokenizer;
        let decoded =
            lease.with_model(&rows, &self.device, self.decode.clone(), |session, _| {
                session.reset();
                session.decode_until(
                    prompt,
                    |ids| {
                        tokenizer.decode(ids, false).map_err(|error| {
                            Error::msg(format!("detokenizing causal output: {error}"))
                        })
                    },
                    params,
                    |fragment| on_fragment(fragment),
                )
            })?;
        if rows.spliced() != total {
            lloom_tensor::bail!(
                "image prefill placed {} image rows; the prompt holds {total}",
                rows.spliced()
            )
        }
        lease.mark_complete();
        Ok(decoded)
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
        let mut lease = self.checkout()?;
        lease.start_external_work();
        let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, &self.device)?;
        let options = self.decode.options(0, ReadbackArm::Blocking, 0);
        let result = match self.ragged_arm() {
            Some(ragged) => (ragged.text)(self, request, &mut readback, &options),
            None => request.execute(&self.model, &self.config, &mut readback, &options),
        };
        if result.is_ok() {
            lease.mark_complete();
        }
        result
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
        let prepared = self.prepare_systemone(body)?;
        self.decide_prepared(prepared)
    }

    /// [`Self::systemone`] for requests that arrived together
    /// (`lfm2::decision_media::decide_together`).
    fn systemone_batch(
        &self,
        bodies: &[&OrderedJson],
    ) -> Vec<std::result::Result<SystemOneResponse, SystemOneError>>
    where
        M::Cache: RetainState,
    {
        super::lfm2::decision_media::decide_together(
            bodies,
            lloom_tensor::tuning::contract().is_fast(),
            |body| self.prepare_systemone(body),
            |request| self.decide(request),
            |prepared| self.decide_prepared(prepared),
        )
    }

    fn prepare_systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<super::lfm2::decision_media::Prepared, SystemOneError> {
        let policy = self.policy.as_ref().ok_or_else(|| {
            SystemOneError::Unsupported(
                "no named or explicit causal systemone policy was installed".into(),
            )
        })?;
        use super::lfm2::decision_media::{prepare_images, ImageVision};
        let vision = self.vision.as_ref().map(|vision| ImageVision {
            processor: &vision.processor,
            config: &vision.config,
            dtype: vision.dtype,
        });
        prepare_images(
            body,
            policy,
            self.capacity,
            self.model.vocab_size(),
            vision,
            super::lfm2::SERVED_DECISION_IMAGES,
            || {
                "this causal owner has no vision tower: load it with the checkpoint's mmproj \
                 (model.mmproj), and without model.text_only, to answer image decisions"
                    .into()
            },
            |text| Ok(self.encode(text, false)?),
        )
    }

    fn decide_prepared(
        &self,
        prepared: super::lfm2::decision_media::Prepared,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        M::Cache: RetainState,
    {
        use super::lfm2::decision_media::{decide_images, ImageVision, Prepared};
        let execution = |error: Error| SystemOneError::Execution(error.to_string());
        let (request, images, positions) = match prepared {
            Prepared::Text(request) => return self.decide(&request).map_err(execution),
            Prepared::Images {
                request,
                images,
                positions,
            } => (request, images, positions),
        };
        let mmproj = self.vision.as_ref().expect("an image request was prepared");
        let vision = ImageVision {
            processor: &mmproj.processor,
            config: &mmproj.config,
            dtype: mmproj.dtype,
        };
        let mut lease = self.checkout().map_err(execution)?;
        lease.start_external_work();
        let options = self.decode.options(0, ReadbackArm::Blocking, 0);
        let response = decide_images(
            &self.model,
            &request,
            &images,
            positions,
            vision,
            &mmproj.encoder,
            &self.device,
            |rows, readback| match self.ragged_arm() {
                Some(ragged) => (ragged.images)(self, rows, &request, readback, &options),
                None => {
                    request.execute_with_cache_factory(rows, readback, &options, |_, device| {
                        self.model.new_cache(&self.config, device)
                    })
                }
            },
        )?;
        lease.mark_complete();
        Ok(response)
    }
}

/// Actual app GGUF LFM2 factory, not a second model-only Loaded wrapper.
/// Tokenizer is explicit (same checkpoint/caller), avoiding a guessed sibling or
/// downloaded vocabulary. Named checkpoint policy or explicit generic policy is
/// optional: ordinary generation and prepared decisions remain available without
/// it. All header/model/tokenizer/initial-chat producers run inside one load scope.
/// Where an LFM2 GGUF's tokenizer comes from: a `tokenizer.json` the caller
/// names, or the GGUF's own vocabulary (llama.cpp's source for the same file).
#[derive(Clone, Copy)]
pub enum TokenizerSource<'a> {
    File(&'a Path),
    Embedded,
}

pub fn load_lfm2_gguf_admitted(
    checkpoint: &Path,
    tokenizer_path: &Path,
    policy_source: Option<CausalPolicySource>,
    device: &Device,
    load: LoadContext,
    decode: DecodeConfig,
) -> Result<Loaded<CausalEngine<lloom::models::quantized_lfm2::Model>>> {
    Loaded::new(device, load, |device, _context| {
        produce_lfm2_gguf(
            checkpoint,
            TokenizerSource::File(tokenizer_path),
            policy_source,
            device,
            decode,
        )
    })
}

fn produce_lfm2_gguf(
    checkpoint: &Path,
    tokenizer_path: TokenizerSource<'_>,
    policy_source: Option<CausalPolicySource>,
    device: &Device,
    decode: DecodeConfig,
) -> Result<CausalEngine<lloom::models::quantized_lfm2::Model>> {
    produce_lfm2_gguf_with(
        checkpoint,
        None,
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
    mmproj: Option<&Path>,
    tokenizer_path: TokenizerSource<'_>,
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
    let tokenizer = match tokenizer_path {
        TokenizerSource::File(path) => Tokenizer::from_file(path)
            .map_err(|error| Error::msg(format!("loading causal tokenizer: {error}")))?,
        TokenizerSource::Embedded => {
            use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
            Tokenizer::from_gguf(&content)
                .map_err(|error| Error::msg(format!("the GGUF's tokenizer: {error}")))?
        }
    };
    let policy = match policy_source {
        Some(CausalPolicySource::Explicit(policy)) => Some(policy),
        Some(CausalPolicySource::Named {
            max_options,
            noul_aliases,
            state_nullability,
        }) => {
            let facts = CheckpointDecision::from_gguf(&content.metadata)?;
            let Some(binding::Binding::Template(binding)) =
                binding::resolve(&Selection::Auto, &facts, SystemOneTemplateKind::D1Causal)?
            else {
                lloom_tensor::bail!("checkpoint is not a native decision model: it has no decision type and systemone template")
            };
            Some(CausalPolicy {
                template: binding.template,
                calibration: binding.calibration,
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
    let vision = mmproj
        .map(|path| MmprojVision::load(path, &tokenizer, config.hidden_size, device))
        .transpose()?;
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
            // This concrete immutable quantized model is constructed on the
            // same supplied Device. Custom generic models opt in separately.
            // Native per-binding/error retirement closure is still unproven.
            recovery: Some(Box::new(AssociatedDeviceRecovery)),
            vision,
            ragged: None,
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

#[cfg(test)]
mod model_stage_tests;

#[cfg(test)]
mod raw_policy_tests;

#[cfg(test)]
mod generic_recovery_tests;
