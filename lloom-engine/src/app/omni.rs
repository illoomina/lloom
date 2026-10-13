//! Supplied-domain non-generating omni execution over its one existing owner.
//! Default registration, encoded-media decoding and complete enrollment remain
//! unfinished acceptance work. Native construction/error custody is not proved.
use super::causal_loaded::served::{Predictor, Request};
use crate::decision::{
    parse_systemone_questions, parse_systemone_state, NoulAliases, OmniMedia, OmniPreparedQuestion,
    OmniReadout, OrderedJson, Question, StateNullability, SystemOneError, SystemOneMedia,
    SystemOneResponse, SystemOneTemplate, SystemOneTemplateKind,
};
use lloom::models::d1_omni::{
    DecisionRow, LoadedOmni, OmniConfig, OmniImage, OmniMetadata, OmniPass, RelativeInputs,
    TrunkBatch, VisionImagePlan,
};
use lloom::nn::memory::LoadContext;
use lloom_tensor::{DType, Device, Error, Result, Tensor};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Explicit caller source/policy; no template/limits or media marker guessed.
pub struct Policy {
    pub template: SystemOneTemplate,
    pub max_options: usize,
    pub noul_aliases: NoulAliases,
    pub state_nullability: StateNullability,
    pub media_marker: String,
}
impl Policy {
    /// d1-omni's own template: the GGUF's `systemone` template, or for a
    /// safetensors directory the engine-shipped one (`decision::binding`).
    /// Calibration comes with the loaded model's configuration.
    pub fn native(
        checkpoint: &Path,
        max_options: usize,
        noul_aliases: NoulAliases,
        state_nullability: StateNullability,
        media_marker: String,
    ) -> Result<Self> {
        use crate::decision::binding::{self, CheckpointDecision, Selection};
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
        let Some(binding::Binding::Template(binding)) =
            binding::resolve(&Selection::Auto, &facts, SystemOneTemplateKind::D1Omni)?
        else {
            lloom_tensor::bail!("checkpoint is not a d1-omni decision model")
        };
        let policy = Self {
            template: binding.template,
            max_options,
            noul_aliases,
            state_nullability,
            media_marker,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        if self.template.kind() != SystemOneTemplateKind::D1Omni || self.max_options == 0 {
            lloom_tensor::bail!("omni needs its explicit template kind and positive option bound")
        }
        Ok(())
    }
}
/// Decoded inputs keep caller origins; decoding is NOT covered retroactively.
/// Raw encoded endpoint media still requires its bounded decoding integration.
#[derive(Clone, Copy)]
pub enum Media<'a> {
    Text,
    Images(&'a [OmniImage<'a>]),
    AudioF32(&'a [f32]),
    AudioI16(&'a [i16]),
}
/// Separate mutable callbacks preserve custom backend/workspace policy. No
/// automatic dense fallback, native topology inference or per-layer dyn call.
pub struct Attention<T, H, V, A> {
    pub trunk: T,
    pub head: H,
    pub vision: V,
    pub audio: A,
}
/// Family-owned recovery covers all queues/scratch touched by its callbacks.
/// Refusal/unwind retains the incomplete marker; no cleanup is called in Drop.
/// Metadata/device equality alone cannot discharge this custom topology contract.
pub trait Recovery {
    fn before_reuse(&mut self, metadata: OmniMetadata<'_>) -> Result<()>;
}

/// The predictor of an owner admitted on the machine at load
/// (`LoadedOmni::from_*_on_machine`), whose passes reserve nothing.
pub struct MachineAdmitted;
impl Predictor for MachineAdmitted {
    fn predicted_extra(&self, _: Request<'_>) -> Result<usize> {
        Ok(0)
    }
}

/// Recovery for an owner whose every callback queues on its one device's
/// registered commands (the default attention does): wait for what an
/// interrupted request left in flight.
pub struct DeviceRecovery(pub Device);
impl Recovery for DeviceRecovery {
    fn before_reuse(&mut self, _: OmniMetadata<'_>) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = &self.0 {
            let marker = device.flush_marker()?;
            return device.wait_marker_prefix_data(&marker);
        }
        self.0.synchronize()
    }
}

/// Where the omni tokenizer comes from: a `tokenizer.json`, or a GGUF's own
/// vocabulary (llama.cpp's source for the same file).
pub enum TokenizerSource<'a> {
    File(&'a Path),
    Gguf(&'a Path),
}
impl TokenizerSource<'_> {
    fn load(&self, facts: OmniMetadata<'_>) -> Result<Text> {
        match self {
            Self::File(path) => Text::load(path, facts),
            Self::Gguf(path) => {
                use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
                let mut file = std::fs::File::open(path)?;
                let content = lloom_tensor::quantized::gguf_file::Content::read(&mut file)?;
                let tokenizer = Tokenizer::from_gguf(&content)
                    .map_err(|e| Error::Msg(format!("the GGUF's tokenizer: {e}")))?;
                Text::new(tokenizer, facts)
            }
        }
    }
}
struct Text {
    tokenizer: Tokenizer,
    marker: u32,
}
impl Text {
    fn load(path: &Path, facts: OmniMetadata<'_>) -> Result<Self> {
        Self::new(Tokenizer::from_file(path).map_err(Error::wrap)?, facts)
    }
    fn new(tokenizer: Tokenizer, facts: OmniMetadata<'_>) -> Result<Self> {
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| id as usize >= facts.vocab_size)
        {
            lloom_tensor::bail!("omni tokenizer id is outside resolved model vocabulary")
        }
        let marker = tokenizer
            .token_to_id("<|mask|>")
            .ok_or_else(|| Error::Msg("omni tokenizer has no decision marker".into()))?;
        Ok(Self { tokenizer, marker })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(Error::wrap)
    }
}
/// &mut self is request exclusion without imposing HTTP-only Sync. The same
/// Loaded model and one tokenizer are private; no unscoped ownership escape.
pub struct Engine<Q, R> {
    owner: LoadedOmni<Text>,
    policy: Arc<Policy>,
    predictor: Q,
    recovery: R,
    incomplete: bool,
}
impl<Q: Predictor, R: Recovery> Engine<Q, R> {
    /// # Safety
    /// Safetensor mappings must remain unchanged for this owner's lifetime.
    /// This does not establish native partial-construction retirement closure.
    pub unsafe fn load_safetensors<P: AsRef<Path>>(
        config: OmniConfig,
        weights: &[P],
        tokenizer: &Path,
        device: &Device,
        dtype: DType,
        load: LoadContext,
        policy: Arc<Policy>,
        predictor: Q,
        recovery: R,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        policy.validate()?;
        let owner = unsafe {
            LoadedOmni::from_safetensors_with(config, weights, dtype, device, load, |facts| {
                Text::load(tokenizer, facts)
            })?
        };
        Ok(Self {
            owner,
            policy,
            predictor,
            recovery,
            incomplete: false,
        })
    }
    /// Actual main+mmproj factory; no companion name or separate budget inferred.
    pub fn load_gguf(
        main: &Path,
        media: &Path,
        tokenizer: &Path,
        device: &Device,
        load: LoadContext,
        policy: Arc<Policy>,
        predictor: Q,
        recovery: R,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        policy.validate()?;
        let owner = LoadedOmni::from_gguf_with(main, media, device, load, |facts| {
            Text::load(tokenizer, facts)
        })?;
        Ok(Self {
            owner,
            policy,
            predictor,
            recovery,
            incomplete: false,
        })
    }
    /// [`Self::load_safetensors`], admitted on the machine at load.
    ///
    /// # Safety
    /// Safetensor mappings must remain unchanged for this owner's lifetime.
    pub unsafe fn load_safetensors_on_machine<P: AsRef<Path>>(
        config: OmniConfig,
        weights: &[P],
        tokenizer: TokenizerSource<'_>,
        device: &Device,
        dtype: DType,
        machine: lloom_tensor::machine::Admitted,
        policy: Arc<Policy>,
        predictor: Q,
        recovery: R,
    ) -> Result<Self> {
        policy.validate()?;
        let owner = unsafe {
            LoadedOmni::from_safetensors_on_machine(
                config,
                weights,
                dtype,
                device,
                machine,
                |facts| tokenizer.load(facts),
            )?
        };
        Ok(Self {
            owner,
            policy,
            predictor,
            recovery,
            incomplete: false,
        })
    }
    /// [`Self::load_gguf`], admitted on the machine at load.
    #[allow(clippy::too_many_arguments)]
    pub fn load_gguf_on_machine(
        main: &Path,
        media: &Path,
        tokenizer: TokenizerSource<'_>,
        device: &Device,
        machine: lloom_tensor::machine::Admitted,
        policy: Arc<Policy>,
        predictor: Q,
        recovery: R,
    ) -> Result<Self> {
        policy.validate()?;
        let owner = LoadedOmni::from_gguf_on_machine(main, media, device, machine, |facts| {
            tokenizer.load(facts)
        })?;
        Ok(Self {
            owner,
            policy,
            predictor,
            recovery,
            incomplete: false,
        })
    }
    /// Ordinary caller templates remain renderable even when their layout does
    /// not support marked decision preparation. Allocation is inside this SAME
    /// owner pass; no tokenizer/model getter or implicit template is introduced.
    pub fn render_systemone(
        &self,
        id: &str,
        state: &OrderedJson,
        question: &Question<OrderedJson>,
    ) -> Result<String> {
        let predicted = self.predictor.predicted_extra(Request::OmniRender {
            id,
            state,
            question,
            policy: &self.policy,
        })?;
        self.owner.with_pass(predicted, |_, _| {
            self.policy
                .template
                .render(id, state, question, &[], SystemOneMedia::Text)
        })
    }
    /// Raw ordered text endpoint contract. Media sources are not silently dropped:
    /// encoded-media decoding remains a separate, unfinished integration edge.
    pub fn systemone_text<T, H, V, A>(
        &mut self,
        body: &OrderedJson,
        attention: &mut Attention<T, H, V, A>,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        V: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        A: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let predicted = self
            .predictor
            .predicted_extra(Request::OmniRawDecisions {
                body,
                policy: &self.policy,
            })
            .map_err(execution)?;
        let policy = &self.policy;
        let recovery = &mut self.recovery;
        let incomplete = &mut self.incomplete;
        self.owner.with_pass(predicted, |model,text| {
            Ok((|| {
                let state = parse_systemone_state(body, policy.state_nullability).map_err(invalid)?;
                if !state.media.is_empty() {
                    return Err(SystemOneError::Unsupported("omni raw encoded-media decoder is not installed; borrowed decoded-media execution remains available".into()));
                }
                let questions = parse_systemone_questions(body, policy.max_options, policy.noul_aliases).map_err(invalid)?;
                execute(model,text,policy,state.state.as_ref(),&questions,Media::Text,attention,recovery,incomplete)
            })())
        }).map_err(execution)?
    }
    /// The raw `/v1/systemone` body with its media decoded: PNG/JPEG images
    /// under `images`, or one WAV/FLAC clip at 16 kHz mono. A request carries
    /// images or audio, not both, as the reference refuses both.
    pub fn systemone_media<T, H, V, A>(
        &mut self,
        body: &OrderedJson,
        attention: &mut Attention<T, H, V, A>,
        images: crate::app::vision::ImageDecodeLimits,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        V: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        A: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let predicted = self
            .predictor
            .predicted_extra(Request::OmniRawDecisions {
                body,
                policy: &self.policy,
            })
            .map_err(execution)?;
        let policy = &self.policy;
        let recovery = &mut self.recovery;
        let incomplete = &mut self.incomplete;
        self.owner
            .with_pass(predicted, |model, text| {
                Ok((|| {
                    let state =
                        parse_systemone_state(body, policy.state_nullability).map_err(invalid)?;
                    let questions =
                        parse_systemone_questions(body, policy.max_options, policy.noul_aliases)
                            .map_err(invalid)?;
                    let state_json = state.state.as_ref();
                    match decode_media(&state.media, images)? {
                        Decoded::Text => execute(
                            model,
                            text,
                            policy,
                            state_json,
                            &questions,
                            Media::Text,
                            attention,
                            recovery,
                            incomplete,
                        ),
                        Decoded::Images(decoded) => {
                            let images: Vec<OmniImage<'_>> = decoded
                                .iter()
                                .map(|image| OmniImage {
                                    rgb: &image.rgb,
                                    height: image.height,
                                    width: image.width,
                                })
                                .collect();
                            execute(
                                model,
                                text,
                                policy,
                                state_json,
                                &questions,
                                Media::Images(&images),
                                attention,
                                recovery,
                                incomplete,
                            )
                        }
                        Decoded::Audio(audio_input::Samples::I16(samples)) => execute(
                            model,
                            text,
                            policy,
                            state_json,
                            &questions,
                            Media::AudioI16(&samples),
                            attention,
                            recovery,
                            incomplete,
                        ),
                        Decoded::Audio(audio_input::Samples::F32(samples)) => execute(
                            model,
                            text,
                            policy,
                            state_json,
                            &questions,
                            Media::AudioF32(&samples),
                            attention,
                            recovery,
                            incomplete,
                        ),
                    }
                })())
            })
            .map_err(execution)?
    }
    /// Actual shared-media execution. State/questions are already ordered typed
    /// caller inputs, not strings; one encoded prefix feeds independent rows.
    pub fn decide<T, H, V, A>(
        &mut self,
        state: &OrderedJson,
        questions: &[(String, Question<OrderedJson>)],
        media: Media<'_>,
        attention: &mut Attention<T, H, V, A>,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        V: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
        A: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let predicted = self
            .predictor
            .predicted_extra(Request::OmniDecisions {
                state,
                questions,
                media,
                policy: &self.policy,
            })
            .map_err(execution)?;
        let policy = &self.policy;
        let recovery = &mut self.recovery;
        let incomplete = &mut self.incomplete;
        self.owner
            .with_pass(predicted, |model, text| {
                Ok(execute(
                    model, text, policy, state, questions, media, attention, recovery, incomplete,
                ))
            })
            .map_err(execution)?
    }
}
/// The bytes a request's audio clip may take: 30 s of 16-bit mono at 16 kHz
/// as WAV, with room for its header.
const MAX_AUDIO_BYTES: usize = 1 << 20;

enum Decoded {
    Text,
    Images(Vec<super::vision::DecodedImage>),
    Audio(audio_input::Samples),
}

/// Each source decoded by what its bytes are: PNG/JPEG images, or one
/// WAV/FLAC clip; anything else, both kinds, or two clips refused by name.
fn decode_media(
    sources: &[crate::decision::MediaSource<'_>],
    images: crate::app::vision::ImageDecodeLimits,
) -> std::result::Result<Decoded, SystemOneError> {
    use super::lfm2::decision_media::{is_image, media_payload};
    let mut decoded_images = Vec::new();
    let mut audio = None;
    for source in sources {
        let bytes = media_payload(*source, images.max_encoded_bytes.max(MAX_AUDIO_BYTES))?;
        if is_image(&bytes) {
            if bytes.len() > images.max_encoded_bytes {
                return Err(SystemOneError::InvalidRequest(
                    "encoded image exceeds supplied byte limit".into(),
                ));
            }
            decoded_images.push(
                super::vision::DecodedImage::decode_bounded(&bytes, images)
                    .map_err(|e| SystemOneError::InvalidRequest(e.to_string()))?,
            );
        } else if audio_input::is_audio(&bytes) {
            if audio.is_some() {
                return Err(SystemOneError::InvalidRequest(
                    "a request carries one audio clip".into(),
                ));
            }
            audio = Some(audio_input::decode(bytes).map_err(SystemOneError::InvalidRequest)?);
        } else {
            return Err(SystemOneError::Unsupported(
                "d1-omni reads PNG/JPEG images and WAV/FLAC audio".into(),
            ));
        }
    }
    match (decoded_images.is_empty(), audio) {
        (true, None) => Ok(Decoded::Text),
        (false, None) => Ok(Decoded::Images(decoded_images)),
        (true, Some(audio)) => Ok(Decoded::Audio(audio)),
        (false, Some(_)) => Err(SystemOneError::InvalidRequest(
            "a request carries images or audio, not both".into(),
        )),
    }
}
fn invalid(error: Error) -> SystemOneError {
    SystemOneError::InvalidRequest(error.to_string())
}
fn execution(error: Error) -> SystemOneError {
    SystemOneError::Execution(error.to_string())
}
/// Marker-only custody: NEVER reset/drop backend state during callback unwind.
struct Work<'a> {
    incomplete: &'a mut bool,
    complete: bool,
}
impl Drop for Work<'_> {
    fn drop(&mut self) {
        *self.incomplete = !self.complete;
    }
}
fn execute<T, H, V, A, R: Recovery>(
    model: OmniPass<'_>,
    text: &Text,
    policy: &Policy,
    state: &OrderedJson,
    questions: &[(String, Question<OrderedJson>)],
    media: Media<'_>,
    attention: &mut Attention<T, H, V, A>,
    recovery: &mut R,
    incomplete: &mut bool,
) -> std::result::Result<SystemOneResponse, SystemOneError>
where
    T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
    H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    V: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    A: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
{
    if questions.is_empty() {
        return Err(SystemOneError::InvalidRequest(
            "omni needs nonempty questions".into(),
        ));
    }
    for (i, (id, q)) in questions.iter().enumerate() {
        q.validate().map_err(invalid)?;
        if q.option_count() > policy.max_options
            || questions[..i].iter().any(|(previous, _)| previous == id)
        {
            return Err(SystemOneError::InvalidRequest(
                "omni option bound or duplicate question identity".into(),
            ));
        }
    }
    let facts = model.metadata();
    let (kind, prefix_tokens, marker_count) = match media {
        Media::Text => (OmniMedia::Text, 0, 0),
        Media::Images(images) => {
            if !(1..=8).contains(&images.len()) {
                return Err(SystemOneError::InvalidRequest(
                    "omni needs1..=8 images".into(),
                ));
            }
            let mut tokens = 0usize;
            for image in images {
                let plan = VisionImagePlan::new(image.height, image.width, image.rgb.len())
                    .map_err(invalid)?;
                tokens = tokens.checked_add(plan.prefix_tokens()).ok_or_else(|| {
                    SystemOneError::InvalidRequest("omni image prefix overflow".into())
                })?;
            }
            (OmniMedia::Image, tokens, images.len())
        }
        Media::AudioF32(samples) => (
            OmniMedia::Audio,
            model
                .audio_frontend_scratch(samples.len())
                .map_err(invalid)?
                .valid_frames
                .div_ceil(8),
            1,
        ),
        Media::AudioI16(samples) => (
            OmniMedia::Audio,
            model
                .audio_frontend_scratch(samples.len())
                .map_err(invalid)?
                .valid_frames
                .div_ceil(8),
            1,
        ),
    };
    let allowance = match kind {
        OmniMedia::Text => facts.max_length,
        OmniMedia::Image => facts
            .text_allowance(prefix_tokens, facts.image_text_length)
            .map_err(invalid)?,
        OmniMedia::Audio => facts
            .text_allowance(prefix_tokens, facts.audio_text_length)
            .map_err(invalid)?,
    };
    if kind != OmniMedia::Text && policy.media_marker.is_empty() {
        return Err(SystemOneError::InvalidRequest(
            "omni needs explicit nonempty media marker".into(),
        ));
    }
    let rendering = match kind {
        OmniMedia::Text => SystemOneMedia::Text,
        OmniMedia::Image => SystemOneMedia::Images {
            count: marker_count,
            marker: &policy.media_marker,
        },
        OmniMedia::Audio => SystemOneMedia::Audio {
            marker: &policy.media_marker,
        },
    };
    let expected_media = policy.media_marker.repeat(marker_count);
    let (prepared, readouts) = prepare_questions(
        text,
        policy,
        state,
        questions,
        kind,
        rendering,
        &expected_media,
        allowance,
        facts,
    )?;
    let mut work = begin_work(facts, recovery, incomplete).map_err(execution)?;
    let prefix = match media {
        Media::Text => None,
        Media::Images(images) => Some(
            model
                .image_prefix(images, &mut attention.vision)
                .map_err(execution)?,
        ),
        Media::AudioF32(samples) => Some(
            model
                .audio_prefix_f32(samples, &mut attention.audio)
                .map_err(execution)?,
        ),
        Media::AudioI16(samples) => Some(
            model
                .audio_prefix_i16(samples, &mut attention.audio)
                .map_err(execution)?,
        ),
    };
    if let Some(prefix) = prefix.as_ref() {
        if prefix.dim(1).map_err(execution)? != prefix_tokens {
            return Err(SystemOneError::Execution(
                "omni actual prefix differs from prepared allowance".into(),
            ));
        }
    }
    let rows = prepared
        .iter()
        .map(|row| DecisionRow {
            tokens: &row.ids,
            markers: &row.markers,
            question_type: row.question_type as u32,
            prefix: prefix.as_ref(),
        })
        .collect::<Vec<_>>();
    let rows = model.prepare_rows(&rows).map_err(execution)?;
    let input_tokens =
        u64::try_from(rows.input_tokens()).map_err(|e| SystemOneError::Execution(e.to_string()))?;
    let logits = model
        .forward_with_attention(&rows, &mut attention.trunk, &mut attention.head)
        .map_err(execution)?
        .to_dtype(DType::F32)
        .map_err(execution)?
        .to_vec2::<f32>()
        .map_err(execution)?;
    if logits.len() != questions.len() {
        return Err(SystemOneError::Execution(
            "omni returned wrong question row count".into(),
        ));
    }
    let mut answers = Vec::with_capacity(questions.len());
    for (((id, q), readout), logits) in questions.iter().zip(&readouts).zip(&logits) {
        let probabilities = readout.probabilities(logits).map_err(execution)?;
        answers.push((
            id.clone(),
            q.systemone_answer(&probabilities).map_err(execution)?,
        ));
    }
    let response = SystemOneResponse::new(answers, input_tokens)?;
    work.complete = true;
    Ok(response)
}

/// Shared actual caller preparation, before model/native work is marked started.
fn prepare_questions(
    text: &Text,
    policy: &Policy,
    state: &OrderedJson,
    questions: &[(String, Question<OrderedJson>)],
    kind: OmniMedia,
    rendering: SystemOneMedia<'_>,
    expected_media: &str,
    allowance: usize,
    facts: OmniMetadata<'_>,
) -> std::result::Result<(Vec<OmniPreparedQuestion>, Vec<OmniReadout>), SystemOneError> {
    let mut prepared = Vec::with_capacity(questions.len());
    let mut readouts = Vec::with_capacity(questions.len());
    for (id, q) in questions {
        let rendered = policy
            .template
            .render(id, state, q, &[], rendering)
            .map_err(invalid)?;
        let row = OmniPreparedQuestion::from_systemone_rendered(
            &rendered,
            q,
            expected_media,
            text.marker,
            allowance,
            |piece| text.encode(piece),
        )
        .map_err(invalid)?;
        if row.ids.iter().any(|&id| id as usize >= facts.vocab_size) {
            return Err(SystemOneError::InvalidRequest(
                "omni token is outside resolved vocabulary".into(),
            ));
        }
        readouts
            .push(OmniReadout::prepare_systemone(q, kind, facts.temperatures).map_err(invalid)?);
        prepared.push(row);
    }
    Ok((prepared, readouts))
}
/// Actual entry gate: refusal/unwind cannot clear the old incomplete marker.
fn begin_work<'a, R: Recovery>(
    facts: OmniMetadata<'_>,
    recovery: &mut R,
    incomplete: &'a mut bool,
) -> Result<Work<'a>> {
    if *incomplete {
        recovery.before_reuse(facts)?;
    }
    *incomplete = true;
    Ok(Work {
        incomplete,
        complete: false,
    })
}
#[cfg(test)]
#[path = "omni_tests.rs"]
mod tests;

mod audio_input;
mod endpoint;
pub use endpoint::Endpoint;
pub(crate) mod serving;
