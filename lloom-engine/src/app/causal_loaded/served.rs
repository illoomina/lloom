//! Actual load-once `Served` caller over the same retained owner as decisions.
//! Prediction is an explicit caller coverage contract, not verified admission.
//! Default registration, enrolled host metadata and worker custody remain open.
use super::*;
use crate::app::common::{thinking_mode, DecodeTiming, MemGauge, PrefillTiming, TextProbe};
use crate::app::family::Capabilities;
use crate::app::served::Served;
use crate::app::template::{Rendered, Template, ThinkTags};
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

    /// The borrowed decoded pixels keep their caller origin. Prediction covers
    /// placeholder preparation, not decoding bytes that occurred before entry.
    ImagePlaceholder(&'a super::super::vision::DecodedImage),
    Images {
        prompt: &'a [u32],
        images: &'a [super::super::vision::DecodedImage],
        params: &'a DecodeParams,
    },
    /// Full incumbent mode inputs, including sequences/rows/speculation and
    /// both measured/reference arms. No zero or generic default prediction.
    Mode(&'a super::super::modes::ModeSpec),
    Decisions(&'a OrderedJson),
    /// The same caller-owned policy, borrowed before ordered parsing/rendering,
    /// token preparation, cache/readout or executor work allocates.
    PolicyDecisions {
        body: &'a OrderedJson,
        policy: &'a CausalPolicy,
    },
    /// Explicit family image bounds are available before any encoded media
    /// is copied/decoded. The same body/policy and existing pass cover all work.
    PolicyImageDecisions {
        body: &'a OrderedJson,
        policy: &'a CausalPolicy,
        limits: super::super::vision::ImageDecodeLimits,
    },
    PolicyRender {
        policy: &'a CausalPolicy,
        id: &'a str,
        state: &'a OrderedJson,
        question: &'a crate::decision::Question<OrderedJson>,
        labels: &'a [String],
    },
    /// Same resolved omni owner. Inputs are borrowed BEFORE ordered parsing,
    /// rendering, media encoding, bidirectional rows and returned readout work.
    OmniHealth {
        model_name: &'a str,
    },
    OmniRender {
        id: &'a str,
        state: &'a OrderedJson,
        question: &'a crate::decision::Question<OrderedJson>,
        policy: &'a super::super::omni::Policy,
    },
    OmniRawDecisions {
        body: &'a OrderedJson,
        policy: &'a super::super::omni::Policy,
    },
    OmniDecisions {
        state: &'a OrderedJson,
        questions: &'a [(String, crate::decision::Question<OrderedJson>)],
        media: super::super::omni::Media<'a>,
        policy: &'a super::super::omni::Policy,
    },
    PreparedDecisions(&'a CausalRequest),
    SystemOneRender {
        id: &'a str,
        state: &'a OrderedJson,
        question: &'a crate::decision::Question<OrderedJson>,
        labels: &'a [String],
    },
    GenerationPrompt {
        kwarg: Option<bool>,
        want: Option<bool>,
        /// Load-time suffix available to bound the returned token vector copy.
        ids: Option<&'a [u32]>,
    },
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
    Metadata(MetadataRequest<'a>),
    Logprobs {
        prompt: &'a [u32],
        k: usize,
    },
}
pub trait Predictor: Send + Sync {
    fn predicted_extra(&self, request: Request<'_>) -> Result<usize>;
}

/// Allocating metadata producers are request work too. Inputs are borrowed;
/// caller prediction/enrollment and legacy infallible API policy remain explicit.
pub enum MetadataRequest<'a> {
    Health,
    Describe,
    GenerationConfig,
    VisionReport,
    Diagnostics,
    LoadWarnings,
    NoteLoadWarnings(&'a [LoadWarning]),
}

/// Shared owning admission seam; E need not implement the HTTP Served boundary.
/// One existing Loaded namespace and one supplied predictor, no new quota.
pub(crate) struct Admitted<P, E> {
    pub(crate) loaded: Loaded<E>,
    predictor: P,
}
impl<P: Predictor, E> Admitted<P, E> {
    pub(crate) fn new(loaded: Loaded<E>, predictor: P) -> Self {
        Self { loaded, predictor }
    }
    pub(crate) fn predict(&self, request: Request<'_>) -> Result<usize> {
        self.predictor.predicted_extra(request)
    }
    pub(crate) fn pass<T>(
        &self,
        request: Request<'_>,
        run: impl FnOnce(&E) -> Result<T>,
    ) -> Result<T> {
        let predicted = self.predict(request)?;
        self.loaded
            .with_pass(predicted, |engine, _context| run(engine))
    }
}

/// Who scopes the owner's request work. `Admitted`: one model domain and the
/// caller's prediction per pass. `Unadmitted`: the default `lloom serve`
/// route, admitted on the machine at load as the other default families are,
/// with no per-pass domain reservation (no default family has a production
/// predictor yet). Both run the same owner and the same adapter.
pub(crate) enum Custody<P, E> {
    Admitted(Admitted<P, E>),
    Unadmitted {
        engine: E,
        /// The machine reservation the load took, held for the model's life.
        _machine: lloom_tensor::machine::Admitted,
    },
}
impl<P: Predictor, E> Custody<P, E> {
    fn pass<T>(&self, request: Request<'_>, run: impl FnOnce(&E) -> Result<T>) -> Result<T> {
        match self {
            Self::Admitted(owner) => owner.pass(request, run),
            Self::Unadmitted { engine, .. } => run(engine),
        }
    }
    fn admission(&self) -> &'static str {
        match self {
            Self::Admitted(_) => "caller-supplied coverage; producer enrollment incomplete",
            Self::Unadmitted { .. } => "machine admission at load; no per-pass domain reservation",
        }
    }
}

/// The predictor type of an unadmitted owner, which asks for none.
pub struct NoPrediction;
impl Predictor for NoPrediction {
    fn predicted_extra(&self, _: Request<'_>) -> Result<usize> {
        lloom_tensor::bail!("an unadmitted owner takes no per-pass prediction")
    }
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
    vocab_size: usize,
    decode: DecodeConfig,
    template: Option<Template>,
    seed: Vec<u32>,
    think: Option<ThinkTags>,
    gen_prompts: [[Option<Vec<u32>>; 3]; 3],
    capabilities: Capabilities,
}
/// No model/tokenizer/quota copy and no borrowed unscoped Loaded escape.
pub struct Engine<P, M: CausalLM = Model> {
    metadata: Metadata,
    warnings: Vec<LoadWarning>,
    decisions: Option<DecisionDispatch<M>>,
    /// The template binding the decision route renders with, for the load
    /// line and `/v1/models` (`Served::decision_binding`).
    binding: Option<(Option<crate::decision::binding::DecisionType>, String)>,
    owner: Custody<P, CausalEngine<M>>,
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
    let serving = Serving {
        config,
        decode,
        capabilities: lfm2_gguf_capabilities(),
    };
    let owner_decode = serving.decode.clone();
    load_from_factory(
        device,
        Admission::Domain { load, predictor },
        serving,
        Some(DecisionDispatch::retained()),
        None,
        |device, preflight| {
            produce_lfm2_gguf_with(
                files.checkpoint,
                None,
                TokenizerSource::File(files.tokenizer),
                policy,
                device,
                owner_decode,
                |resolved, tokenizer, policy| {
                    preflight(
                        FamilyFacts {
                            vocab_size: resolved.vocab_size,
                            capacity: resolved.context_length,
                        },
                        tokenizer,
                        policy,
                    )
                },
            )
            .map(|(engine, ())| engine)
        },
    )
}

/// The default `lloom serve` route for an LFM2 GGUF: the same producer and
/// adapter as [`load_lfm2_gguf`], admitted on the machine at load (`machine`,
/// taken by the family before any weight is read) instead of in a model
/// domain. `binding` is the template `decision.template` and the checkpoint
/// resolved to; `None` serves chat only.
#[allow(clippy::too_many_arguments)]
pub fn load_lfm2_gguf_unadmitted(
    checkpoint: &Path,
    mmproj: Option<&Path>,
    tokenizer: TokenizerSource<'_>,
    binding: Option<BoundPolicy>,
    device: &Device,
    machine: lloom_tensor::machine::Admitted,
    decode: DecodeConfig,
    config: Config,
) -> Result<Engine<NoPrediction>> {
    let serving = Serving {
        config,
        decode,
        capabilities: lfm2_gguf_capabilities(),
    };
    let owner_decode = serving.decode.clone();
    let (policy, described) = match binding {
        Some(bound) => (
            Some(CausalPolicySource::Explicit(bound.policy)),
            Some((bound.native, bound.origin)),
        ),
        None => (None, None),
    };
    load_from_factory(
        device,
        Admission::Machine(machine),
        serving,
        Some(DecisionDispatch::retained()),
        described,
        |device, preflight| {
            produce_lfm2_gguf_with(
                checkpoint,
                mmproj,
                tokenizer,
                policy,
                device,
                owner_decode,
                |resolved, tokenizer, policy| {
                    preflight(
                        FamilyFacts {
                            vocab_size: resolved.vocab_size,
                            capacity: resolved.context_length,
                        },
                        tokenizer,
                        policy,
                    )
                },
            )
            .map(|(engine, ())| engine)
        },
    )
}

/// Request-level function dispatch installs the retained decision operation
/// without imposing RetainState on generic generation or template rendering.
/// No model/tokenizer/cache copy; this is not a per-token dispatch mechanism.
struct DecisionDispatch<M: CausalLM> {
    prepared: Option<PreparedDecision<M>>,
    raw: Option<RawDecision<M>>,
    batch: Option<BatchDecision<M>>,
}
type PreparedDecision<M> = fn(&CausalEngine<M>, &CausalRequest) -> Result<SystemOneResponse>;
type RawDecision<M> =
    fn(&CausalEngine<M>, &OrderedJson) -> std::result::Result<SystemOneResponse, SystemOneError>;
type BatchDecision<M> = fn(
    &CausalEngine<M>,
    &[&OrderedJson],
) -> Vec<std::result::Result<SystemOneResponse, SystemOneError>>;
impl<M: CausalLM> DecisionDispatch<M>
where
    M::Cache: CacheReset + CacheRewind + RetainState,
{
    fn retained() -> Self {
        Self {
            prepared: Some(CausalEngine::<M>::decide),
            raw: Some(CausalEngine::<M>::systemone),
            batch: Some(CausalEngine::<M>::systemone_batch),
        }
    }
}

/// Resolved family facts supplied by the actual typed factory before weights.
/// No weight schema or byte predictor is reconstructed at the app boundary.
struct FamilyFacts {
    vocab_size: usize,
    capacity: usize,
}
struct Serving {
    config: Config,
    decode: DecodeConfig,
    capabilities: Capabilities,
}
/// What scopes the load: a supplied model domain and its predictor, or the
/// machine reservation the family already took.
enum Admission<P> {
    Domain { load: LoadContext, predictor: P },
    Machine(lloom_tensor::machine::Admitted),
}

/// Shared supplied-domain factory route. The actual LFM2 GGUF caller above
/// invokes its typed preflight before materialization; future families must
/// do the same. This does not register a family or infer complete admission.
fn load_from_factory<P: Predictor, M: CausalLM>(
    device: &Device,
    admission: Admission<P>,
    serving: Serving,
    decisions: Option<DecisionDispatch<M>>,
    binding: Option<(Option<crate::decision::binding::DecisionType>, String)>,
    produce: impl FnOnce(
        &Device,
        &mut dyn FnMut(FamilyFacts, &Tokenizer, Option<&CausalPolicy>) -> Result<()>,
    ) -> Result<CausalEngine<M>>,
) -> Result<Engine<P, M>> {
    lloom_tensor::memory_admission::begin_model_load();
    let mut metadata = None;
    let Serving {
        config,
        decode,
        mut capabilities,
    } = serving;
    // This is a one-sequence text adapter, not a replacement for incumbent
    // batch/speculation/media family owners. Refuse overclaimed embedding facts.
    if !capabilities.generation
        || capabilities.batch
        || capabilities.prefix_share
        || !capabilities.speculation.is_empty()
        || capabilities.vision
        || capabilities.model_axes
        || capabilities.executor_arms
        || capabilities.eos_gate
    {
        lloom_tensor::bail!("generic causal adapter requires one-sequence text capabilities without unimplemented family mechanisms")
    }
    let mut inputs = Some((config, decode));
    let produce_checked = |device: &Device| -> Result<CausalEngine<M>> {
        let mut preflight =
            |facts: FamilyFacts, tokenizer: &Tokenizer, policy: Option<&CausalPolicy>| {
                let (config, decode) = inputs
                    .take()
                    .ok_or_else(|| Error::msg("serving preflight called more than once"))?;
                capabilities.decisions =
                    policy.is_some() && decisions.as_ref().is_some_and(|route| route.raw.is_some());
                metadata = Some(prepare_metadata(
                    facts,
                    tokenizer,
                    device,
                    config,
                    decode,
                    capabilities,
                )?);
                Ok(())
            };
        let engine = produce(device, &mut preflight)?;
        if metadata.is_none() {
            lloom_tensor::bail!("family producer did not run serving preflight")
        }
        let facts = metadata.as_ref().expect("checked serving preflight");
        if engine.capacity != facts.capacity || engine.model.vocab_size() != facts.vocab_size {
            lloom_tensor::bail!("family materialization differs from serving preflight")
        }
        Ok(engine)
    };
    let owner = match admission {
        Admission::Domain { load, predictor } => Custody::Admitted(Admitted::new(
            Loaded::new(device, load, |device, _context| produce_checked(device))?,
            predictor,
        )),
        Admission::Machine(machine) => Custody::Unadmitted {
            engine: produce_checked(device)?,
            _machine: machine,
        },
    };
    Ok(Engine {
        owner,
        metadata: metadata.expect("successful preflight installed serving facts"),
        warnings: Vec::new(),
        decisions,
        binding,
    })
}

fn prepare_metadata(
    facts: FamilyFacts,
    tokenizer: &Tokenizer,
    device: &Device,
    config: Config,
    decode: DecodeConfig,
    capabilities: Capabilities,
) -> Result<Metadata> {
    if facts.capacity == 0
        || facts.vocab_size == 0
        || config.eos_ids.is_empty()
        || config
            .eos_ids
            .iter()
            .chain(config.bos_id.iter())
            .any(|&id| id as usize >= facts.vocab_size)
    {
        lloom_tensor::bail!("serving BOS/EOS ids must fit the loaded vocabulary and EOS/context/vocabulary must be nonempty")
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
    if seed.is_empty() || seed.iter().any(|&id| id as usize >= facts.vocab_size) {
        lloom_tensor::bail!("serving seed must encode valid nonempty vocabulary ids")
    }
    let probe = TextProbe {
        tokenizer,
        template: template.as_ref(),
        bos_token: &config.bos_token,
    };
    let think = probe.think_tags();
    let gen_prompts = probe.generation_prompts(think.as_ref());
    let marker_ids = think.iter().flat_map(|t| {
        [Some(t.start), Some(t.end), t.tool_call]
            .into_iter()
            .flatten()
    });
    let prompt_ids = gen_prompts.iter().flatten().flatten().flatten().copied();
    if marker_ids
        .chain(prompt_ids)
        .any(|id| id as usize >= facts.vocab_size)
    {
        lloom_tensor::bail!("serving template probe ids must fit the loaded vocabulary")
    }
    Ok(Metadata {
        config,
        device: device.clone(),
        capacity: facts.capacity,
        vocab_size: facts.vocab_size,
        decode,
        template,
        seed,
        think,
        gen_prompts,
        capabilities,
    })
}

impl<P: Predictor, M: CausalLM> Engine<P, M>
where
    M::Cache: CacheReset + CacheRewind,
{
    /// Programmatic generation does not inherit the HTTP Served Send/Sync
    /// boundary. Every producer still uses the same predicted admitted pass.
    pub fn encode_admitted(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        self.pass(Request::Encode { text, add_special }, |engine| {
            engine.encode(text, add_special)
        })
    }
    pub fn complete_admitted(
        &self,
        prompt: &[u32],
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded> {
        self.pass(Request::Generation { prompt, params }, |engine| {
            engine.complete(prompt, params, &mut on_fragment)
        })
    }
    /// Preserve explicit generic prepared decisions when named raw rendering is
    /// absent or uncertified. The adapter never exposes an unscoped inner owner.
    pub fn decide_prepared(&self, request: &CausalRequest) -> Result<SystemOneResponse> {
        let execute = self
            .decisions
            .as_ref()
            .and_then(|route| route.prepared)
            .ok_or_else(|| Error::msg("family supplied no retained decision operation"))?;
        self.pass(Request::PreparedDecisions(request), |engine| {
            execute(engine, request)
        })
    }
    /// Ordinary generic policy rendering does not require a shared-state
    /// certificate. Its tokens can feed an explicit prepared request; the
    /// adapter does not invent certification or replay a shared trunk.
    pub fn render_systemone(
        &self,
        id: &str,
        state: &OrderedJson,
        question: &crate::decision::Question<OrderedJson>,
        labels: &[String],
    ) -> Result<String> {
        self.pass(
            Request::SystemOneRender {
                id,
                state,
                question,
                labels,
            },
            |engine| engine.render_systemone(id, state, question, labels),
        )
    }
    fn pass<T>(
        &self,
        request: Request<'_>,
        run: impl FnOnce(&CausalEngine<M>) -> Result<T>,
    ) -> Result<T> {
        self.owner.pass(request, run)
    }
    pub fn seed_prompt_row_checked(&self, n: usize, row: usize) -> Result<Vec<u32>> {
        check_context_capacity(self.metadata.capacity, n, 0, 0).map_err(Error::wrap)?;
        self.pass(Request::Seed { positions: n, row }, |_| {
            Ok(self.seed(n, row))
        })
    }
    pub fn health_checked(&self) -> Result<Value> {
        self.pass(Request::Metadata(MetadataRequest::Health), |_| {
            Ok(json!({"capacity": self.metadata.capacity, "capabilities": self.metadata.capabilities.json(), "admission": self.owner.admission()}))
        })
    }
    pub fn describe_checked(&self) -> Result<String> {
        self.pass(Request::Metadata(MetadataRequest::Describe), |_| {
            Ok(format!(
                "load-once causal owner; {}",
                self.metadata.capabilities.line()
            ))
        })
    }
    pub fn load_warnings_checked(&self) -> Result<Vec<LoadWarning>> {
        self.pass(Request::Metadata(MetadataRequest::LoadWarnings), |_| {
            Ok(self.warnings.clone())
        })
    }
    pub fn note_load_warnings_checked(&mut self, warnings: Vec<LoadWarning>) -> Result<()> {
        let incoming = &warnings;
        let held = &mut self.warnings;
        self.owner.pass(
            Request::Metadata(MetadataRequest::NoteLoadWarnings(incoming)),
            |_| {
                held.splice(0..0, incoming.iter().cloned());
                Ok(())
            },
        )
    }
    fn seed(&self, n: usize, row: usize) -> Vec<u32> {
        (0..n)
            .map(|i| self.metadata.seed[(i.wrapping_add(row)) % self.metadata.seed.len()])
            .collect()
    }
    fn benchmark<'a>(&self, engine: &'a CausalEngine<M>) -> Result<RequestLease<'a, M>> {
        let mut lease = engine.checkout()?;
        lease.start_decode(&engine.model, &engine.device, engine.decode.clone());
        lease.session().reset();
        Ok(lease)
    }
}
impl<P: Predictor, M: CausalLM + Send + Sync> Served for Engine<P, M>
where
    M::Config: Send + Sync,
    M::Cache: CacheReset + CacheRewind + Send,
{
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
    fn health_checked(&self) -> anyhow::Result<Value> {
        Ok(Engine::health_checked(self)?)
    }
    fn describe_checked(&self) -> anyhow::Result<String> {
        Ok(Engine::describe_checked(self)?)
    }
    fn load_warnings_checked(&self) -> anyhow::Result<Vec<LoadWarning>> {
        Ok(Engine::load_warnings_checked(self)?)
    }
    fn note_load_warnings_checked(&mut self, warnings: Vec<LoadWarning>) -> anyhow::Result<()> {
        Ok(Engine::note_load_warnings_checked(self, warnings)?)
    }
    fn health(&self) -> Value {
        self.health_checked()
            .expect("legacy infallible health cannot report admission refusal; use health_checked")
    }
    fn describe(&self) -> String {
        self.describe_checked().expect(
            "legacy infallible description cannot report admission refusal; use describe_checked",
        )
    }
    fn capabilities(&self) -> Capabilities {
        self.metadata.capabilities
    }

    fn load_warnings(&self) -> Vec<LoadWarning> {
        self.load_warnings_checked().expect(
            "legacy infallible warnings cannot report admission refusal; use load_warnings_checked",
        )
    }
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>) {
        self.note_load_warnings_checked(warnings).expect("legacy infallible warning insertion cannot report admission refusal; use note_load_warnings_checked");
    }
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let execute = self
            .decisions
            .as_ref()
            .and_then(|route| route.raw)
            .ok_or_else(|| {
                SystemOneError::Unsupported("family supplied no retained decision operation".into())
            })?;
        self.pass(Request::Decisions(body), |engine| Ok(execute(engine, body)))
            .map_err(|e| SystemOneError::Execution(e.to_string()))?
    }
    /// Packed only under the default route's load-time admission: an
    /// admitted owner prices each request's pass, so it answers one by one.
    fn systemone_batch(
        &self,
        bodies: &[&OrderedJson],
    ) -> Vec<std::result::Result<SystemOneResponse, SystemOneError>> {
        let batch = self.decisions.as_ref().and_then(|route| route.batch);
        match (&self.owner, batch) {
            (Custody::Unadmitted { engine, .. }, Some(execute)) => execute(engine, bodies),
            _ => bodies.iter().map(|body| self.systemone(body)).collect(),
        }
    }
    fn decision_binding(&self) -> Option<(Option<crate::decision::binding::DecisionType>, String)> {
        self.binding.clone()
    }
    fn encode(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self.encode_admitted(text, add_special)?)
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
    fn think_tags(&self) -> Option<ThinkTags> {
        // Scalar-only metadata: no request allocation or tokenizer escape.
        self.metadata.think.clone()
    }
    fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<Vec<u32>> {
        self.generation_prompt_checked(kwarg, want).expect(
            "legacy infallible prompt metadata cannot report admission refusal; use generation_prompt_checked",
        )
    }
    fn generation_prompt_checked(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let ids = self.metadata.gen_prompts[thinking_mode(kwarg)][thinking_mode(want)].as_deref();
        Ok(
            self.pass(Request::GenerationPrompt { kwarg, want, ids }, |_| {
                Ok(ids.map(<[u32]>::to_vec))
            })?,
        )
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
        Ok(self.complete_admitted(prompt, params, |text| on_fragment(text))?)
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
            let mut cache = engine.model.new_cache(&engine.config, &engine.device)?;
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
            anyhow::bail!(
                "{} text owner serves exactly one sequence, requested {batch}",
                self.metadata.capabilities.family
            )
        }
        Ok(())
    }
    fn shutdown(&self) {} // No owner worker threads are launched by this adapter.
}

pub(crate) fn lfm2_gguf_capabilities() -> Capabilities {
    Capabilities {
        family: "lfm2-gguf",
        generation: true,
        decisions: false,
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
        vision_missing: "chat takes text only on this owner; decisions read images through \
            the mmproj's tower (model.mmproj), and the safetensors checkpoint's lfm2 family chats \
            with images",
        eos_gate: false,
        eos_gate_missing: "generic GGUF forward has no gated kernels",
    }
}
