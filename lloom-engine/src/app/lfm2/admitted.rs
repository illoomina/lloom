//! Same full LFM owner for supplied-domain generation and prepared decisions.
//! Caller prediction is a coverage contract, not proof of producer enrollment.
//! Legacy infallible metadata refusal policy and default registration stay open.
use super::*;
use crate::app::causal_loaded::served::{Admitted, MetadataRequest, Predictor, Request};
use crate::app::causal_loaded::CausalPolicy;
use crate::app::served::Served;
use crate::decision::{
    CausalRequest, OrderedJson, SystemOneError, SystemOneMedia, SystemOneResponse,
    SystemOneTemplateKind,
};
use lloom::nn::memory::LoadContext;

struct Metadata {
    text: Arc<Text>,
    think_tags: Option<super::super::template::ThinkTags>,
    device: Device,
    model_name: String,
    capacity: usize,
    moe: bool,
    readback: ReadbackArm,
    prefill_tile: usize,
    installed: super::super::report::Installed,
    no_workers: NoWorkerWitness,
}
/// The full model/vision/DSpark/cache stays exclusively inside Loaded, not copied into a
/// one-sequence replacement. Shared Text metadata exposes no tokenizer escape.
pub struct Engine<P: Lfm2Shape, Q> {
    metadata: Metadata,
    warnings: Vec<LoadWarning>,
    owner: Admitted<Q, super::Engine<P>>,
    // Same caller-owned template/calibration, never a model/tokenizer/policy copy.
    policy: Option<Arc<CausalPolicy>>,
    image_limits: Option<super::super::vision::ImageDecodeLimits>,
}
fn metadata<P: Lfm2Shape>(engine: &super::Engine<P>, no_workers: NoWorkerWitness) -> Metadata {
    Metadata {
        text: Arc::clone(&engine.text),
        // load_on primes both OnceLocks before its single-sequence return.
        // ThinkTags contains only scalar ids; this clone allocates no storage.
        think_tags: engine
            .text
            .think
            .get()
            .expect("full LFM factory primed thinking metadata")
            .clone(),
        device: engine.core.device.clone(),
        model_name: engine.model_name.clone(),
        capacity: engine.kv_capacity,
        moe: engine.config.moe.is_some(),
        readback: engine.readback,
        prefill_tile: engine.prefill_tile,
        installed: engine.installed.clone(),
        no_workers,
    }
}
/// Actual full-family factory in the same existing supplied load scope.
/// Language/DSpark and initial state are retained once; lazy vision tower loading
/// remains a request producer. Existing cheap-preflight ordering and cold host
/// metadata/Arc allocation/unwind custody are not complete admission proofs.
pub fn load<P: Lfm2Shape, Q: Predictor>(
    opts: LoadOptions,
    device: &Device,
    load: LoadContext,
    predictor: Q,
) -> Result<Engine<P, Q>> {
    load_with_optional_policy(opts, device, load, predictor, None, None)
}
/// Explicit borrowed-at-request raw policy on the same full owner. The Arc keeps
/// the caller's single compiled template/calibration alive; no default policy or
/// template certificate is invented. Prepared decisions remain available.
pub fn load_with_policy<P: Lfm2Shape, Q: Predictor>(
    opts: LoadOptions,
    device: &Device,
    load: LoadContext,
    predictor: Q,
    policy: Arc<CausalPolicy>,
) -> Result<Engine<P, Q>> {
    load_with_optional_policy(opts, device, load, predictor, Some(policy), None)
}
/// Additive image-capable raw policy on the same supplied full owner.
/// Limits do not certify complete decoder/native allocation enrollment.
pub fn load_with_image_policy<P: Lfm2Shape, Q: Predictor>(
    opts: LoadOptions,
    device: &Device,
    load: LoadContext,
    predictor: Q,
    policy: Arc<CausalPolicy>,
    limits: super::super::vision::ImageDecodeLimits,
) -> Result<Engine<P, Q>> {
    lloom_tensor::memory_admission::begin_model_load();
    limits.validate()?;
    load_with_optional_policy(opts, device, load, predictor, Some(policy), Some(limits))
}
fn load_with_optional_policy<P: Lfm2Shape, Q: Predictor>(
    opts: LoadOptions,
    device: &Device,
    load: LoadContext,
    predictor: Q,
    policy: Option<Arc<CausalPolicy>>,
    image_limits: Option<super::super::vision::ImageDecodeLimits>,
) -> Result<Engine<P, Q>> {
    let (loaded, metadata) = super::Engine::<P>::load_as_admitted_with_preflight(
        opts,
        device,
        load,
        |config, text| {
            if let Some(policy) = policy.as_deref() {
                if policy.template.kind() != SystemOneTemplateKind::D1Causal {
                    anyhow::bail!("full LFM causal policy needs a causal template kind");
                }
                // Include sparse/added ids, not just a vocabulary count.
                if text
                    .tokenizer
                    .get_vocab(true)
                    .values()
                    .any(|&id| id as usize >= config.vocab_size)
                {
                    anyhow::bail!("tokenizer id is outside resolved LFM vocabulary");
                }
            }
            Ok(())
        },
        metadata::<P>,
    )?;
    Ok(Engine {
        metadata,
        warnings: Vec::new(),
        owner: Admitted::new(loaded, predictor),
        policy,
        image_limits,
    })
}

impl<P: Lfm2Shape, Q: Predictor> Engine<P, Q> {
    fn pass<T>(
        &self,
        request: Request<'_>,
        run: impl FnOnce(&dyn Served) -> Result<T>,
    ) -> Result<T> {
        // Keep typed family errors inside the result; only admission is mapped.
        self.owner.pass(request, |engine| Ok(run(engine)))?
    }
    pub fn decide_prepared(&self, request: &CausalRequest) -> Result<SystemOneResponse> {
        let predicted = self.owner.predict(Request::PreparedDecisions(request))?;
        super::Engine::<P>::decide_admitted(&self.owner.loaded, predicted, request)
    }
    /// Ordinary caller template rendering remains usable without a state-prefix
    /// certificate; predictor sees the actual borrowed policy and request inputs.
    pub fn render_systemone(
        &self,
        id: &str,
        state: &OrderedJson,
        question: &crate::decision::Question<OrderedJson>,
        labels: &[String],
    ) -> Result<String> {
        let policy = self
            .policy
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("no explicit causal policy installed"))?;
        Ok(self.owner.pass(
            Request::PolicyRender {
                policy,
                id,
                state,
                question,
                labels,
            },
            |_| {
                Ok(policy
                    .template
                    .render(id, state, question, labels, SystemOneMedia::Text)?)
            },
        )?)
    }
    pub fn systemone_admitted(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let policy = self.policy.as_deref().ok_or_else(|| {
            SystemOneError::Unsupported("no explicit causal policy installed".into())
        })?;
        let predicted_request = match self.image_limits {
            Some(limits) => Request::PolicyImageDecisions {
                body,
                policy,
                limits,
            },
            None => Request::PolicyDecisions { body, policy },
        };
        self.owner.pass(predicted_request, |engine| {
            if !policy.template.has_causal_certificate() {
                return Ok(Err(SystemOneError::Unsupported("template has no audited causal certificate; ordinary rendering and prepared decisions remain available".into())));
            }
            if let Some(limits) = self.image_limits {
                return Ok(engine.systemone_images_in_scope(body, policy, limits));
            }
            Ok(policy.prepare_text(body, engine.kv_capacity, |text|
                engine.encode(text, false).map_err(lloom_tensor::Error::wrap))
                .and_then(|request| engine.decide_in_scope(&request)
                    .map_err(|error| SystemOneError::Execution(error.to_string()))))
        }).map_err(|error| SystemOneError::Execution(error.to_string()))?
    }

    pub fn complete_admitted(
        &self,
        prompt: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        self.pass(Request::Generation { prompt, params }, |engine| {
            engine.complete_until(prompt, params, on_fragment)
        })
    }
    pub fn health_checked(&self) -> Result<Value> {
        self.pass(Request::Metadata(MetadataRequest::Health), |engine| {
            engine.health_checked()
        })
    }
    pub fn describe_checked(&self) -> Result<String> {
        self.pass(Request::Metadata(MetadataRequest::Describe), |engine| {
            engine.describe_checked()
        })
    }
    pub fn load_warnings_checked(&self) -> Result<Vec<LoadWarning>> {
        self.pass(Request::Metadata(MetadataRequest::LoadWarnings), |engine| {
            let mut warnings = self.warnings.clone();
            warnings.extend(engine.load_warnings_checked()?);
            Ok(warnings)
        })
    }
    pub fn note_load_warnings_checked(&mut self, warnings: Vec<LoadWarning>) -> Result<()> {
        let predicted =
            self.owner
                .predict(Request::Metadata(MetadataRequest::NoteLoadWarnings(
                    &warnings,
                )))?;
        self.owner.loaded.with_pass(predicted, |_, _context| {
            self.warnings.splice(0..0, warnings);
            Ok(())
        })?;
        Ok(())
    }
}
impl<P: Lfm2Shape, Q: Predictor> Served for Engine<P, Q> {
    fn device(&self) -> &Device {
        &self.metadata.device
    }
    fn installed(&self) -> Option<&super::super::report::Installed> {
        Some(&self.metadata.installed)
    }
    fn model_name(&self) -> &str {
        &self.metadata.model_name
    }
    fn kv_capacity(&self) -> usize {
        self.metadata.capacity
    }
    fn readback(&self) -> ReadbackArm {
        self.metadata.readback
    }
    fn prefill_tile(&self) -> usize {
        self.metadata.prefill_tile
    }
    fn batch(&self) -> usize {
        1
    }
    fn eos_ids(&self) -> &[u32] {
        &self.metadata.text.eos_ids
    }
    fn bos_id(&self) -> Option<u32> {
        self.metadata.text.bos_id
    }
    fn health_checked(&self) -> Result<Value> {
        Engine::health_checked(self)
    }
    fn describe_checked(&self) -> Result<String> {
        Engine::describe_checked(self)
    }
    fn load_warnings_checked(&self) -> Result<Vec<LoadWarning>> {
        Engine::load_warnings_checked(self)
    }
    fn note_load_warnings_checked(&mut self, warnings: Vec<LoadWarning>) -> Result<()> {
        Engine::note_load_warnings_checked(self, warnings)
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
    fn capabilities(&self) -> super::super::family::Capabilities {
        // Installed adapter support, not the broader legacy family surface.
        super::super::family::Capabilities {
            batch: false,
            prefix_share: false,
            decisions: self
                .policy
                .as_deref()
                .is_some_and(|policy| policy.template.has_causal_certificate()),
            // Preserve the dense adapter's existing metadata contract.
            ..if self.metadata.moe {
                super::super::family::LFM2_MOE_CAPABILITIES
            } else {
                super::super::family::LFM2_CAPABILITIES
            }
        }
    }
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        self.systemone_admitted(body)
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        self.load_warnings_checked().expect(
            "legacy infallible warnings cannot report admission refusal; use load_warnings_checked",
        )
    }
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>) {
        self.note_load_warnings_checked(warnings).expect("legacy infallible warning insertion cannot report admission refusal; use note_load_warnings_checked");
    }
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        self.pass(Request::Encode { text, add_special }, |engine| {
            engine.encode(text, add_special)
        })
    }
    fn token_str(&self, id: u32) -> String {
        self.token_str_checked(id).expect("legacy infallible token display cannot report admission refusal; use token_str_checked")
    }
    fn token_str_checked(&self, id: u32) -> Result<String> {
        self.pass(Request::Token(id), |engine| engine.token_str_checked(id))
    }
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        self.pass(
            Request::Render {
                messages,
                kwargs,
                checked,
            },
            |engine| engine.render_checked(messages, kwargs, checked),
        )
    }
    fn template_reads(&self, name: &str) -> Option<bool> {
        self.metadata.text.template.as_ref()?.reads(name)
    }
    fn think_tags(&self) -> Option<super::super::template::ThinkTags> {
        self.metadata.think_tags.clone()
    }
    fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<Vec<u32>> {
        self.generation_prompt_checked(kwarg, want).expect("legacy infallible prompt metadata cannot report admission refusal; use generation_prompt_checked")
    }
    fn generation_prompt_checked(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
    ) -> Result<Option<Vec<u32>>> {
        // Borrow only the cold-prepared ids before predicting. The returned
        // Vec is cloned by the actual inner method inside the admitted pass.
        let prompts = self
            .metadata
            .text
            .gen_prompts
            .get()
            .expect("full LFM factory primed generation prompts");
        let ids = prompts[super::super::common::thinking_mode(kwarg)]
            [super::super::common::thinking_mode(want)]
        .as_deref();
        self.pass(Request::GenerationPrompt { kwarg, want, ids }, |engine| {
            engine.generation_prompt_checked(kwarg, want)
        })
    }
    fn template_renders(&self) -> u64 {
        self.metadata
            .text
            .template
            .as_ref()
            .map_or(0, Template::renders)
    }
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32> {
        self.seed_prompt_row_checked(n, row).expect(
            "legacy infallible seed cannot report admission refusal; use seed_prompt_row_checked",
        )
    }
    fn seed_prompt_row_checked(&self, n: usize, row: usize) -> Result<Vec<u32>> {
        super::super::common::check_context_capacity(self.kv_capacity(), n, 0, 0)?;
        self.pass(Request::Seed { positions: n, row }, |engine| {
            engine.seed_prompt_row_checked(n, row)
        })
    }
    fn complete_until(
        &self,
        prompt: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        self.complete_admitted(prompt, params, on_fragment)
    }
    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming> {
        self.check_batch(batch)?;
        self.pass(Request::Prefill { positions: n }, |engine| {
            engine.prefill_throughput(n, batch)
        })
    }
    fn decode_throughput(
        &self,
        prompt: usize,
        generated: usize,
        batch: usize,
    ) -> Result<DecodeTiming> {
        self.check_batch(batch)?;
        self.pass(Request::Decode { prompt, generated }, |engine| {
            engine.decode_throughput(prompt, generated, batch)
        })
    }
    fn next_token_logprobs(&self, prompt: &[u32], k: usize) -> Result<(u32, Vec<(u32, f32)>)> {
        self.pass(Request::Logprobs { prompt, k }, |engine| {
            engine.next_token_logprobs(prompt, k)
        })
    }
    fn generation_config(&self) -> Option<lloom::hub::GenerationConfig> {
        self.generation_config_checked().expect("legacy infallible generation metadata cannot report admission refusal; use generation_config_checked")
    }
    fn generation_config_checked(&self) -> Result<Option<lloom::hub::GenerationConfig>> {
        self.pass(
            Request::Metadata(MetadataRequest::GenerationConfig),
            |engine| engine.generation_config_checked(),
        )
    }
    fn image_placeholder(&self, image: &DecodedImage) -> Result<String> {
        self.pass(Request::ImagePlaceholder(image), |engine| {
            engine.image_placeholder(image)
        })
    }
    fn complete_images_until(
        &self,
        prompt: &[u32],
        images: &[DecodedImage],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        self.pass(
            Request::Images {
                prompt,
                images,
                params,
            },
            |engine| engine.complete_images_until(prompt, images, params, on_fragment),
        )
    }
    fn vision_report(&self) -> Value {
        self.vision_report_checked().expect("legacy infallible vision report cannot report admission refusal; use vision_report_checked")
    }
    fn vision_report_checked(&self) -> Result<Value> {
        self.pass(Request::Metadata(MetadataRequest::VisionReport), |engine| {
            engine.vision_report_checked()
        })
    }
    fn diagnostics(&self) -> Vec<String> {
        self.diagnostics_checked().expect("legacy infallible diagnostics cannot report admission refusal; use diagnostics_checked")
    }
    fn diagnostics_checked(&self) -> Result<Vec<String>> {
        self.pass(Request::Metadata(MetadataRequest::Diagnostics), |engine| {
            engine.diagnostics_checked()
        })
    }
    fn run_mode(&self, spec: &super::super::modes::ModeSpec) -> Result<()> {
        self.pass(Request::Mode(spec), |engine| engine.run_mode(spec))
    }
    fn check_batch(&self, batch: usize) -> Result<()> {
        if batch != 1 {
            anyhow::bail!("admitted LFM2 serves exactly one sequence, requested {batch}")
        }
        Ok(())
    }
    /// The cold witness checked this owner's scheduler/thread are absent.
    /// Same worker-stop meaning as incumbent shutdown, not native DATA,
    /// callback return, model destruction or complete retirement closure.
    fn shutdown(&self) {
        self.metadata.no_workers.shutdown();
    }
}
