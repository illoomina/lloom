//! Same full Qwen owner for supplied-domain generation and prepared decisions.
//! Caller prediction is a coverage contract, not proof of producer enrollment.
//! Legacy infallible metadata refusal policy and default registration stay open.
use super::*;
use crate::app::causal_loaded::served::{Admitted, MetadataRequest, Predictor, Request};
use crate::decision::{CausalRequest, SystemOneResponse};
use lloom::nn::memory::LoadContext;

struct Metadata {
    text: Arc<Text>,
    device: Device,
    model_name: String,
    capacity: usize,
    readback: ReadbackArm,
    prefill_tile: usize,
    installed: super::super::report::Installed,
    mtp: bool,
}
/// The full model/MTP/cache stays exclusively inside Loaded, not copied into a
/// one-sequence replacement. Shared Text metadata exposes no tokenizer escape.
pub struct Engine<P: Profile, Q> {
    metadata: Metadata,
    warnings: Vec<LoadWarning>,
    owner: Admitted<Q, Engines<P>>,
}
fn metadata<M, X>(engine: &super::Engine<M, X>) -> Metadata
where
    M: Qwen3_5Model,
    M::Cache: CacheReset + CacheRewind + Send + Sync + 'static,
    X: Layer + 'static,
    X::State: Send + Sync,
{
    Metadata {
        text: Arc::clone(&engine.text),
        device: engine.device.clone(),
        model_name: engine.model_name.clone(),
        capacity: engine.kv_capacity,
        readback: engine.readback,
        prefill_tile: engine.prefill_tile,
        installed: engine.installed.clone(),
        mtp: engine.mtp.is_some(),
    }
}
/// Actual full-family factory: one existing supplied load scope covers header,
/// cheap preflight, weights/MTP, initial state and shared serving metadata.
/// Cold host metadata/Arc custody is explicitly not implicitly enrolled.
pub fn load<P: Profile, Q: Predictor>(
    opts: LoadOptions,
    device: &Device,
    load: LoadContext,
    predictor: Q,
) -> Result<Engine<P, Q>> {
    let (loaded, metadata) = load_admitted_with::<P, _>(opts, device, load, |engine| {
        Ok(match engine {
            Engines::Dense(engine) => metadata(engine),
            Engines::Moe(engine) => metadata(engine),
        })
    })?;
    Ok(Engine {
        metadata,
        warnings: Vec::new(),
        owner: Admitted::new(loaded, predictor),
    })
}
impl<P: Profile> Engines<P> {
    fn served(&self) -> &dyn Served {
        match self {
            Self::Dense(engine) => engine,
            Self::Moe(engine) => engine,
        }
    }
}
impl<P: Profile, Q: Predictor> Engine<P, Q> {
    fn pass<T>(
        &self,
        request: Request<'_>,
        run: impl FnOnce(&dyn Served) -> Result<T>,
    ) -> Result<T> {
        // Keep typed family errors inside the result; only admission is mapped.
        self.owner
            .pass(request, |engine| Ok(run(engine.served())))?
    }
    pub fn decide_prepared(&self, request: &CausalRequest) -> Result<SystemOneResponse> {
        let predicted = self.owner.predict(Request::PreparedDecisions(request))?;
        Engines::decide_admitted(&self.owner.loaded, predicted, request)
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
impl<P: Profile, Q: Predictor> Served for Engine<P, Q> {
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
        None
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
        super::super::family::QWEN3_5_CAPABILITIES
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
        self.metadata.text.think_tags().cloned()
    }
    fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<Vec<u32>> {
        self.generation_prompt_checked(kwarg, want).expect("legacy infallible prompt metadata cannot report admission refusal; use generation_prompt_checked")
    }
    fn generation_prompt_checked(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
    ) -> Result<Option<Vec<u32>>> {
        let ids = self.metadata.text.generation_prompt(kwarg, want);
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
    fn thinking_budget_refused(&self) -> Option<&'static str> {
        self.metadata
            .mtp
            .then_some(super::super::budgets::SPECULATIVE)
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
    fn check_batch(&self, batch: usize) -> Result<()> {
        if batch != 1 {
            anyhow::bail!("Qwen3.5 serves exactly one sequence, requested {batch}")
        }
        Ok(())
    }
    fn shutdown(&self) {} // The incumbent full Qwen owner launches no worker.
}
