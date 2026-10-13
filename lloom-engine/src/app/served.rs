//! The request seam the server and the chat REPL speak: one trait per served
//! family, `dyn` at the request level (RESTRUCTURING §4, `DESIGN.md` §15.2
//! #9 — one indirect call per request; the per-token loop behind each method
//! is `DecodeSession`'s, generic and monomorphized).
//!
//! Two implementors: the LFM2 engine (`lfm2.rs`, every arm the serving
//! binary had — the crown ring, DSpark, `--batch N`) and the Qwen4-exp
//! engine (`qwen4exp.rs`, the single-sequence path over the resident or the
//! streamed expert tier, MTP behind `--mtp`). A family that cannot serve a
//! shape refuses it with the reason (`check_batch`), never serves it wrong.

use super::common::{DecodeTiming, PrefillTiming};
use super::template::Rendered;
use super::vision::DecodedImage;
use crate::{DecodeParams, Decoded, ReadbackArm};
use anyhow::Result;
use lloom::load_warning::LoadWarning;
use lloom_tensor::Device;
use serde_json::{Map, Value};

pub trait Served: Send + Sync {
    /// The device the model decodes on.
    fn device(&self) -> &Device;
    /// The `model` id `/v1/models` reports.
    fn model_name(&self) -> &str;
    /// KV positions per request; a request past it is refused.
    fn kv_capacity(&self) -> usize;
    /// The readback arm requests take (resolved per family: the LFM2 crown
    /// ring under `auto`; the blocking arm on Qwen4-exp, whose forward needs
    /// its input ids host-side).
    fn readback(&self) -> ReadbackArm;
    fn prefill_tile(&self) -> usize;
    /// The batch width `N`; `1` is the single-sequence path.
    fn batch(&self) -> usize;
    /// The ids a run stops on (the checkpoint's EOS set).
    fn eos_ids(&self) -> &[u32];
    /// The BOS the template starts with, if the checkpoint has one.
    fn bos_id(&self) -> Option<u32>;
    /// The family-specific half of `/health` (tile, chunk, rows, the pool,
    /// the streamed tier's counters …).
    fn health(&self) -> Value;
    /// Fallible allocating metadata seam. Incumbent families retain their
    /// behavior; admitted adapters override before any request producer.
    fn health_checked(&self) -> Result<Value> {
        Ok(self.health())
    }
    /// The family-specific tail of the `[loaded …]` line.
    fn describe(&self) -> String;
    fn describe_checked(&self) -> Result<String> {
        Ok(self.describe())
    }
    /// What the family can do (`family.rs`): the loaded line and `/health` say it.
    fn capabilities(&self) -> super::family::Capabilities;
    /// Every load-time degradation, in the order found: the notes recorded
    /// before the family loaded, the family's own, then each emulated type
    /// the device has admitted (`lloom::load_warning`). The loaded line,
    /// `/health` and the model card render it.
    fn load_warnings(&self) -> Vec<LoadWarning>;
    fn load_warnings_checked(&self) -> Result<Vec<LoadWarning>> {
        Ok(self.load_warnings())
    }
    /// Record the warnings found before the family loaded (the readback
    /// clamps `load_engine` applies), ahead of the family's own.
    fn note_load_warnings(&mut self, warnings: Vec<LoadWarning>);
    fn note_load_warnings_checked(&mut self, warnings: Vec<LoadWarning>) -> Result<()> {
        self.note_load_warnings(warnings);
        Ok(())
    }
    /// The checkpoint's `generation_config.json`, when it carries one (the
    /// sampling defaults under `--sampling-defaults checkpoint`).
    fn generation_config(&self) -> Option<lloom::hub::GenerationConfig> {
        None
    }
    fn generation_config_checked(&self) -> Result<Option<lloom::hub::GenerationConfig>> {
        Ok(self.generation_config())
    }

    /// Decide without generating tokens, using this already-loaded model.
    /// Raw ordered state/questions reach the shared decision frontend; families
    /// supply execution, not copies of rendering/readout. The default refusal is
    /// transitional until every causal family implements retained-trunk execution.
    fn systemone(
        &self,
        _body: &crate::decision::OrderedJson,
    ) -> std::result::Result<crate::decision::SystemOneResponse, crate::decision::SystemOneError>
    {
        Err(crate::decision::SystemOneError::Unsupported(format!(
            "{} has no installed decision executor",
            self.model_name()
        )))
    }

    /// [`Self::systemone`] for requests that arrived together (`serve`'s
    /// decision group). A family that reads single-question requests as one
    /// tree packs them; the default answers them one by one.
    fn systemone_batch(
        &self,
        bodies: &[&crate::decision::OrderedJson],
    ) -> Vec<std::result::Result<crate::decision::SystemOneResponse, crate::decision::SystemOneError>>
    {
        bodies.iter().map(|body| self.systemone(body)).collect()
    }

    /// The template binding this family's decision executor was built with:
    /// the native decision type the checkpoint is (`None` for a template
    /// `decision.template` selected) and where the template came from.
    /// `None` where the family does not say (an owner bound elsewhere).
    fn decision_binding(&self) -> Option<(Option<crate::decision::binding::DecisionType>, String)> {
        None
    }

    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>>;
    fn token_str(&self, id: u32) -> String;
    /// Fallible request seam for admitted adapters; existing families preserve
    /// their incumbent behavior through this compatibility default.
    fn token_str_checked(&self, id: u32) -> Result<String> {
        Ok(self.token_str(id))
    }
    /// Render `messages` through the checkpoint's own chat template with the
    /// generation prompt appended, `kwargs` laid over its variables and each
    /// of `checked` tested against it (`template::Template::render_checked`).
    /// A value the template rejects is a `template::Rejected`.
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered>;
    /// [`Self::render_checked`] with nothing checked.
    fn render_chat(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<String> {
        Ok(self.render_checked(messages, kwargs, &[])?.text)
    }
    /// Whether the chat template reads `name`; `None` without a template, or
    /// when its source could not be analysed.
    fn template_reads(&self, name: &str) -> Option<bool>;
    /// The chat template's reasoning tags (`template::ThinkTags`); `None`
    /// when it writes no thinking block.
    fn think_tags(&self) -> Option<super::template::ThinkTags> {
        None
    }
    /// The ids a prompt rendered with `enable_thinking` = `kwarg` and the
    /// engine's thinking switch `want` ends with — the template's
    /// generation prompt, read at load (`common::Text::generation_prompt`):
    /// where the turn checkpoint is placed. `None` without a template.
    fn generation_prompt(&self, _kwarg: Option<bool>, _want: Option<bool>) -> Option<Vec<u32>> {
        None
    }
    /// Fallible prompt-copy seam for admitted adapters. None denotes actual
    /// absent/unreadable metadata, never an admission refusal. Existing families
    /// retain their incumbent load-time cache through this default.
    fn generation_prompt_checked(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
    ) -> Result<Option<Vec<u32>>> {
        Ok(self.generation_prompt(kwarg, want))
    }
    /// The renders the chat template has run, for a per-turn count.
    fn template_renders(&self) -> u64 {
        0
    }
    /// Why this process's decode path cannot feed a thinking budget's
    /// forced tokens (`budgets::BATCHED`, `budgets::SPECULATIVE`); `None`
    /// on the K=1 body, which carries it on every readback arm.
    fn thinking_budget_refused(&self) -> Option<&'static str> {
        None
    }
    /// The benchmark seed tiled to `n` ids, `row` seed tokens in.
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32>;
    /// Admission refusal must not masquerade as successful empty prompt data.
    fn seed_prompt_row_checked(&self, n: usize, row: usize) -> Result<Vec<u32>> {
        Ok(self.seed_prompt_row(n, row))
    }

    /// One decode over the whole token sequence, streaming fragments; the
    /// hook's `false` ends the run (`Stop::Cancelled`).
    fn complete_until(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded>;

    fn complete(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) + Send),
    ) -> Result<Decoded> {
        self.complete_until(prompt_ids, params, &mut |frag| {
            on_fragment(frag);
            true
        })
    }

    /// The text one attached image's `<image>` expands to (its delimiters, its
    /// tiles' token runs). A family or checkpoint without vision refuses.
    fn image_placeholder(&self, image: &DecodedImage) -> Result<String> {
        let _ = image;
        Err(self.no_vision())
    }

    /// [`Self::complete_until`] for a prompt whose image positions take the
    /// `images`' rows, in order. The same refusal as `image_placeholder`.
    fn complete_images_until(
        &self,
        prompt_ids: &[u32],
        images: &[DecodedImage],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        let _ = (prompt_ids, images, params, on_fragment);
        Err(self.no_vision())
    }

    /// What the last request loaded and encoded: the model's tensors, whether
    /// the vision tower is resident, the image census. `Null` for a family
    /// without vision.
    fn vision_report(&self) -> Value {
        Value::Null
    }
    fn vision_report_checked(&self) -> Result<Value> {
        Ok(self.vision_report())
    }

    /// The refusal a family without vision gives, by the family's own reason.
    fn no_vision(&self) -> anyhow::Error {
        let c = self.capabilities();
        anyhow::anyhow!(
            "{} does not take image input: {}",
            c.family,
            c.vision_missing
        )
    }

    /// The executor arms armed at load, for their engagement reports
    /// (`report::print_executor`); `None` where nothing was armed.
    fn installed(&self) -> Option<&super::report::Installed> {
        None
    }

    /// The family's lines for a verbose run report: the state a turn left
    /// behind that only the family can read (the held sequence's KV, the
    /// pacer's depth against what warmup measured).
    fn diagnostics(&self) -> Vec<String> {
        Vec::new()
    }
    fn diagnostics_checked(&self) -> Result<Vec<String>> {
        Ok(self.diagnostics())
    }

    /// The multi-sequence measurement modes (`lloom bench --mode reuse |
    /// concurrent`): built where the family can drive several sequences of
    /// one model directly; refused by name elsewhere.
    fn run_mode(&self, spec: &super::modes::ModeSpec) -> Result<()> {
        anyhow::bail!(
            "{} has no `lloom bench --mode {}`: the reuse and concurrent modes drive LFM2's \
             sessions, caches and pools directly (its `Served` is the only one that builds them)",
            self.capabilities().family,
            spec.mode.name()
        )
    }

    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming>;
    fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming>;
    fn next_token_logprobs(&self, prompt_ids: &[u32], k: usize) -> Result<(u32, Vec<(u32, f32)>)>;
    /// The full-vocabulary log-probability of each of `labels` as the token
    /// after `prompt_ids`, in their order, and the most likely token there.
    /// This default reads the whole distribution through
    /// [`Self::next_token_logprobs`]; a family may read only the labels.
    fn first_token_label_read(
        &self,
        prompt_ids: &[u32],
        labels: &[u32],
    ) -> Result<(Vec<f32>, u32)> {
        let (argmax, all) = self.next_token_logprobs(prompt_ids, usize::MAX)?;
        let mut out = vec![f32::NAN; labels.len()];
        for (id, logprob) in all {
            for (slot, &label) in out.iter_mut().zip(labels) {
                if label == id {
                    *slot = logprob;
                }
            }
        }
        if out.iter().any(|value| value.is_nan()) {
            anyhow::bail!("the next-token distribution has no entry for a decision label")
        }
        Ok((out, argmax))
    }
    /// A cell's `batch` must fit what the process was started with.
    fn check_batch(&self, batch: usize) -> Result<()>;
    /// Stop whatever the engine runs on its own threads.
    fn shutdown(&self);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::app::family::QWEN4EXP_CAPABILITIES;

    /// A family with no vision, reached only through the seam: every method
    /// the image path could call past the refusal panics.
    pub(crate) struct NoVision {
        device: Device,
        metadata_mode: u8,
        pub(crate) warning_epoch: std::sync::atomic::AtomicUsize,
        pub(crate) metadata_calls: std::sync::atomic::AtomicUsize,
        pub(crate) calls: std::sync::atomic::AtomicUsize,
        handler: Option<
            fn(
                &crate::decision::OrderedJson,
            ) -> std::result::Result<
                crate::decision::SystemOneResponse,
                crate::decision::SystemOneError,
            >,
        >,
    }
    impl NoVision {
        pub(crate) fn new(
            handler: Option<
                fn(
                    &crate::decision::OrderedJson,
                ) -> std::result::Result<
                    crate::decision::SystemOneResponse,
                    crate::decision::SystemOneError,
                >,
            >,
        ) -> Self {
            Self {
                device: Device::Cpu,
                metadata_mode: 0,
                warning_epoch: std::sync::atomic::AtomicUsize::new(0),
                metadata_calls: std::sync::atomic::AtomicUsize::new(0),
                calls: std::sync::atomic::AtomicUsize::new(0),
                handler,
            }
        }
        /// Same inert fixture; nonzero modes make any legacy metadata call
        /// fail so the actual consumer must select the fallible seam.
        pub(crate) fn refusing_metadata() -> Self {
            let mut engine = Self::new(None);
            engine.metadata_mode = 1;
            engine
        }
        pub(crate) fn dynamic_metadata() -> Self {
            let mut engine = Self::new(None);
            engine.metadata_mode = 2;
            engine
        }
    }

    impl Served for NoVision {
        fn systemone(
            &self,
            body: &crate::decision::OrderedJson,
        ) -> std::result::Result<crate::decision::SystemOneResponse, crate::decision::SystemOneError>
        {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match self.handler {
                Some(handler) => handler(body),
                None => Err(crate::decision::SystemOneError::Unsupported(
                    "stub has no installed decision executor".into(),
                )),
            }
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn model_name(&self) -> &str {
            "stub"
        }
        fn kv_capacity(&self) -> usize {
            0
        }
        fn readback(&self) -> ReadbackArm {
            ReadbackArm::Blocking
        }
        fn prefill_tile(&self) -> usize {
            0
        }
        fn batch(&self) -> usize {
            1
        }
        fn eos_ids(&self) -> &[u32] {
            &[]
        }
        fn bos_id(&self) -> Option<u32> {
            None
        }
        fn health(&self) -> Value {
            assert_eq!(self.metadata_mode, 0, "legacy health fallback reached");
            Value::Null
        }
        fn health_checked(&self) -> Result<Value> {
            if self.metadata_mode == 0 {
                return Ok(self.health());
            }
            self.metadata_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            anyhow::bail!("bounded metadata health refusal")
        }
        fn describe(&self) -> String {
            String::new()
        }
        fn capabilities(&self) -> super::super::family::Capabilities {
            QWEN4EXP_CAPABILITIES
        }
        fn load_warnings(&self) -> Vec<LoadWarning> {
            assert_eq!(self.metadata_mode, 0, "legacy warning fallback reached");
            Vec::new()
        }
        fn load_warnings_checked(&self) -> Result<Vec<LoadWarning>> {
            if self.metadata_mode == 0 {
                return Ok(self.load_warnings());
            }
            self.metadata_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.metadata_mode == 1 {
                anyhow::bail!("bounded metadata warning refusal");
            }
            let epoch = self
                .warning_epoch
                .load(std::sync::atomic::Ordering::Relaxed);
            Ok(vec![LoadWarning::new(
                lloom::load_warning::LoadWarningKind::EmulatedType,
                format!("bounded dynamic warning epoch {epoch}"),
            )])
        }
        fn vision_report(&self) -> Value {
            assert_eq!(
                self.metadata_mode, 0,
                "legacy vision metadata fallback reached"
            );
            Value::Null
        }
        fn vision_report_checked(&self) -> Result<Value> {
            if self.metadata_mode == 0 {
                return Ok(self.vision_report());
            }
            self.metadata_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            anyhow::bail!("bounded metadata vision refusal")
        }
        fn note_load_warnings(&mut self, _: Vec<LoadWarning>) {}
        fn encode(&self, _: &str, _: bool) -> Result<Vec<u32>> {
            unreachable!("the refusal comes first")
        }
        fn token_str(&self, _: u32) -> String {
            unreachable!()
        }
        fn render_checked(
            &self,
            _: &Value,
            _: &Map<String, Value>,
            _: &[&str],
        ) -> Result<Rendered> {
            unreachable!()
        }
        fn template_reads(&self, _: &str) -> Option<bool> {
            None
        }
        fn seed_prompt_row(&self, _: usize, _: usize) -> Vec<u32> {
            unreachable!()
        }
        fn complete_until(
            &self,
            _: &[u32],
            _: &DecodeParams,
            _: &mut (dyn FnMut(&str) -> bool + Send),
        ) -> Result<Decoded> {
            unreachable!("an image request must not reach the text path")
        }
        fn prefill_throughput(&self, _: usize, _: usize) -> Result<PrefillTiming> {
            unreachable!()
        }
        fn decode_throughput(&self, _: usize, _: usize, _: usize) -> Result<DecodeTiming> {
            unreachable!()
        }
        fn next_token_logprobs(&self, _: &[u32], _: usize) -> Result<(u32, Vec<(u32, f32)>)> {
            unreachable!()
        }
        fn check_batch(&self, _: usize) -> Result<()> {
            unreachable!()
        }
        fn shutdown(&self) {}
    }

    /// A family without vision refuses an image at the seam, by its own
    /// reason, before anything is rendered or decoded.
    #[test]
    fn a_family_without_vision_refuses_an_image_by_name() {
        let engine = NoVision::new(None);
        let image = DecodedImage::from_rgb8(1, 1, vec![0, 0, 0]);
        let want = "qwen4exp does not take image input: no vision tower is built for Qwen4-exp";
        let err = engine.image_placeholder(&image).unwrap_err().to_string();
        assert!(err.starts_with(want), "{err}");
        let err = engine
            .complete_images_until(&[1, 2], &[image], &DecodeParams::default(), &mut |_| true)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with(want), "{err}");
        assert!(engine.vision_report().is_null());
    }
}
