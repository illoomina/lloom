//! `/v1/systemone`'s binding on a served model, resolved once at load from
//! `decision.template` and what the checkpoint carries
//! (`decision::binding`). A family executor answers for the binding it was
//! built with; the generic strategy runs here, over any generating family,
//! through the `Served` seam alone.
use super::served::Served;
use crate::decision::binding::{self, CheckpointDecision, DecisionType, Selection};
use crate::decision::generic::{self, GenericLabels, LabelRead};
use crate::decision::{
    parse_systemone_questions, parse_systemone_state, NoulAliases, OrderedJson, StateNullability,
    SystemOneError, SystemOneResponse,
};
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::path::Path;

pub(crate) enum Decisions {
    /// The family's own executor, built with its binding.
    Family {
        native: Option<DecisionType>,
    },
    Generic(Generic),
    /// No binding: the refusal names why and what would give one.
    Refused(String),
}

pub(crate) struct Generic {
    labels: GenericLabels,
    kwargs: Map<String, Value>,
}

impl Decisions {
    /// `selection` is `decision.template`; `checkpoint` the path the family
    /// loaded, `None` for an owner loaded elsewhere (its binding is its own).
    pub(crate) fn resolve(
        selection: Option<&str>,
        engine: &dyn Served,
        checkpoint: Option<&Path>,
    ) -> Result<Self> {
        let selection = Selection::parse(selection)?;
        let facts = match checkpoint {
            Some(path) => checkpoint_decision(path)
                .with_context(|| format!("reading {}'s decision metadata", path.display()))?,
            None => CheckpointDecision::default(),
        };
        let capabilities = engine.capabilities();
        let family = capabilities.family;
        Ok(match selection {
            Selection::Generic => {
                if !capabilities.generation {
                    anyhow::bail!(
                        "decision.template=generic reads a reply's first token; {family} does not generate"
                    )
                }
                Self::Generic(Generic::probe(engine)?)
            }
            Selection::Auto if capabilities.decisions => Self::Family {
                native: facts.decision_type,
            },
            Selection::Auto => match facts.decision_type {
                Some(native) => Self::Refused(format!(
                    "{} is a native {} decision checkpoint, but the {family} family serving it has no \
                     decision executor installed; decision.template=generic reads it as a plain \
                     causal model instead (uncalibrated)",
                    engine.model_name(),
                    native.name()
                )),
                None => Self::Refused(match binding::no_binding(engine.model_name()) {
                    SystemOneError::Unsupported(message) => message,
                    other => other.to_string(),
                }),
            },
            Selection::Shipped(_) | Selection::Path(_) => anyhow::bail!(
                "decision.template names a systemone template, which only a family decision executor \
                 renders, and {family} has none installed; available here: {}",
                binding::GENERIC
            ),
        })
    }

    /// The load line, once.
    pub(crate) fn describe(&self) -> Option<String> {
        match self {
            Self::Family {
                native: Some(native),
            } => Some(format!("decision template: {} (native)", native.name())),
            Self::Family { native: None } => Some("decision template: the owner's".into()),
            Self::Generic(_) => Some("decision template: generic (uncalibrated)".into()),
            Self::Refused(_) => None,
        }
    }

    /// `/v1/models`' `output_modalities`: decisions are advertised for a
    /// native binding only (llama.cpp's shape).
    pub(crate) fn output_modalities(&self, generates: bool) -> Vec<&'static str> {
        let mut out = Vec::with_capacity(2);
        match self {
            Self::Family {
                native: Some(native),
            } => {
                if native.generates() && generates {
                    out.push("text");
                }
                out.push("decisions");
            }
            _ => out.push("text"),
        }
        out
    }

    pub(crate) fn decide(
        &self,
        engine: &dyn Served,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        match self {
            Self::Family { .. } => engine.systemone(body),
            Self::Generic(generic) => generic.decide(engine, body),
            Self::Refused(reason) => Err(SystemOneError::Unsupported(reason.clone())),
        }
    }
}

/// GGUF metadata or a safetensors `config.json`; anything else carries none.
fn checkpoint_decision(path: &Path) -> Result<CheckpointDecision> {
    if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
        let mut file = std::fs::File::open(path)?;
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut file)?;
        return Ok(CheckpointDecision::from_gguf(&content.metadata)?);
    }
    let config = path.join("config.json");
    if path.is_dir() && config.is_file() {
        let value: Value = serde_json::from_slice(&std::fs::read(&config)?)?;
        return Ok(CheckpointDecision::from_config(&value)?);
    }
    Ok(CheckpointDecision::default())
}

fn invalid(error: impl std::fmt::Display) -> SystemOneError {
    SystemOneError::InvalidRequest(error.to_string())
}

impl Generic {
    /// The label tokens, read in the context they will be read in: a probe
    /// conversation rendered to its generation prompt with thinking off.
    /// Refused by name when thinking cannot be turned off or a label is not
    /// one distinct token there.
    fn probe(engine: &dyn Served) -> Result<Self> {
        let mut kwargs = Map::new();
        if engine.template_reads(super::template::ENABLE_THINKING) == Some(true) {
            kwargs.insert(super::template::ENABLE_THINKING.into(), Value::Bool(false));
        }
        let render = |content: &str| -> Result<Vec<u32>> {
            let text =
                engine.render_chat(&json!([{"role": "user", "content": content}]), &kwargs)?;
            engine.encode(&text, true)
        };
        let first = render("lloom decision probe")?;
        let second = render("another probe, 7")?;
        if let Some(tags) = engine.think_tags() {
            if !kwargs.contains_key(super::template::ENABLE_THINKING) {
                anyhow::bail!(
                    "{}: the chat template writes a thinking block and does not read enable_thinking, \
                     so thinking cannot be turned off for a first-token read",
                    engine.model_name()
                )
            }
            if tags.open_at_end(&first).is_some() {
                anyhow::bail!(
                    "{}: the chat template leaves a thinking block open with enable_thinking=false",
                    engine.model_name()
                )
            }
        }
        // The ids every conversation ends with: the generation prompt.
        let shared = first
            .iter()
            .rev()
            .zip(second.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let tail = first[first.len() - shared..].to_vec();
        let text = engine.render_chat(
            &json!([{"role": "user", "content": "lloom decision probe"}]),
            &kwargs,
        )?;
        let labels = GenericLabels::probe(&text, &first, tail, |text| {
            engine.encode(text, true).map_err(lloom_tensor::Error::wrap)
        })
        .map_err(|error| anyhow::anyhow!("{}: {error}", engine.model_name()))?;
        Ok(Self { labels, kwargs })
    }

    fn decide(
        &self,
        engine: &dyn Served,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let state = parse_systemone_state(body, StateNullability::RequireValue).map_err(invalid)?;
        if !state.media.is_empty() {
            return Err(SystemOneError::Unsupported(
                "the generic decision template reads text only".into(),
            ));
        }
        let instructions = match body.get("instructions") {
            None | Some(OrderedJson::Null) => None,
            Some(OrderedJson::String(text)) if text.is_empty() => None,
            Some(OrderedJson::String(text)) => Some(text.as_str()),
            Some(_) => return Err(invalid("\"instructions\" must be a string")),
        };
        let questions =
            parse_systemone_questions(body, generic::MAX_OPTIONS, NoulAliases::TrueFalseOnly)
                .map_err(invalid)?;
        let execution = |error: anyhow::Error| SystemOneError::Execution(format!("{error:#}"));
        let mut reads = Vec::with_capacity(questions.len());
        for (_, question) in &questions {
            let user = generic::user_text(&state.state, question).map_err(invalid)?;
            let mut messages = Vec::with_capacity(2);
            if let Some(instructions) = instructions {
                messages.push(json!({"role": "system", "content": instructions}));
            }
            messages.push(json!({"role": "user", "content": user}));
            let text = engine
                .render_checked(&Value::Array(messages), &self.kwargs, &[])
                .map_err(|error| invalid(format!("{error:#}")))?
                .text;
            let ids = engine.encode(&text, true).map_err(execution)?;
            if !ids.ends_with(self.labels.tail()) {
                return Err(invalid(
                    "this request's chat prompt ends differently from the generation prompt the \
                     labels were checked against",
                ));
            }
            let labels = self.labels.ids(question).map_err(invalid)?;
            let (logprobs, argmax) = engine
                .first_token_label_read(&ids, labels)
                .map_err(execution)?;
            reads.push(LabelRead {
                logprobs,
                argmax,
                input_tokens: ids.len() as u64,
            });
        }
        generic::answers(&questions, &reads, &self.labels)
            .map_err(|error| SystemOneError::Execution(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    //! A toy family behind the `Served` seam: a ChatML-shaped template, a
    //! character tokenizer, and a "model" whose next token is the label the
    //! state names (`[[B]]`). No model, device work or socket.
    use super::*;
    use crate::app::common::{DecodeTiming, PrefillTiming};
    use crate::app::family::{Capabilities, LFM2_CAPABILITIES};
    use crate::app::template::{Rendered, ThinkTags};
    use crate::{DecodeParams, Decoded, ReadbackArm};
    use lloom::load_warning::LoadWarning;
    use lloom_tensor::Device;
    use parking_lot::Mutex;

    const START: u32 = 1000;
    const END: u32 = 1001;
    const THINK: u32 = 1002;
    const UNTHINK: u32 = 1003;
    const YES: u32 = 500;
    const NO: u32 = 501;
    const VOCAB: u32 = 1100;

    struct Toy {
        device: Device,
        capabilities: Capabilities,
        thinking: Option<bool>,
        prompts: Mutex<Vec<String>>,
    }

    fn toy() -> Toy {
        Toy {
            device: Device::Cpu,
            capabilities: LFM2_CAPABILITIES,
            thinking: None,
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn piece(id: u32) -> String {
        match id {
            START => "<|im_start|>".into(),
            END => "<|im_end|>".into(),
            THINK => "<think>".into(),
            UNTHINK => "</think>".into(),
            YES => "yes".into(),
            NO => "no".into(),
            c => char::from_u32(c).unwrap().to_string(),
        }
    }

    impl Served for Toy {
        fn device(&self) -> &Device {
            &self.device
        }
        fn model_name(&self) -> &str {
            "toy"
        }
        fn kv_capacity(&self) -> usize {
            4096
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
            &[END]
        }
        fn bos_id(&self) -> Option<u32> {
            None
        }
        fn health(&self) -> Value {
            Value::Null
        }
        fn describe(&self) -> String {
            String::new()
        }
        fn capabilities(&self) -> Capabilities {
            self.capabilities
        }
        fn load_warnings(&self) -> Vec<LoadWarning> {
            Vec::new()
        }
        fn note_load_warnings(&mut self, _: Vec<LoadWarning>) {}
        fn encode(&self, text: &str, _: bool) -> anyhow::Result<Vec<u32>> {
            let mut ids = Vec::new();
            let mut rest = text;
            'outer: while !rest.is_empty() {
                for id in [START, END, THINK, UNTHINK, YES, NO] {
                    if let Some(tail) = rest.strip_prefix(piece(id).as_str()) {
                        ids.push(id);
                        rest = tail;
                        continue 'outer;
                    }
                }
                let c = rest.chars().next().unwrap();
                ids.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
            Ok(ids)
        }
        fn token_str(&self, id: u32) -> String {
            piece(id)
        }
        fn render_checked(
            &self,
            messages: &Value,
            kwargs: &Map<String, Value>,
            _: &[&str],
        ) -> anyhow::Result<Rendered> {
            let mut text = String::new();
            for message in messages.as_array().unwrap() {
                text += &format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    message["role"].as_str().unwrap(),
                    message["content"].as_str().unwrap()
                );
            }
            text += "<|im_start|>assistant\n";
            match (self.thinking, kwargs.get("enable_thinking")) {
                (Some(true), Some(Value::Bool(false))) => text += "<think>\n\n</think>\n\n",
                (Some(_), _) => text += "<think>\n",
                (None, _) => (),
            }
            Ok(Rendered {
                text,
                ignored: Vec::new(),
            })
        }
        fn template_reads(&self, name: &str) -> Option<bool> {
            Some(name == "enable_thinking" && self.thinking == Some(true))
        }
        fn think_tags(&self) -> Option<ThinkTags> {
            self.thinking.map(|_| ThinkTags {
                start: THINK,
                end: UNTHINK,
                tool_call: None,
            })
        }
        fn seed_prompt_row(&self, _: usize, _: usize) -> Vec<u32> {
            unreachable!()
        }
        fn complete_until(
            &self,
            _: &[u32],
            _: &DecodeParams,
            _: &mut (dyn FnMut(&str) -> bool + Send),
        ) -> anyhow::Result<Decoded> {
            unreachable!("a decision generates nothing")
        }
        fn prefill_throughput(&self, _: usize, _: usize) -> anyhow::Result<PrefillTiming> {
            unreachable!()
        }
        fn decode_throughput(&self, _: usize, _: usize, _: usize) -> anyhow::Result<DecodeTiming> {
            unreachable!()
        }
        /// Every token at logit 0 except the one the prompt's `[[x]]` names,
        /// at 5; the full distribution, most likely first.
        fn next_token_logprobs(
            &self,
            prompt: &[u32],
            k: usize,
        ) -> anyhow::Result<(u32, Vec<(u32, f32)>)> {
            let text: String = prompt.iter().map(|&id| piece(id)).collect();
            self.prompts.lock().push(text.clone());
            let named = text
                .split("[[")
                .nth(1)
                .and_then(|rest| rest.split("]]").next())
                .map(|name| self.encode(name, false).unwrap()[0]);
            let logit = |id: u32| if Some(id) == named { 5.0f32 } else { 0.0 };
            let norm = (0..VOCAB).map(|id| logit(id).exp()).sum::<f32>().ln();
            let mut all: Vec<(u32, f32)> = (0..VOCAB).map(|id| (id, logit(id) - norm)).collect();
            all.sort_by(|a, b| b.1.total_cmp(&a.1));
            all.truncate(k);
            Ok((all[0].0, all))
        }
        fn check_batch(&self, _: usize) -> anyhow::Result<()> {
            Ok(())
        }
        fn shutdown(&self) {}
    }

    fn body(text: &str) -> OrderedJson {
        OrderedJson::from_slice(text.as_bytes()).unwrap()
    }

    #[test]
    fn generic_reads_the_named_label_through_the_served_seam() {
        let engine = toy();
        let decisions = Decisions::resolve(Some("generic"), &engine, None).unwrap();
        assert_eq!(
            decisions.describe().as_deref(),
            Some("decision template: generic (uncalibrated)")
        );
        let response = decisions
            .decide(
                &engine,
                &body(
                    r#"{"state":"[[B]] sky","instructions":"Be brief.","questions":{
                        "colour":{"type":"choice","instructions":"Which?","criteria":{"red":null,"blue":"cold"}},
                        "fact":{"type":"noul","instructions":"Is it?"}}}"#,
                ),
            )
            .unwrap();
        let wire: Value = serde_json::to_value(&response).unwrap();
        assert_eq!(wire["answers"]["colour"]["choice"], "blue");
        assert!(
            wire["diagnostics"]["colour"]["label_mass"]
                .as_f64()
                .unwrap()
                > 0.1
        );
        assert_eq!(wire["diagnostics"]["colour"]["argmax_is_label"], true);
        // The noul read sees the same [[B]]: no label is the argmax, and the
        // labels share what little mass is left.
        assert_eq!(wire["diagnostics"]["fact"]["argmax_is_label"], false);
        assert!(wire["diagnostics"]["fact"]["label_mass"].as_f64().unwrap() < 0.01);
        let prompts = engine.prompts.lock();
        assert_eq!(
            prompts[0],
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n[[B]] sky\n\n\
             Question: Which?\nA: red\nB: blue - cold\nAnswer with the letter of one option only.\
             <|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            wire["usage"]["input_tokens"].as_u64().unwrap(),
            prompts
                .iter()
                .map(|p| engine.encode(p, true).unwrap().len() as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn thinking_is_turned_off_or_the_model_is_refused_by_name() {
        let mut engine = toy();
        engine.thinking = Some(true);
        let decisions = Decisions::resolve(Some("generic"), &engine, None).unwrap();
        decisions
            .decide(
                &engine,
                &body(r#"{"state":"[[no]]","questions":{"q":{"type":"noul","instructions":"?"}}}"#),
            )
            .unwrap();
        assert!(engine.prompts.lock()[0].ends_with("assistant\n<think>\n\n</think>\n\n"));
        engine.thinking = Some(false); // writes a block, ignores enable_thinking
        let error = Decisions::resolve(Some("generic"), &engine, None)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("thinking cannot be turned off"), "{error}");
    }

    #[test]
    fn nothing_selected_on_a_plain_model_is_a_named_refusal() {
        let engine = toy();
        let decisions = Decisions::resolve(None, &engine, None).unwrap();
        assert!(decisions.describe().is_none());
        assert_eq!(decisions.output_modalities(true), ["text"]);
        let Err(SystemOneError::Unsupported(message)) =
            decisions.decide(&engine, &body(r#"{"state":"x","questions":{}}"#))
        else {
            panic!("refused as unsupported")
        };
        assert!(
            message.contains("generic, lfm2-d1, lfm2-d1-omni"),
            "{message}"
        );
        let error = Decisions::resolve(Some("lfm2-d1"), &engine, None)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("only a family decision executor renders"),
            "{error}"
        );
    }

    #[test]
    fn a_native_checkpoint_is_named_and_only_native_advertises_decisions() {
        let dir = std::env::temp_dir().join(format!("lloom-d1-binding-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"lfm2_vl","auto_map":{"AutoModel":"modeling_d1.D1Model"}}"#,
        )
        .unwrap();
        let engine = toy();
        let refused = Decisions::resolve(None, &engine, Some(&dir)).unwrap();
        let Err(SystemOneError::Unsupported(message)) =
            refused.decide(&engine, &body(r#"{"state":"x","questions":{}}"#))
        else {
            panic!("a native checkpoint without an executor is refused")
        };
        assert!(message.contains("native lfm2-d1"), "{message}");
        let mut owner = toy();
        owner.capabilities.decisions = true;
        let native = Decisions::resolve(None, &owner, Some(&dir)).unwrap();
        assert_eq!(native.output_modalities(true), ["text", "decisions"]);
        assert_eq!(
            Decisions::Family {
                native: Some(DecisionType::Lfm2D1Omni)
            }
            .output_modalities(false),
            ["decisions"]
        );
        let generic = Decisions::resolve(Some("generic"), &engine, Some(&dir)).unwrap();
        assert_eq!(generic.output_modalities(true), ["text"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
