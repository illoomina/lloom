//! Named systemone templates over the incumbent compiled Jinja renderer.
//! This is rendering, not tokenization, media decoding or model execution.
use super::omni::{OMNI_OPTION, OMNI_QUESTION, OMNI_SEPARATOR, OMNI_STATE};
use super::{OmniMedia, OrderedJson, Question};
use hf_chat_template::ChatTemplate;
use lloom_tensor::{bail, Error, Result};
use serde::Serialize;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemOneTemplateKind {
    D1Causal,
    D1Omni,
}

/// Media classification follows decoding/sniffing, not the request field name.
/// The loaded family supplies the actual tokenizer/media marker. Audio is one
/// clip, never mixed with images; this type cannot represent mixed media.
#[derive(Clone, Copy, Debug)]
pub enum SystemOneMedia<'a> {
    Text,
    Images { count: usize, marker: &'a str },
    Audio { marker: &'a str },
}

// One media classification/validation for full and certified-prefix rendering.
fn media_parameters(media: SystemOneMedia<'_>) -> Result<(OmniMedia, usize, Option<&str>)> {
    match media {
        SystemOneMedia::Text => Ok((OmniMedia::Text, 0, None)),
        SystemOneMedia::Images { count, marker } => {
            if !(1..=8).contains(&count) || marker.is_empty() {
                bail!("systemone images require 1..=8 items and a nonempty marker")
            }
            Ok((OmniMedia::Image, count, Some(marker)))
        }
        SystemOneMedia::Audio { marker } => {
            if marker.is_empty() {
                bail!("systemone audio requires a nonempty marker")
            }
            Ok((OmniMedia::Audio, 1, Some(marker)))
        }
    }
}

pub struct SystemOneTemplate {
    compiled: ChatTemplate,
    kind: SystemOneTemplateKind,
    /// Only source-audited state-first templates receive a branch certificate.
    /// Arbitrary templates remain renderable via the incumbent render method.
    causal_prefix: Option<ChatTemplate>,
}

const OMNI_MARKER: &str = "<<d1omni:";

#[derive(Serialize)]
struct OptionContext<'a> {
    key: Cow<'a, str>,
    description: Cow<'a, OrderedJson>,
    label: Option<String>,
}
#[derive(Serialize)]
struct Context<'a> {
    id: Cow<'a, str>,
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: Cow<'a, OrderedJson>,
    state: Cow<'a, OrderedJson>,
    options: Vec<OptionContext<'a>>,
    images: Vec<&'a str>,
    audio: bool,
    sep: &'static str,
    mark_state: &'static str,
    mark_question: &'static str,
    mark_option: &'static str,
}

impl SystemOneTemplate {
    pub fn kind(&self) -> SystemOneTemplateKind {
        self.kind
    }

    /// Exact source-audited certificate availability, not a heuristic render probe.
    /// A false result does not disable ordinary generic template rendering.
    pub fn has_causal_certificate(&self) -> bool {
        self.causal_prefix.is_some()
    }

    /// Compile the checkpoint's named systemone source once. No fetch, guessed
    /// template or fallback is performed. Existing chat templates are unchanged.
    pub fn compile(source: &str, kind: SystemOneTemplateKind) -> Result<Self> {
        Ok(Self {
            compiled: ChatTemplate::from_str(source).map_err(Error::wrap)?,
            kind,
            causal_prefix: if kind == SystemOneTemplateKind::D1Causal
                && source == include_str!("fixtures/d1-systemone.jinja")
            {
                // Audited pinned source: every state read is before this unique
                // expression, and no question/id/options read precedes it. Do not
                // infer this property from a sentinel render or token LCP.
                let (prefix, _) = source.split_once("{{ instructions ").ok_or_else(|| {
                    Error::Msg("audited causal template lost its state boundary".into())
                })?;
                Some(ChatTemplate::from_str(prefix).map_err(Error::wrap)?)
            } else {
                None
            },
        })
    }

    /// Complete state-bearing rendered prefix for the audited causal template.
    /// Prefix rendering reuses the same HF renderer/ordered value semantics;
    /// callers must additionally verify token nesting against EVERY full prompt.
    /// This certificate does not authorize unknown templates by heuristic.
    pub(super) fn causal_state_prefix(&self, state: &OrderedJson) -> Result<String> {
        self.causal_state_prefix_with_media(state, SystemOneMedia::Text)
    }

    /// The SAME audited prefix with caller-supplied media markers. Expansion to
    /// geometry-specific image placeholders is the encoder's responsibility;
    /// full/prefix token nesting is checked by CausalRequest, not assumed here.
    pub(super) fn causal_state_prefix_with_media(
        &self,
        state: &OrderedJson,
        media: SystemOneMedia<'_>,
    ) -> Result<String> {
        #[derive(Serialize)]
        struct Prefix<'a> {
            state: Cow<'a, OrderedJson>,
            images: Vec<&'a str>,
        }
        let prefix = self.causal_prefix.as_ref().ok_or_else(|| Error::Msg(
            "systemone template has no audited state-prefix certificate; ordinary rendering remains supported".into()))?;
        let (_, n_media, marker) = media_parameters(media)?;
        // Full rendering removes injected marker spellings from JSON string
        // values. The state prefix applies that SAME transform before proving
        // byte/token nesting; image markers themselves remain outside JSON.
        let state = match marker {
            Some(marker) => Cow::Owned(transform_json(state, false, Some(marker))),
            None => Cow::Borrowed(state),
        };
        prefix
            .render_context(&Prefix {
                state,
                images: marker.map_or_else(Vec::new, |marker| vec![marker; n_media]),
            })
            .map_err(Error::wrap)
    }

    /// Preserve raw JSON descriptions and state through ordered serialization.
    /// Causal labels are resolved by the tokenizer/readout owner in the same
    /// order as public Question probabilities (noul yes/true, then no/false).
    /// Omni has no label tokens: pass an empty slice, and its head options render
    /// in false/true order. This does not reorder the public answer legend.
    ///
    /// Caller admits context/string allocations and bounds source/request bytes.
    /// Renderer uses incumbent Python-compatible JSON formatting; float byte
    /// identity with llama's ostream serializer is NOT established or assumed.
    pub fn render(
        &self,
        id: &str,
        state: &OrderedJson,
        question: &Question<OrderedJson>,
        labels: &[String],
        media: SystemOneMedia<'_>,
    ) -> Result<String> {
        question.validate()?;
        let count = question.option_count();
        match self.kind {
            SystemOneTemplateKind::D1Causal if labels.len() != count => {
                bail!("systemone causal labels must match option count")
            }
            SystemOneTemplateKind::D1Omni if !labels.is_empty() => {
                bail!("systemone omni uses head markers, not causal labels")
            }
            _ => (),
        }
        let (media_kind, n_media, marker) = media_parameters(media)?;
        let omni = self.kind == SystemOneTemplateKind::D1Omni;
        let text = |value: &str| transform_text(value, omni, marker);
        let raw = |value: &OrderedJson| transform_json(value, omni, marker);
        let mut options = Vec::with_capacity(count);
        let mut add = |key: Cow<'_, str>, description: Option<&OrderedJson>, index: usize| {
            options.push(OptionContext {
                key: Cow::Owned(text(&key)),
                description: Cow::Owned(raw(description.unwrap_or(&OrderedJson::Null))),
                label: (!omni).then(|| text(&labels[index])),
            });
        };
        let (kind, instructions) = match question {
            Question::Choice {
                instructions,
                criteria,
            } => {
                for (index, (key, description)) in criteria.iter().enumerate() {
                    add(Cow::Borrowed(key), description.as_ref(), index);
                }
                ("choice", instructions)
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                for (index, description) in criteria.iter().enumerate() {
                    add(Cow::Owned(index.to_string()), Some(description), index);
                }
                ("score", instructions)
            }
            Question::Noul {
                instructions,
                criteria,
            } => {
                let (yes, no) = criteria
                    .as_ref()
                    .map_or((None, None), |(yes, no)| (yes.as_ref(), no.as_ref()));
                if omni {
                    add(Cow::Borrowed("false"), no, 1);
                    add(Cow::Borrowed("true"), yes, 0);
                } else {
                    add(Cow::Borrowed("true"), yes, 0);
                    add(Cow::Borrowed("false"), no, 1);
                }
                ("noul", instructions)
            }
        };
        let context = Context {
            id: Cow::Owned(text(id)),
            kind,
            instructions: Cow::Owned(raw(instructions)),
            state: Cow::Owned(raw(state)),
            options,
            images: marker.map_or_else(Vec::new, |marker| vec![marker; n_media]),
            audio: media_kind == OmniMedia::Audio,
            sep: OMNI_SEPARATOR,
            mark_state: OMNI_STATE,
            mark_question: OMNI_QUESTION,
            mark_option: OMNI_OPTION,
        };
        self.compiled.render_context(&context).map_err(Error::wrap)
    }
}

fn transform_text(value: &str, omni: bool, media_marker: Option<&str>) -> String {
    let text = if omni {
        super::omni::escape(value).replace(OMNI_MARKER, "<<d1omni ")
    } else {
        value.to_owned()
    };
    match media_marker {
        Some(marker) => text.replace(marker, " "),
        None => text,
    }
}

fn transform_json(value: &OrderedJson, omni: bool, marker: Option<&str>) -> OrderedJson {
    match value {
        OrderedJson::String(value) => OrderedJson::String(transform_text(value, omni, marker)),
        OrderedJson::Array(values) => OrderedJson::Array(
            values
                .iter()
                .map(|v| transform_json(v, omni, marker))
                .collect(),
        ),
        OrderedJson::Object(entries) => {
            let mut out: Vec<(String, OrderedJson)> = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                // Reference escapes special tokens in omni keys, but the
                // later marker replacements affect string VALUES only.
                let key = if omni {
                    super::omni::escape(key)
                } else {
                    key.clone()
                };
                let value = transform_json(value, omni, marker);
                if let Some((_, previous)) = out.iter_mut().find(|(name, _)| name == &key) {
                    // Escaping can collide with a literal already-escaped key;
                    // reference object assignment keeps first position/last value.
                    *previous = value;
                } else {
                    out.push((key, value));
                }
            }
            OrderedJson::Object(out)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn json(value: &str) -> OrderedJson {
        OrderedJson::from_slice(value.as_bytes()).unwrap()
    }
    fn causal() -> SystemOneTemplate {
        SystemOneTemplate::compile(
            include_str!("fixtures/d1-systemone.jinja"),
            SystemOneTemplateKind::D1Causal,
        )
        .unwrap()
    }
    fn omni() -> SystemOneTemplate {
        SystemOneTemplate::compile(
            include_str!("fixtures/d1-omni-systemone.jinja"),
            SystemOneTemplateKind::D1Omni,
        )
        .unwrap()
    }

    #[test]
    fn causal_raw_json_order_and_false_description_follow_pinned_template() -> Result<()> {
        let question = Question::Choice {
            instructions: json(r#"{"z":"雪","a":false}"#),
            criteria: vec![
                ("first_name".into(), Some(OrderedJson::Bool(false))),
                ("second".into(), Some(json(r#"{"z":2,"a":1}"#))),
            ],
        };
        let got = causal().render(
            "q",
            &json(r#"{"z":1,"a":[null,true]}"#),
            &question,
            &["A".into(), "B".into()],
            SystemOneMedia::Text,
        )?;
        assert_eq!(got, "<|startoftext|><|im_start|>user\n{\n  \"z\": 1,\n  \"a\": [\n    null,\n    true\n  ]\n}\n\n\nQUESTION:\n{\"z\": \"雪\", \"a\": false}\n\nOptions:\nA first name\nB {\"z\": 2, \"a\": 1}\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n");
        let noul = Question::Noul {
            instructions: json(r#""holds?""#),
            criteria: Some((Some(OrderedJson::Null), Some(OrderedJson::Bool(false)))),
        };
        let got = causal().render(
            "q",
            &OrderedJson::Null,
            &noul,
            &["true".into(), "false".into()],
            SystemOneMedia::Text,
        )?;
        assert_eq!(got,"<|startoftext|><|im_start|>user\nholds?\nYes: None\nNo: false\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n");
        Ok(())
    }

    #[test]
    fn omni_audio_null_state_and_raw_false_description_keep_head_order() -> Result<()> {
        let q = Question::Noul {
            instructions: json(r#""<|mask|> <<d1omni:state>>""#),
            criteria: Some((
                Some(json(r#""custom yes""#)),
                Some(OrderedJson::Bool(false)),
            )),
        };
        let got = omni().render(
            "q",
            &OrderedJson::Null,
            &q,
            &[],
            SystemOneMedia::Audio { marker: "<media>" },
        )?;
        assert_eq!(got, "<media><<d1omni:sep>><|startoftext|><|reserved_7|><<d1omni:sep>><<d1omni:state>>{}<<d1omni:sep>><<d1omni:question>><|reserved_8|><¦mask¦> <<d1omni state>><<d1omni:sep>><|reserved_9|><|mask|><<d1omni:sep>><<d1omni:option>> false: no<<d1omni:sep>><|reserved_10|><<d1omni:sep>><|reserved_9|><|mask|><<d1omni:sep>><<d1omni:option>> true: yes<<d1omni:sep>><|reserved_10|><<d1omni:sep>><|reserved_11|>");
        let choice = Question::Choice {
            instructions: OrderedJson::Bool(false),
            criteria: vec![("only".into(), Some(OrderedJson::Bool(false)))],
        };
        // Endpoint template accepts singleton choice; existing Python preparation
        // keeps its separately documented >=2 contract.
        let got = omni().render("q", &OrderedJson::Null, &choice, &[], SystemOneMedia::Text)?;
        assert!(got.contains("<|reserved_8|>false"));
        assert!(got.contains("<<d1omni:option>> only: false"));
        assert_eq!(got.matches("<|mask|>").count(), 1);
        Ok(())
    }

    #[test]
    fn raw_key_escape_collisions_and_invalid_media_label_contracts() -> Result<()> {
        let q = Question::Noul {
            instructions: OrderedJson::Bool(false),
            criteria: None,
        };
        let template = SystemOneTemplate::compile(
            "{{ state | tojson }}|{{ instructions | tojson }}",
            SystemOneTemplateKind::D1Omni,
        )?;
        let state = json(r#"{"<|mask|>":1,"<¦mask¦>":2,"<media>":"<media>"}"#);
        assert_eq!(
            template.render(
                "q",
                &state,
                &q,
                &[],
                SystemOneMedia::Images {
                    count: 1,
                    marker: "<media>"
                }
            )?,
            r#"{"<¦mask¦>": 2, "<media>": " "}|false"#
        );
        assert!(causal()
            .render("q", &state, &q, &[], SystemOneMedia::Text)
            .is_err());
        assert!(omni()
            .render(
                "q",
                &state,
                &q,
                &["unexpected".into()],
                SystemOneMedia::Text
            )
            .is_err());
        for media in [
            SystemOneMedia::Images {
                count: 0,
                marker: "m",
            },
            SystemOneMedia::Images {
                count: 9,
                marker: "m",
            },
            SystemOneMedia::Images {
                count: 1,
                marker: "",
            },
            SystemOneMedia::Audio { marker: "" },
        ] {
            assert!(omni().render("q", &state, &q, &[], media).is_err());
        }
        Ok(())
    }
}
