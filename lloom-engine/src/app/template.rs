//! The checkpoint's chat template as the engine drives it: the compiled
//! template, the variables it reads, and the variables the engine sets on it
//! from a flag (`lloom chat`) or a request field (`lloom serve`).
//!
//! Which variables a family supports is read from its template, never listed
//! per family: a variable the template's source never names is refused by
//! `chat` and ignored with a warning by `serve`, and a variable the template
//! names but renders identically for any value (Qwen4-exp's
//! `reasoning_effort` with thinking off) is treated the same way. A value the
//! template itself rejects (`raise_exception`) is a [`Rejected`] error, which
//! `serve` answers with a 400.

use anyhow::Result;
use hf_chat_template::ChatTemplate;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// Thinking on or off (`lloom chat --think`, `/eval`'s `enable_thinking`).
pub const ENABLE_THINKING: &str = "enable_thinking";
/// How long the model thinks (`lloom chat --reasoning-effort`, OpenAI's
/// `reasoning_effort` request field). The values are the template's.
pub const REASONING_EFFORT: &str = "reasoning_effort";

/// The variables the engine sets from its own flags and request fields.
pub const ENGINE_KWARGS: [&str; 2] = [ENABLE_THINKING, REASONING_EFFORT];

/// A value no template recognises: rendering with it in place of a
/// variable's value shows whether the template's output depends on that
/// value at all.
const PROBE_VALUE: &str = "lloom-probe: not a value";

/// The template refused to render the request (`raise_exception`): the
/// message is the template's own.
#[derive(Debug)]
pub struct Rejected(pub String);

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the chat template rejected the request: {}", self.0)
    }
}

impl std::error::Error for Rejected {}

/// A variable a request set that the rendered prompt does not reflect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ignored {
    pub name: String,
    pub value: Value,
    pub reason: &'static str,
}

impl std::fmt::Display for Ignored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={} {}", self.name, self.value, self.reason)
    }
}

/// A prompt, with the checked variables it does not reflect.
#[derive(Debug)]
pub struct Rendered {
    pub text: String,
    pub ignored: Vec<Ignored>,
}

pub const NOT_READ: &str = "is not read by the chat template";
pub const NO_EFFECT: &str =
    "has no effect on the chat template's output under the other variables (any value renders the same prompt)";

/// A compiled chat template and the variables its source names.
pub struct Template {
    compiled: ChatTemplate,
    /// `None` when the source cannot be analysed (a construct the static
    /// parser rejects and the renderer accepts): nothing is then refused as
    /// unread, and the render probe still runs.
    reads: Option<BTreeSet<String>>,
}

impl Template {
    pub fn compile(source: &str) -> Result<Self> {
        let compiled = ChatTemplate::from_str(source)
            .map_err(|e| anyhow::anyhow!("compiling chat_template.jinja: {e}"))?;
        Ok(Self {
            compiled,
            reads: undeclared_variables(source),
        })
    }

    /// Whether the template's source reads `name`; `None` when it could not
    /// be analysed.
    pub fn reads(&self, name: &str) -> Option<bool> {
        self.reads.as_ref().map(|r| r.contains(name))
    }

    /// Render `messages` with the generation prompt appended; `kwargs` are
    /// extra template variables, laid over the defaults.
    pub fn render(
        &self,
        messages: &Value,
        bos_token: &str,
        kwargs: &Map<String, Value>,
    ) -> Result<String> {
        let mut ctx = json!({
            "messages": messages,
            "bos_token": bos_token,
            "add_generation_prompt": true,
        });
        for (k, v) in kwargs {
            ctx[k] = v.clone();
        }
        self.compiled.render_context(&ctx).map_err(|e| match e {
            hf_chat_template::Error::TemplateRaised { message } => {
                anyhow::Error::new(Rejected(message))
            }
            e => anyhow::anyhow!("rendering the chat template: {e}"),
        })
    }

    /// [`Self::render`], with each of `checked` that `kwargs` sets tested
    /// against the template: dropped before rendering when the template does
    /// not read it, and reported when the prompt is the same for any value
    /// of it.
    pub fn render_checked(
        &self,
        messages: &Value,
        bos_token: &str,
        kwargs: &Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        let mut kwargs = kwargs.clone();
        let mut ignored = Vec::new();
        for &name in checked {
            if self.reads(name) == Some(false) {
                if let Some(value) = kwargs.remove(name) {
                    ignored.push(Ignored {
                        name: name.to_string(),
                        value,
                        reason: NOT_READ,
                    });
                }
            }
        }
        let text = self.render(messages, bos_token, &kwargs)?;
        for &name in checked {
            let Some(value) = kwargs.get(name).cloned() else {
                continue;
            };
            let mut probe = kwargs.clone();
            probe.insert(name.to_string(), Value::from(PROBE_VALUE));
            if matches!(self.render(messages, bos_token, &probe), Ok(t) if t == text) {
                ignored.push(Ignored {
                    name: name.to_string(),
                    value,
                    reason: NO_EFFECT,
                });
            }
        }
        Ok(Rendered { text, ignored })
    }
}

/// The top-level variables `source` looks up, from minijinja's static
/// analysis of its AST (control flow is not considered: a variable read on
/// any branch counts).
fn undeclared_variables(source: &str) -> Option<BTreeSet<String>> {
    let source = without_generation_blocks(source);
    let mut env = minijinja::Environment::new();
    env.add_template("chat", &source).ok()?;
    let names = env.get_template("chat").ok()?.undeclared_variables(false);
    Some(names.into_iter().collect())
}

/// `source` with transformers' `{% generation %}` / `{% endgeneration %}`
/// block tags (an assistant-token mask, no output of its own) as a plain
/// `if true` block, which the parser accepts. hf-chat-template renders
/// through the same rewrite, privately; this copy feeds the analysis only,
/// so whitespace control is not carried over.
fn without_generation_blocks(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let Some(len) = rest[open + 2..].find("%}") else {
            break;
        };
        let inner = rest[open + 2..open + 2 + len]
            .trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..open]);
        match inner {
            "generation" => out.push_str("{% if true %}"),
            "endgeneration" => out.push_str("{% endif %}"),
            _ => out.push_str(&rest[open..open + 2 + len + 2]),
        }
        rest = &rest[open + 2 + len + 2..];
    }
    out.push_str(rest);
    out
}

/// The template variables a request sets: its `chat_template_kwargs`, with
/// each of the top-level `fields` folded in under the same name. A field set
/// both ways must agree.
pub fn request_kwargs(
    v: &Value,
    fields: &[&str],
) -> std::result::Result<Map<String, Value>, String> {
    let mut kwargs = match v.get("chat_template_kwargs") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return Err("field 'chat_template_kwargs' must be an object".to_string()),
    };
    for &field in fields {
        let top = match v.get(field) {
            None | Some(Value::Null) => continue,
            Some(t) => t,
        };
        match kwargs.get(field) {
            Some(k) if k != top => {
                return Err(format!(
                    "field '{field}' ({top}) and 'chat_template_kwargs.{field}' ({k}) disagree"
                ))
            }
            _ => {
                kwargs.insert(field.to_string(), top.clone());
            }
        }
    }
    if let Some(e) = kwargs.get(REASONING_EFFORT) {
        if !e.is_string() && !e.is_null() {
            return Err(format!("'{REASONING_EFFORT}' must be a string, not {e}"));
        }
    }
    Ok(kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom::testing::tiny::TinyTokenizer;

    const LOW: &str = "Reasoning effort is set to low.";
    const XHIGH: &str = "Reasoning effort is set to xhigh.";

    fn qwen() -> Template {
        Template::compile(TinyTokenizer::qwen4exp().chat_template).unwrap()
    }

    fn lfm2() -> Template {
        Template::compile(TinyTokenizer::lfm2().chat_template).unwrap()
    }

    fn user() -> Value {
        json!([{ "role": "user", "content": "The capital of France?" }])
    }

    fn kw(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn think(effort: Option<&str>) -> Map<String, Value> {
        let mut m = kw(&[(ENABLE_THINKING, json!(true))]);
        if let Some(e) = effort {
            m.insert(REASONING_EFFORT.into(), json!(e));
        }
        m
    }

    /// Qwen4-exp's template names both thinking variables; LFM2's names
    /// neither, and a variable no template has is read by neither.
    #[test]
    fn the_variables_come_from_the_templates_source() {
        let (q, l) = (qwen(), lfm2());
        assert_eq!(q.reads(REASONING_EFFORT), Some(true));
        assert_eq!(q.reads(ENABLE_THINKING), Some(true));
        assert_eq!(q.reads("messages"), Some(true));
        assert_eq!(l.reads(REASONING_EFFORT), Some(false));
        assert_eq!(l.reads(ENABLE_THINKING), Some(false));
        assert_eq!(l.reads("messages"), Some(true));
        assert_eq!(q.reads("thinking_budget"), Some(false));
    }

    /// The analysis reads through a `{% generation %}` block (LFM2's
    /// template has one) and misses nothing inside it.
    #[test]
    fn a_generation_block_does_not_blind_the_analysis() {
        let src = "{%- for m in messages -%}{%- generation -%}{{ m.content }}{{ inside }}{%- endgeneration -%}{%- endfor -%}";
        assert!(
            minijinja::Environment::new()
                .add_template("x", src)
                .is_err(),
            "the parser rejects the tag"
        );
        let t = Template::compile(src).unwrap();
        assert_eq!(t.reads("inside"), Some(true));
        assert_eq!(t.reads("messages"), Some(true));
        assert_eq!(t.reads("outside"), Some(false));
        assert_eq!(
            without_generation_blocks(
                "a{% generation %}b{%- endgeneration +%}c{% if x %}{% endif %}"
            ),
            "a{% if true %}b{% endif %}c{% if x %}{% endif %}"
        );
    }

    /// Each value renders its own instruction; `medium` renders none; no
    /// value is the template's default, byte-identical to `xhigh`.
    #[test]
    fn each_effort_renders_its_instruction_and_none_is_the_templates_default() {
        let q = qwen();
        let r = |e| q.render(&user(), "", &think(e)).unwrap();
        let (low, medium, xhigh, default) =
            (r(Some("low")), r(Some("medium")), r(Some("xhigh")), r(None));
        assert!(
            low.starts_with(&format!("<|im_start|>system\n{LOW}")),
            "{low}"
        );
        assert!(
            xhigh.starts_with(&format!("<|im_start|>system\n{XHIGH}")),
            "{xhigh}"
        );
        assert!(!medium.contains("Reasoning effort"), "{medium}");
        assert!(medium.starts_with("<|im_start|>user\n"), "{medium}");
        assert_ne!(low, xhigh);
        assert_eq!(default, xhigh);
        for p in [&low, &medium, &xhigh] {
            assert!(p.ends_with("<|im_start|>assistant\n<think>\n"), "{p}");
        }
    }

    /// A value the template does not accept is its own `raise_exception`,
    /// surfaced as [`Rejected`] with the template's message.
    #[test]
    fn a_value_the_template_rejects_is_a_rejection_with_its_message() {
        let err = qwen()
            .render(&user(), "", &think(Some("high")))
            .unwrap_err();
        let rejected = err
            .downcast_ref::<Rejected>()
            .expect("a Rejected, not a render failure");
        assert_eq!(
            rejected.0,
            "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
        );
    }

    /// Thinking on: the effort is applied. Thinking off: the template renders
    /// the same prompt for any effort, so it is reported, and the prompt is
    /// the one the template renders with no effort at all.
    #[test]
    fn the_check_reports_an_effort_the_prompt_does_not_reflect() {
        let q = qwen();
        let on = q
            .render_checked(&user(), "", &think(Some("low")), &[REASONING_EFFORT])
            .unwrap();
        assert!(on.ignored.is_empty(), "{:?}", on.ignored);
        assert!(on.text.contains(LOW));

        let off = kw(&[
            (ENABLE_THINKING, json!(false)),
            (REASONING_EFFORT, json!("low")),
        ]);
        let r = q
            .render_checked(&user(), "", &off, &[REASONING_EFFORT])
            .unwrap();
        assert_eq!(
            r.ignored,
            vec![Ignored {
                name: REASONING_EFFORT.into(),
                value: json!("low"),
                reason: NO_EFFECT
            }]
        );
        let bare = q
            .render(&user(), "", &kw(&[(ENABLE_THINKING, json!(false))]))
            .unwrap();
        assert_eq!(r.text, bare);

        let err = q
            .render_checked(&user(), "", &think(Some("high")), &[REASONING_EFFORT])
            .unwrap_err();
        assert!(err.downcast_ref::<Rejected>().is_some(), "{err:#}");
    }

    /// LFM2: the effort is dropped before rendering, reported as unread, and
    /// the prompt is byte-identical to one rendered without it.
    #[test]
    fn a_template_that_does_not_read_the_effort_renders_without_it() {
        let l = lfm2();
        let with = kw(&[(REASONING_EFFORT, json!("low"))]);
        let r = l
            .render_checked(&user(), "<|startoftext|>", &with, &[REASONING_EFFORT])
            .unwrap();
        assert_eq!(
            r.ignored,
            vec![Ignored {
                name: REASONING_EFFORT.into(),
                value: json!("low"),
                reason: NOT_READ
            }]
        );
        assert_eq!(
            r.text,
            l.render(&user(), "<|startoftext|>", &Map::new()).unwrap()
        );
    }

    /// The request's two spellings fold into one variable, agree or are
    /// refused, and a malformed one is refused by name.
    #[test]
    fn the_request_spellings_fold_into_one_variable() {
        let f = [REASONING_EFFORT];
        let get = |v: Value| request_kwargs(&v, &f);
        assert_eq!(get(json!({})).unwrap(), Map::new());
        assert_eq!(
            get(json!({"reasoning_effort": "low"})).unwrap(),
            kw(&[(REASONING_EFFORT, json!("low"))])
        );
        assert_eq!(
            get(json!({"chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": true}})).unwrap(),
            kw(&[(REASONING_EFFORT, json!("low")), (ENABLE_THINKING, json!(true))])
        );
        assert_eq!(
            get(json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "low"}})).unwrap(),
            kw(&[(REASONING_EFFORT, json!("low"))])
        );
        assert_eq!(get(json!({"reasoning_effort": null})).unwrap(), Map::new());
        let e = get(json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "xhigh"}}))
            .unwrap_err();
        assert!(e.contains("disagree"), "{e}");
        let e = get(json!({"reasoning_effort": 3})).unwrap_err();
        assert!(e.contains("must be a string"), "{e}");
        let e = get(json!({"chat_template_kwargs": {"reasoning_effort": ["low"]}})).unwrap_err();
        assert!(e.contains("must be a string"), "{e}");
        let e = get(json!({"chat_template_kwargs": "enable_thinking=true"})).unwrap_err();
        assert!(e.contains("must be an object"), "{e}");
    }
}
