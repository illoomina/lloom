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
//! `serve` answers with a 400; so is a render the template fails on the
//! request's own data ([`Unrenderable`]).

use anyhow::Result;
use hf_chat_template::ChatTemplate;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// Thinking on or off (`lloom chat --think` / `--no-think`, a request's
/// `enable_thinking`): the template's own switch where it reads it
/// (`thinking.rs` applies it where it does not).
pub const ENABLE_THINKING: &str = "enable_thinking";
/// How long the model thinks (`lloom chat --reasoning-effort`, OpenAI's
/// `reasoning_effort` request field). The values are the template's.
pub const REASONING_EFFORT: &str = "reasoning_effort";

/// The tool list (OpenAI's `tools` request field).
pub const TOOLS: &str = "tools";

/// The variables the engine sets from its own flags and request fields.
pub const ENGINE_KWARGS: [&str; 2] = [ENABLE_THINKING, REASONING_EFFORT];

/// The top-level request fields a chat request folds into the template's
/// variables, beside its `chat_template_kwargs`.
pub const REQUEST_FIELDS: [&str; 3] = [REASONING_EFFORT, ENABLE_THINKING, TOOLS];

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

/// The template failed on the request's data outside `raise_exception` (a
/// type it does not handle, an attribute the data lacks): the same template
/// renders the engine's own one-message conversation.
#[derive(Debug)]
pub struct Unrenderable(pub String);

impl std::fmt::Display for Unrenderable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the chat template cannot render the request: {}", self.0)
    }
}

impl std::error::Error for Unrenderable {}

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

    /// Every top-level name the template's source reads (the functions it
    /// calls included); `None` when it could not be analysed.
    pub fn variables(&self) -> Option<&BTreeSet<String>> {
        self.reads.as_ref()
    }

    /// Render `messages` with the generation prompt appended; `kwargs` are
    /// extra template variables, laid over the defaults. A failure is the
    /// request's ([`Rejected`], [`Unrenderable`]) when the template renders a
    /// one-message conversation with no variables of the caller's, and the
    /// engine's otherwise.
    pub fn render(
        &self,
        messages: &Value,
        bos_token: &str,
        kwargs: &Map<String, Value>,
    ) -> Result<String> {
        let ctx = |messages: &Value, kwargs: &Map<String, Value>| {
            let mut ctx = json!({
                "messages": messages,
                "bos_token": bos_token,
                "add_generation_prompt": true,
            });
            for (k, v) in kwargs {
                ctx[k] = v.clone();
            }
            ctx
        };
        self.compiled
            .render_context(&ctx(messages, kwargs))
            .map_err(|e| match e {
                hf_chat_template::Error::TemplateRaised { message } => {
                    anyhow::Error::new(Rejected(message))
                }
                e => {
                    let own = json!([{ "role": "user", "content": PROBE_USER }]);
                    match self.compiled.render_context(&ctx(&own, &Map::new())) {
                        Ok(_) => anyhow::Error::new(Unrenderable(e.to_string())),
                        Err(_) => anyhow::anyhow!("rendering the chat template: {e}"),
                    }
                }
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

/// The thinking block's markers as a checkpoint's template writes them, and
/// the id that starts a tool call: the family's reasoning tags, read by
/// rendering the template ([`Template::think_tags`]), never listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkTags {
    pub start: u32,
    pub end: u32,
    /// A tool call ends the block too; `None` when the template writes tool
    /// calls without a marker token of their own.
    pub tool_call: Option<u32>,
}

impl ThinkTags {
    /// The ids that close the block.
    pub fn close_ids(&self) -> Vec<u32> {
        let mut v = vec![self.end];
        v.extend(self.tool_call);
        v
    }

    /// Whether `ids` end inside a thinking block — a start with no end or
    /// tool call after it — and if so how many tokens of the block they hold
    /// after the start marker (what vLLM counts against a budget).
    pub fn open_at_end(&self, ids: &[u32]) -> Option<usize> {
        let start = ids.iter().rposition(|&id| id == self.start)?;
        let after = &ids[start + 1..];
        let close = self.close_ids();
        (!after.iter().any(|id| close.contains(id))).then_some(after.len())
    }
}

/// A user message no template writes on its own, for the probe renders.
const PROBE_USER: &str = "lloom-probe-user";

/// Reasoning text no template writes on its own, for the probe renders.
const PROBE_REASONING: &str = "lloom-probe-reasoning";

impl Template {
    /// The reasoning tags this template writes, found by rendering: an
    /// assistant turn carrying [`PROBE_REASONING`] (under each field name
    /// the families read) puts the start marker before it and the end marker
    /// after it; the same turn with a tool call and without one first differ
    /// at the tool-call marker. A marker must be one token `is_marker`
    /// accepts (the tokenizer's added tokens), so plain text around the
    /// reasoning is never taken for one. `None` when the template writes no
    /// such block.
    pub fn think_tags(
        &self,
        bos_token: &str,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
        is_marker: impl Fn(u32) -> bool,
    ) -> Option<ThinkTags> {
        let user = json!({ "role": "user", "content": PROBE_USER });
        let reasoned = json!([user, {
            "role": "assistant",
            "content": "lloom-probe-answer",
            "reasoning_content": PROBE_REASONING,
            "thinking": PROBE_REASONING,
            "reasoning": PROBE_REASONING,
        }]);
        let text = self.render(&reasoned, bos_token, &Map::new()).ok()?;
        let at = text.find(PROBE_REASONING)?;
        let start = *encode(text[..at].trim_end()).ok()?.last()?;
        let end = *encode(text[at + PROBE_REASONING.len()..].trim_start())
            .ok()?
            .first()?;
        if start == end || !is_marker(start) || !is_marker(end) {
            return None;
        }
        let bare = json!([user, { "role": "assistant", "content": "" }]);
        let called = json!([user, {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": { "name": "lloom_probe_tool", "arguments": { "lloom_probe_arg": "v" } },
            }],
        }]);
        let tool_call = (|| {
            let bare = encode(&self.render(&bare, bos_token, &Map::new()).ok()?).ok()?;
            let called = encode(&self.render(&called, bos_token, &Map::new()).ok()?).ok()?;
            let at = bare.iter().zip(&called).take_while(|(a, b)| a == b).count();
            let id = *called.get(at)?;
            (is_marker(id) && id != start && id != end).then_some(id)
        })();
        Some(ThinkTags {
            start,
            end,
            tool_call,
        })
    }
}

/// The top-level variables `source` looks up, from minijinja's static
/// analysis of its AST (control flow is not considered: a variable read on
/// any branch counts).
fn undeclared_variables(source: &str) -> Option<BTreeSet<String>> {
    let source = with_self_assignments_renamed(&without_generation_blocks(source));
    let mut env = minijinja::Environment::new();
    env.add_template("chat", &source).ok()?;
    let names = env.get_template("chat").ok()?.undeclared_variables(false);
    Some(names.into_iter().collect())
}

/// `source` with the target of each `{% set name = … name … %}` renamed, so
/// the analysis sees the read on the right-hand side. That is how a template
/// defaults a caller's variable (`set x = x | default(…)`, the LFM2
/// checkpoints' `preserve_thinking`), and minijinja's analysis assigns the
/// target before it visits the value, so it reports none of them. A name
/// the template declared before (a loop variable, a macro argument, an
/// earlier `set`) is still declared at the read and still not reported.
fn with_self_assignments_renamed(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let Some(len) = rest[open + 2..].find("%}") else {
            break;
        };
        let tag = &rest[open..open + 2 + len + 2];
        out.push_str(&rest[..open]);
        out.push_str(&self_assignment_renamed(tag).unwrap_or_else(|| tag.to_string()));
        rest = &rest[open + 2 + len + 2..];
    }
    out.push_str(rest);
    out
}

/// A `{% set name = … %}` tag whose value reads `name`, with the target
/// renamed; `None` for any other tag.
fn self_assignment_renamed(tag: &str) -> Option<String> {
    let body = &tag[2..tag.len() - 2];
    let lead = body.len() - body.trim_start_matches(['-', '+']).trim_start().len();
    let after_set = body[lead..].strip_prefix("set")?;
    let gap = after_set.len() - after_set.trim_start().len();
    if gap == 0 {
        return None;
    }
    let (target, value) = after_set.split_once('=')?;
    let target = target.trim();
    if value.starts_with('=') || !is_identifier(target) {
        return None;
    }
    if !read_names(value).iter().any(|n| n == target) {
        return None;
    }
    let at = 2 + lead + "set".len() + gap;
    Some(format!(
        "{}lloom_self_{}{}",
        &tag[..at],
        target,
        &tag[at + target.len()..]
    ))
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// The bare names an expression looks up: string literals, attributes
/// (`.name`) and keyword-argument names (`name=`) excluded.
fn read_names(expr: &str) -> Vec<String> {
    let b = expr.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'\'' || c == b'"' {
            i += 1;
            while i < b.len() && b[i] != c {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
        } else if c == b'_' || c.is_ascii_alphabetic() {
            let start = i;
            while i < b.len() && (b[i] == b'_' || b[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            let attribute = expr[..start].trim_end().ends_with('.');
            let after = expr[i..].trim_start();
            let keyword = after.starts_with('=') && !after.starts_with("==");
            if !attribute && !keyword {
                out.push(expr[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    out
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
/// both ways must agree; one set to `null` either way is unset.
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
        if kwargs.get(field) == Some(&Value::Null) {
            kwargs.remove(field);
        }
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
    if let Some(e) = kwargs.get(ENABLE_THINKING) {
        if !e.is_boolean() && !e.is_null() {
            return Err(format!(
                "'{ENABLE_THINKING}' must be true or false, not {e}"
            ));
        }
    }
    Ok(kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom::testing::tiny::TinyTokenizer;

    const LOW: &str = "Thinking effort: low.";
    const XHIGH: &str = "Thinking effort: xhigh.";

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

    /// A variable a template defaults onto itself (`set x = x |
    /// default(…)`) is read, though minijinja's own analysis does not report
    /// it; a name declared before its self-assignment (an earlier `set`, a
    /// loop variable), a local set from something else, an attribute and a
    /// keyword argument's name are not reads.
    #[test]
    fn a_variable_the_template_defaults_onto_itself_is_read() {
        let src = "{%- set preserve_thinking = preserve_thinking | default(false) -%}\
                   {% set local = 'local' %}{% set local = local ~ '!' %}\
                   {% for m in messages %}{% set m = m | trim %}{{ m }}{% endfor %}\
                   {% set attr = ns.attr %}{% set kw = f(kw=1) %}\
                   {{ preserve_thinking }}{{ local }}{{ attr }}{{ kw }}";
        let mut env = minijinja::Environment::new();
        env.add_template("x", src).unwrap();
        let theirs = env.get_template("x").unwrap().undeclared_variables(false);
        assert!(
            !theirs.contains("preserve_thinking"),
            "minijinja now reports it: {theirs:?}"
        );
        let t = Template::compile(src).unwrap();
        assert_eq!(t.reads("preserve_thinking"), Some(true));
        for name in ["local", "m", "attr", "kw"] {
            assert_eq!(t.reads(name), Some(false), "{name}");
        }
        for name in ["messages", "ns", "f"] {
            assert_eq!(t.reads(name), Some(true), "{name}");
        }
        assert_eq!(
            with_self_assignments_renamed("{%- set x = x | default(1) -%}{% set y = 2 %}"),
            "{%- set lloom_self_x = x | default(1) -%}{% set y = 2 %}"
        );
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
        assert!(!medium.contains("Thinking effort"), "{medium}");
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
            "tiny qwen4exp template: reasoning_effort high is not one of low, medium, xhigh"
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

    /// `tok`'s tokenizer, and the reasoning tags `template` yields through it.
    fn tags_of(tok: &TinyTokenizer, bos: &str) -> Option<ThinkTags> {
        let json = lloom::testing::tiny::tokenizer::tokenizer_json(&tok.added);
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        let added = tokenizer.get_added_tokens_decoder();
        Template::compile(tok.chat_template).unwrap().think_tags(
            bos,
            |s| Ok(tokenizer.encode(s, false).unwrap().get_ids().to_vec()),
            |id| added.contains_key(&id),
        )
    }

    /// Each family's tags, read from its template by rendering: Qwen4-exp's
    /// `<think>` / `</think>` and `<tool_call>`, LFM2's `<think>` / `</think>`
    /// and `<|tool_call_start|>` — the ids the fixture's token table gives
    /// them. A template that writes no block has no tags.
    #[test]
    fn the_reasoning_tags_come_from_the_template() {
        let q = TinyTokenizer::qwen4exp();
        assert_eq!(
            tags_of(&q, ""),
            Some(ThinkTags {
                start: q.id("<think>").unwrap(),
                end: q.id("</think>").unwrap(),
                tool_call: q.id("<tool_call>"),
            })
        );
        let l = TinyTokenizer::lfm2();
        assert_eq!(
            tags_of(&l, "<|startoftext|>"),
            Some(ThinkTags {
                start: l.id("<think>").unwrap(),
                end: l.id("</think>").unwrap(),
                tool_call: l.id("<|tool_call_start|>"),
            })
        );
        let plain = TinyTokenizer {
            added: q.added.clone(),
            chat_template: "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}",
        };
        assert_eq!(tags_of(&plain, ""), None);
    }

    /// The block's state at the end of a prompt: open after a start (with the
    /// block's tokens counted after it), closed after an end or a tool call,
    /// closed with no start at all.
    #[test]
    fn a_prompt_ends_inside_the_block_only_after_an_unclosed_start() {
        let t = ThinkTags {
            start: 1,
            end: 2,
            tool_call: Some(3),
        };
        assert_eq!(t.open_at_end(&[9, 1]), Some(0));
        assert_eq!(t.open_at_end(&[1, 9, 9]), Some(2));
        assert_eq!(t.open_at_end(&[1, 9, 2, 9]), None);
        assert_eq!(t.open_at_end(&[1, 9, 3]), None);
        assert_eq!(t.open_at_end(&[9, 9]), None);
        assert_eq!(t.open_at_end(&[1, 2, 1, 9]), Some(1));
        assert_eq!(t.close_ids(), vec![2, 3]);
    }

    /// A render that fails outside `raise_exception` is the request's
    /// ([`Unrenderable`]) when the template renders the engine's own
    /// one-message conversation, and the engine's otherwise: a content the
    /// tiny templates cannot iterate; a template that fails on everything.
    /// And the checkpoint's Qwen template on OpenAI's string tool-call
    /// arguments, when it is in the Hugging Face cache.
    #[test]
    fn a_failure_on_the_requests_data_is_the_requests() {
        let number = json!([{ "role": "user", "content": 5 }]);
        for (t, bos) in [(lfm2(), "<|startoftext|>"), (qwen(), "")] {
            let err = t.render(&number, bos, &Map::new()).unwrap_err();
            assert!(err.downcast_ref::<Unrenderable>().is_some(), "{err:#}");
            assert!(err.downcast_ref::<Rejected>().is_none(), "{err:#}");
            assert!(t.render(&user(), bos, &Map::new()).is_ok());
        }
        let broken = Template::compile("{{ messages[0].content }}{{ 1 + 'a' }}").unwrap();
        let err = broken.render(&user(), "", &Map::new()).unwrap_err();
        assert!(err.downcast_ref::<Unrenderable>().is_none(), "{err:#}");
        assert!(err.downcast_ref::<Rejected>().is_none(), "{err:#}");

        let hub = std::env::var("HF_HUB_CACHE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                    .join(".cache/huggingface/hub")
            });
        let snapshots = hub.join("models--Qwen--Qwen3.8-Flash-Next/snapshots");
        let sources: Vec<String> = std::fs::read_dir(&snapshots)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path().join("chat_template.jinja")).ok())
            .collect();
        if sources.is_empty() {
            eprintln!("skipped: no Qwen3.8-Flash-Next chat template under {snapshots:?}");
        }
        let called = json!([
            { "role": "user", "content": "Weather?" },
            { "role": "assistant", "content": "", "tool_calls": [{ "type": "function",
              "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" } }] },
        ]);
        for src in sources {
            let err = Template::compile(&src)
                .unwrap()
                .render(&called, "", &Map::new())
                .unwrap_err();
            assert!(err.downcast_ref::<Unrenderable>().is_some(), "{err:#}");
        }
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

        let get = |v: Value| request_kwargs(&v, &REQUEST_FIELDS);
        let tools = json!([{ "type": "function", "function": { "name": "f" } }]);
        assert_eq!(
            get(json!({"tools": tools, "enable_thinking": false})).unwrap(),
            kw(&[(TOOLS, tools.clone()), (ENABLE_THINKING, json!(false))])
        );
        assert_eq!(
            get(
                json!({"chat_template_kwargs": {"enable_thinking": null, "reasoning_effort": null}})
            )
            .unwrap(),
            Map::new()
        );
        let e = get(json!({"tools": tools, "chat_template_kwargs": {"tools": []}})).unwrap_err();
        assert!(e.contains("disagree"), "{e}");
        let e = get(json!({"enable_thinking": "yes"})).unwrap_err();
        assert_eq!(e, "'enable_thinking' must be true or false, not \"yes\"");
    }
}
