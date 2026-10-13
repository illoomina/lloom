//! The request front end: a chat request rendered and tokenized once, the
//! replies of ours its history carries given the ids the model sampled, and
//! the thinking switch, the turn checkpoint and the prompt budget acting on
//! spans found from constants read at load (`sequence-reuse-design.md` §2.3d).
//!
//! Everything here runs on the request's thread before the ids reach
//! `Served::complete_until`: the engine and the batch thread see token ids
//! only.

use super::budgets::Dropped;
use super::common::Text;
use super::served::Served;
use super::template::{ThinkTags, ENABLE_THINKING, NOT_READ, PROBE_VALUE, REASONING_EFFORT};
use super::thinking::{self, By, Thinking};
use crate::Decoded;
use anyhow::Result;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The template variable that keeps past replies' reasoning in the history
/// (the Qwen3.6, Qwen3.8 and LFM2.5 templates' switch).
pub const PRESERVE_THINKING: &str = "preserve_thinking";

/// The message fields a template reads a reply's reasoning from: Qwen's
/// `reasoning_content`, LFM2's `thinking` / `reasoning` / `reasoning_content`.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "thinking", "reasoning"];

const PROBE_USER: &str = "lloom-probe-user";
const PROBE_USER_NEXT: &str = "lloom-probe-next";
const PROBE_ANSWER: &str = "lloom-probe-answer";
const PROBE_REASONING: &str = "lloom-probe-reasoning";

/// Units a re-anchoring alignment tries per reply segment before it gives
/// the rest of the segment to the tokenizer.
const MAX_ANCHORS: usize = 8;
/// The bytes an anchor spans at least, so a short common unit (`the`) does
/// not anchor a reply inside unrelated text.
const ANCHOR_BYTES: usize = 8;

/// The log's bounds: replies, and their ids summed.
const LOG_REPLIES: usize = 256;
const LOG_TOKENS: usize = 1 << 20;

/// A request the front end refuses on the client's account (a 400).
#[derive(Debug)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// What the session log keeps beside a reply's sampled ids
/// (`session.keep_logprobs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepLogprobs {
    #[default]
    None,
    /// The sampled token's log-probability: one f32 per token.
    Sampled,
    /// Every position's log-probabilities over the vocabulary.
    Full,
}

impl KeepLogprobs {
    pub fn parse(s: &str) -> Self {
        match s {
            "sampled" => Self::Sampled,
            "full" => Self::Full,
            _ => Self::None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sampled => "sampled",
            Self::Full => "full",
        }
    }

    /// What a run must capture for the log to keep this.
    pub fn capture(self) -> Option<crate::LogprobRequest> {
        match self {
            Self::None => None,
            Self::Sampled => Some(crate::LogprobRequest::default()),
            Self::Full => Some(crate::LogprobRequest { top: 0, rows: true }),
        }
    }
}

/// The `session` group as the front end runs it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SessionOptions {
    /// `preserve_thinking` for every request that does not set it; `None`
    /// is the template's own default.
    pub persist: Option<bool>,
    /// A reply of ours sent back without its reasoning takes it from the log.
    pub restore_reasoning: bool,
    pub keep: KeepLogprobs,
}

impl SessionOptions {
    pub fn of(c: &crate::options::SessionConfig) -> Self {
        Self {
            persist: c.persist_thinking,
            restore_reasoning: c.missing_reasoning == "log",
            keep: KeepLogprobs::parse(&c.keep_logprobs),
        }
    }
}

fn slot(m: Option<bool>) -> usize {
    match m {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    }
}

const MODES: [Option<bool>; 3] = [None, Some(true), Some(false)];

/// One think mode's generation prompt: `enable_thinking` = `kwarg` (where
/// the template reads it) and the engine's switch `want`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mode {
    /// The ids the template's generation prompt encodes to.
    pub template: Vec<u32>,
    /// The ids a prompt ends with as it runs: `template`, with the end
    /// marker the engine appends where it closes the block itself.
    pub gen: Vec<u32>,
    pub thinking: Thinking,
    /// Per `preserve_thinking` (unset, true, false): how many of `gen`'s ids
    /// the next turn's rendering of this turn keeps. The turn checkpoint
    /// goes after them, where the next request diverges.
    pub keep: [Option<usize>; 3],
}

/// Whether a template's prompt depends on `reasoning_effort`, per
/// `enable_thinking` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// The source does not read it.
    NotRead,
    /// Every value renders the same one-message prompt.
    NoEffect,
    /// The prompt depends on it (or the template refuses a value it does not
    /// know, which is the request's to find out).
    Read,
}

/// What the front end reads from a template once, at load.
#[derive(Debug, Clone)]
pub struct Constants {
    modes: [[Option<std::result::Result<Mode, &'static str>>; 3]; 3],
    effort: [Effort; 3],
    /// The template reads `preserve_thinking` and renders a past reply's
    /// reasoning differently by it.
    pub persist_switch: bool,
    /// Whether a past reply's reasoning is in the history by default:
    /// `None` where the template writes no reasoning block at all.
    pub persists: Option<bool>,
    /// The text every message's rendering starts with (`<|im_start|>`):
    /// where the history's spans begin. `None` where the renders share none.
    pub open: Option<String>,
    /// Consecutive tool results share one span (Qwen's `<|im_start|>user`
    /// around several `<tool_response>`s).
    pub groups_tools: bool,
    /// The family's reasoning tags (`Text::think_tags`).
    pub tags: Option<ThinkTags>,
}

impl Constants {
    fn none() -> Self {
        Self {
            modes: Default::default(),
            effort: [Effort::Read; 3],
            persist_switch: false,
            persists: None,
            open: None,
            groups_tools: false,
            tags: None,
        }
    }

    /// The mode for `kwarg` × `want`: `Err` with the reason thinking on
    /// cannot apply; `None` where the template's renders do not nest.
    pub fn mode(
        &self,
        kwarg: Option<bool>,
        want: Option<bool>,
    ) -> Option<std::result::Result<&Mode, &'static str>> {
        self.modes[slot(kwarg)][slot(want)]
            .as_ref()
            .map(|m| m.as_ref().map_err(|e| *e))
    }

    pub fn effort(&self, kwarg: Option<bool>) -> Effort {
        self.effort[slot(kwarg)]
    }

    /// Read them from `text`'s template: a handful of renders of probe
    /// conversations, once.
    pub fn read(text: &Text) -> Self {
        let Some(t) = text.template.as_ref() else {
            return Self::none();
        };
        let bos = text.bos_token.as_str();
        let enc = |s: &str| -> Option<Vec<u32>> {
            Some(text.tokenizer.encode(s, false).ok()?.get_ids().to_vec())
        };
        let render = |m: &Value, kw: &Map<String, Value>, gen: bool| -> Option<String> {
            t.render_with(m, bos, kw, gen).ok()
        };
        let tags = text.think_tags().cloned();
        let reads = |n: &str| t.reads(n) == Some(true);
        let user = json!({ "role": "user", "content": PROBE_USER });
        let next = json!({ "role": "user", "content": PROBE_USER_NEXT });
        let reasoned = |reasoning: bool| -> Value {
            let mut a = json!({ "role": "assistant", "content": PROBE_ANSWER });
            if reasoning {
                for f in REASONING_FIELDS {
                    a[f] = json!(PROBE_REASONING);
                }
            }
            a
        };

        let switch = reads(PRESERVE_THINKING);
        let with_persist =
            |kw: &Map<String, Value>, p: Option<bool>| -> Option<Map<String, Value>> {
                let mut kw = kw.clone();
                match p {
                    None => {}
                    Some(_) if !switch => return None,
                    Some(b) => {
                        kw.insert(PRESERVE_THINKING.into(), Value::Bool(b));
                    }
                }
                Some(kw)
            };

        let mut modes: [[Option<std::result::Result<Mode, &'static str>>; 3]; 3] =
            Default::default();
        for kwarg in MODES {
            let mut kw = Map::new();
            if let Some(k) = kwarg {
                if !reads(ENABLE_THINKING) {
                    continue;
                }
                kw.insert(ENABLE_THINKING.into(), Value::Bool(k));
            }
            for want in MODES {
                let mode = (|| -> Option<std::result::Result<Mode, &'static str>> {
                    let one = json!([user]);
                    let bare = enc(&render(&one, &kw, false)?)?;
                    let mut with = enc(&render(&one, &kw, true)?)?;
                    if with.len() <= bare.len() || !with.starts_with(&bare) {
                        return None;
                    }
                    let template = with[bare.len()..].to_vec();
                    let thinking = match thinking::apply(want, tags.as_ref(), &mut with) {
                        Ok(th) => th,
                        Err(why) => return Some(Err(why)),
                    };
                    let gen = with[bare.len()..].to_vec();
                    let mut keep = [None; 3];
                    for p in MODES {
                        let Some(kwp) = with_persist(&kw, p) else {
                            continue;
                        };
                        let two = json!([user, reasoned(thinking.on), next]);
                        let Some(next_ids) = render(&two, &kwp, true).and_then(|s| enc(&s)) else {
                            continue;
                        };
                        let lcp = with
                            .iter()
                            .zip(&next_ids)
                            .take_while(|(a, b)| a == b)
                            .count();
                        keep[slot(p)] = Some(lcp.saturating_sub(bare.len()).min(gen.len()));
                    }
                    Some(Ok(Mode {
                        template,
                        gen,
                        thinking,
                        keep,
                    }))
                })();
                modes[slot(kwarg)][slot(want)] = mode;
            }
        }

        let mut effort = [Effort::Read; 3];
        for kwarg in MODES {
            effort[slot(kwarg)] = if t.reads(REASONING_EFFORT) == Some(false) {
                Effort::NotRead
            } else {
                let mut kw = Map::new();
                if let Some(k) = kwarg {
                    kw.insert(ENABLE_THINKING.into(), Value::Bool(k));
                }
                let one = json!([user]);
                let mut probe = kw.clone();
                probe.insert(REASONING_EFFORT.into(), json!(PROBE_VALUE));
                match (render(&one, &kw, true), render(&one, &probe, true)) {
                    (Some(a), Some(b)) if a == b => Effort::NoEffect,
                    _ => Effort::Read,
                }
            };
        }

        let history = json!([user, reasoned(true), next]);
        let has_reasoning = |kw: &Map<String, Value>| {
            render(&history, kw, false).map(|s| s.contains(PROBE_REASONING))
        };
        let persists = if tags.is_some() {
            has_reasoning(&Map::new())
        } else {
            None
        };
        let persist_switch = switch && {
            let on = with_persist(&Map::new(), Some(true)).and_then(|kw| has_reasoning(&kw));
            let off = with_persist(&Map::new(), Some(false)).and_then(|kw| has_reasoning(&kw));
            matches!((on, off), (Some(a), Some(b)) if a != b)
        };

        let open = (|| {
            // Each history is a prefix of the next one's rendering: a user
            // turn after a user turn, an assistant turn after a user turn.
            let u = render(&json!([user]), &Map::new(), false)?;
            let ua = render(&json!([user, reasoned(false)]), &Map::new(), false)?;
            let uu = render(&json!([user, next]), &Map::new(), false)?;
            let a = ua.strip_prefix(u.as_str())?;
            let n = uu.strip_prefix(u.as_str())?;
            let common: String = a
                .chars()
                .zip(n.chars())
                .take_while(|(x, y)| x == y)
                .map(|(x, _)| x)
                .collect();
            // The common opening must itself begin each message's rendering
            // and the generation prompt's.
            let gen = render(&json!([user]), &Map::new(), true)?;
            let gen = gen.strip_prefix(u.as_str())?;
            (!common.is_empty() && gen.starts_with(&common)).then_some(common)
        })();

        let groups_tools = (|| {
            let open = open.as_ref()?;
            let call = json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": { "name": "lloom_probe_tool", "arguments": { "lloom_probe_arg": "v" } },
                }],
            });
            let result = json!({ "role": "tool", "content": "lloom-probe-result" });
            let one = render(&json!([user, call, result]), &Map::new(), false)?;
            let two = render(&json!([user, call, result, result]), &Map::new(), false)?;
            Some(one.matches(open.as_str()).count() == two.matches(open.as_str()).count())
        })()
        .unwrap_or(false);

        Self {
            modes,
            effort,
            persist_switch,
            persists,
            open,
            groups_tools,
            tags,
        }
    }
}

/// One reply the session log keeps: the ids the model sampled (the EOS
/// excluded), the text they decode to, and where the reasoning sits in them.
#[derive(Debug, Clone)]
pub struct Reply {
    pub ids: Vec<u32>,
    pub text: String,
    /// The reasoning block's interior, as a range of `ids`.
    pub reasoning: Option<Range<usize>>,
    /// Where the content starts in `ids` (past the block's end marker).
    pub content: usize,
    /// Past the last end marker in `ids`: `content`, unless the reply wrote
    /// the marker again after its block closed.
    pub tail: usize,
    pub reasoning_text: String,
    pub content_text: String,
    /// `session.keep_logprobs = sampled | full`: the sampled token's
    /// log-probability per id.
    pub logprobs: Option<Vec<f32>>,
    /// `session.keep_logprobs = full`: each position's log-probabilities.
    pub rows: Option<Vec<Vec<f32>>>,
}

/// How a message's text matched a logged reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Matched {
    /// The content is the whole reply as we returned it.
    Full,
    /// The content is the reply's content: the reasoning was sent apart or
    /// dropped.
    Content,
}

/// The replies of ours a front end has returned, keyed by the text the
/// client may send back. Bounded; the oldest go first.
#[derive(Default)]
struct ReplyLog {
    entries: VecDeque<Arc<Reply>>,
    by_text: HashMap<String, Arc<Reply>>,
    tokens: usize,
}

impl ReplyLog {
    fn insert(&mut self, reply: Reply) {
        let reply = Arc::new(reply);
        self.tokens += reply.ids.len();
        self.by_text.insert(reply.text.clone(), Arc::clone(&reply));
        let content = reply.content_text.trim();
        if !content.is_empty() && content != reply.text {
            self.by_text.insert(content.to_string(), Arc::clone(&reply));
        }
        self.entries.push_back(reply);
        while self.entries.len() > LOG_REPLIES
            || (self.tokens > LOG_TOKENS && self.entries.len() > 1)
        {
            let Some(old) = self.entries.pop_front() else {
                break;
            };
            self.tokens -= old.ids.len();
            for key in [old.text.as_str(), old.content_text.trim()] {
                if self.by_text.get(key).is_some_and(|r| Arc::ptr_eq(r, &old)) {
                    self.by_text.remove(key);
                }
            }
        }
    }

    fn find(&self, content: &str) -> Option<(Arc<Reply>, Matched)> {
        if let Some(r) = self.by_text.get(content) {
            if r.text == content {
                return Some((Arc::clone(r), Matched::Full));
            }
        }
        let r = self.by_text.get(content.trim())?;
        (r.content_text.trim() == content.trim()).then(|| (Arc::clone(r), Matched::Content))
    }
}

/// What a rendered prompt becomes before it is encoded.
pub type Expand<'a> = &'a dyn Fn(&str) -> Result<String>;

/// A chat request as the front end takes it.
pub struct Request<'a> {
    pub messages: &'a Value,
    /// The request's template variables, `enable_thinking` set where the
    /// template reads it (`thinking::set_kwarg`).
    pub kwargs: Map<String, Value>,
    /// The thinking switch.
    pub want: Option<bool>,
    /// Encode with the checkpoint's BOS.
    pub add_bos: bool,
    /// What the rendered text becomes before it is encoded (the images'
    /// expansions).
    pub expand: Option<Expand<'a>>,
}

/// The replies of ours a prompt carries, and what of them it ran as sampled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Spliced {
    /// Assistant messages in the request.
    pub replies: usize,
    /// Those that are a reply the log holds.
    pub matched: usize,
    /// Of those, the ones whose reasoning came from the log.
    pub restored: usize,
    /// The matched replies' ids.
    pub reply_tokens: usize,
    /// The ids the prompt took from the log.
    pub spliced_tokens: usize,
}

/// Where each message's rendering starts in the prompt's ids.
#[derive(Debug, Clone)]
pub struct Blocks {
    /// The ids before the first message's span: the BOS, the system turn,
    /// a tools preamble.
    pub head: usize,
    /// Per message, the start of the span it opens; `None` for a message
    /// whose rendering is in the head or in the previous span.
    pub starts: Vec<Option<usize>>,
    roles: Vec<String>,
}

/// A prompt the front end built.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub ids: Vec<u32>,
    /// The rendered text, with the end marker's text where the engine closed
    /// the block.
    pub text: String,
    /// The variables the render ran with.
    pub kwargs: Map<String, Value>,
    pub thinking: Option<Thinking>,
    /// The turn checkpoint's distance from the end (`DecodeParams::turn_tail`).
    pub turn_tail: usize,
    pub blocks: Option<Blocks>,
    pub spliced: Spliced,
    /// `session.keep_logprobs`: the logged log-probability of each id the
    /// prompt took from a reply, `None` elsewhere.
    pub logprobs: Option<Vec<Option<f32>>>,
    /// The template renders this prompt took (1 by construction).
    pub renders: u64,
}

/// A prompt fitted to a budget by dropping its oldest turns.
pub struct Fitted {
    /// The messages kept: the leading system message, then the rest from
    /// the first kept user turn.
    pub messages: Vec<Value>,
    pub ids: Vec<u32>,
    pub dropped: Dropped,
}

/// The front end: the constants, the session log, and counters a test reads.
pub struct Front {
    constants: Option<Constants>,
    options: SessionOptions,
    log: Mutex<ReplyLog>,
    tokenizations: AtomicU64,
}

impl Front {
    /// Read `engine`'s constants (once: this is the load-time read).
    pub fn new(engine: &dyn Served, options: SessionOptions) -> Self {
        Self {
            constants: engine.text().map(Constants::read),
            options,
            log: Mutex::new(ReplyLog::default()),
            tokenizations: AtomicU64::new(0),
        }
    }

    pub fn constants(&self) -> Option<&Constants> {
        self.constants.as_ref()
    }

    pub fn options(&self) -> SessionOptions {
        self.options
    }

    /// Tokenization passes over a rendered prompt so far (one per prompt).
    pub fn tokenizations(&self) -> u64 {
        self.tokenizations.load(Ordering::Relaxed)
    }

    /// Replies the log holds.
    pub fn logged(&self) -> usize {
        self.log.lock().entries.len()
    }

    /// The `[session: …]` line a front end prints at load.
    pub fn line(&self) -> String {
        let Some(c) = &self.constants else {
            return "no template constants (the family keeps no text side)".into();
        };
        let persists = match c.persists {
            Some(true) => "persist",
            Some(false) => "strip",
            None => "none",
        };
        let gen = |k, w| match c.mode(k, w) {
            Some(Ok(m)) => format!(
                "{}/{}",
                m.gen.len(),
                m.keep[0].map_or("-".into(), |k| (m.gen.len() - k).to_string())
            ),
            Some(Err(_)) => "refused".into(),
            None => "-".into(),
        };
        format!(
            "reasoning_default={persists} switch={} persist_thinking={} missing_reasoning={} keep_logprobs={} \
             open={:?} groups_tools={} gen/tail[unset,on,off]=[{}, {}, {}]",
            c.persist_switch,
            self.options
                .persist
                .map_or("template".into(), |p| p.to_string()),
            if self.options.restore_reasoning {
                "log"
            } else {
                "as-sent"
            },
            self.options.keep.name(),
            c.open,
            c.groups_tools,
            gen(None, None),
            gen(None, Some(true)),
            gen(None, Some(false)),
        )
    }

    /// The reason a `reasoning_effort` in `kwargs` is refused: the template
    /// does not read it, or renders the same prompt for any value of it.
    fn check_effort(
        &self,
        engine: &dyn Served,
        kwargs: &Map<String, Value>,
    ) -> std::result::Result<(), Refused> {
        let Some(value) = kwargs.get(REASONING_EFFORT) else {
            return Ok(());
        };
        let Some(c) = &self.constants else {
            return Ok(());
        };
        let kwarg = kwargs.get(ENABLE_THINKING).and_then(Value::as_bool);
        let family = engine.capabilities().family;
        let why = match c.effort(kwarg) {
            Effort::Read => return Ok(()),
            Effort::NotRead => format!("{family}'s chat template does not read {REASONING_EFFORT}"),
            Effort::NoEffect => format!(
                "{family}'s chat template renders the same prompt for any {REASONING_EFFORT} with \
                 {ENABLE_THINKING}={}",
                kwargs.get(ENABLE_THINKING).unwrap_or(&Value::Null)
            ),
        };
        Err(Refused(format!("{REASONING_EFFORT}={value}: {why}")))
    }

    /// Build a request's prompt: one render, one tokenization pass, the
    /// replies of ours its history carries taking the logged ids.
    pub fn prompt(&self, engine: &dyn Served, req: Request<'_>) -> Result<Prompt> {
        self.check_effort(engine, &req.kwargs)
            .map_err(anyhow::Error::new)?;
        let (Some(text), Some(c)) = (engine.text(), self.constants.as_ref()) else {
            return self.plain(engine, req);
        };
        let Some(template) = text.template.as_ref() else {
            return self.plain(engine, req);
        };
        let mut kwargs = req.kwargs;
        if let (Some(p), false) = (self.options.persist, kwargs.contains_key(PRESERVE_THINKING)) {
            if c.persist_switch {
                kwargs.insert(PRESERVE_THINKING.into(), Value::Bool(p));
            }
        }
        let empty = Vec::new();
        let list = req.messages.as_array().unwrap_or(&empty);

        // The replies of ours, matched by text; a matched reply with a
        // reasoning block is given to the template as the model wrote it,
        // reasoning and content apart.
        let mut spliced = Spliced::default();
        let mut matched: Vec<(usize, Arc<Reply>)> = Vec::new();
        let mut rewritten: Option<Vec<Value>> = None;
        // Whether the template keeps a past reply's reasoning: its switch as
        // set, else its default; a reply after the last user turn is kept
        // by every template that writes the block.
        let persists = kwargs
            .get(PRESERVE_THINKING)
            .and_then(Value::as_bool)
            .filter(|_| c.persist_switch)
            .or(c.persists)
            .unwrap_or(false);
        let last_user = list
            .iter()
            .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"));
        {
            let log = self.log.lock();
            for (i, m) in list.iter().enumerate() {
                if m.get("role").and_then(Value::as_str) != Some("assistant") {
                    continue;
                }
                spliced.replies += 1;
                let Some(content) = m.get("content").and_then(Value::as_str) else {
                    continue;
                };
                let Some((reply, how)) = log.find(content) else {
                    continue;
                };
                spliced.matched += 1;
                spliced.reply_tokens += reply.ids.len();
                let has_field = REASONING_FIELDS.iter().any(|f| {
                    m.get(*f)
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                });
                // Kept, the reply is rendered as the model wrote it,
                // reasoning and content apart. Stripped, the message goes to
                // the template as sent: what survives of the whole text is
                // the template's own split (`content.split('</think>')[-1]`
                // keeps what follows the *last* marker, and a reply can write
                // one again after the engine closed its block), and the
                // sampled ids are spliced wherever its text is theirs.
                let kept = persists || last_user.is_some_and(|u| i > u);
                let restore = kept
                    && reply.reasoning.is_some()
                    && !has_field
                    && (how == Matched::Full || self.options.restore_reasoning);
                if restore {
                    if how == Matched::Content {
                        spliced.restored += 1;
                    }
                    let msgs = rewritten.get_or_insert_with(|| list.clone());
                    let mut msg = m.clone();
                    msg["content"] = json!(reply.content_text);
                    for f in REASONING_FIELDS {
                        msg[f] = json!(reply.reasoning_text);
                    }
                    msgs[i] = msg;
                }
                matched.push((i, reply));
            }
        }
        let shown: std::borrow::Cow<'_, [Value]> = match &rewritten {
            Some(v) => std::borrow::Cow::Borrowed(v.as_slice()),
            None => std::borrow::Cow::Borrowed(list.as_slice()),
        };
        let renders_before = template.renders();
        let rendered = template.render_with(&json!(&*shown), &text.bos_token, &kwargs, true)?;
        let renders = template.renders() - renders_before;
        let rendered = match req.expand {
            Some(f) => f(&rendered)?,
            None => rendered,
        };

        // The spans: where each message's rendering starts in the text.
        let text_blocks = c
            .open
            .as_deref()
            .and_then(|open| text_blocks(&rendered, open, &shown, c.groups_tools));

        // The logged replies' ids, aligned within each one's own span.
        let mut runs: Vec<Run> = Vec::new();
        if let Some(tb) = &text_blocks {
            for (i, reply) in &matched {
                let Some(region) = tb.region(*i, rendered.len()) else {
                    continue;
                };
                align_reply(text, &rendered, region, reply, &mut runs);
            }
        }
        runs.sort_by_key(|r| r.text.start);

        // One pass over the text: gaps encoded, runs taken as logged.
        let cuts: Vec<usize> = text_blocks
            .as_ref()
            .map(|tb| tb.starts.clone())
            .unwrap_or_default();
        let mut ids: Vec<u32> = Vec::with_capacity(rendered.len() / 3);
        let mut logprobs: Vec<Option<f32>> = Vec::new();
        let keep_lp = self.options.keep != KeepLogprobs::None;
        let mut cut_ids: Vec<(usize, usize)> = Vec::with_capacity(cuts.len());
        let mut pos = 0usize;
        let encode_gap = |from: usize,
                          to: usize,
                          ids: &mut Vec<u32>,
                          logprobs: &mut Vec<Option<f32>>,
                          cut_ids: &mut Vec<(usize, usize)>|
         -> Result<()> {
            let mut at = from;
            for &cut in cuts.iter().filter(|&&c| c >= from && c < to) {
                encode_into(text, &rendered[at..cut], ids)?;
                cut_ids.push((cut, ids.len()));
                at = cut;
            }
            encode_into(text, &rendered[at..to], ids)?;
            if keep_lp {
                logprobs.resize(ids.len(), None);
            }
            Ok(())
        };
        for run in &runs {
            if run.text.start < pos {
                continue;
            }
            encode_gap(pos, run.text.start, &mut ids, &mut logprobs, &mut cut_ids)?;
            if cuts.contains(&run.text.start) {
                cut_ids.push((run.text.start, ids.len()));
            }
            ids.extend_from_slice(&run.reply.ids[run.ids.clone()]);
            spliced.spliced_tokens += run.ids.len();
            if keep_lp {
                match &run.reply.logprobs {
                    Some(lp) => logprobs.extend(lp[run.ids.clone()].iter().map(|&v| Some(v))),
                    None => logprobs.resize(ids.len(), None),
                }
            }
            pos = run.text.end;
        }
        encode_gap(pos, rendered.len(), &mut ids, &mut logprobs, &mut cut_ids)?;
        self.tokenizations.fetch_add(1, Ordering::Relaxed);
        let mut head_shift = 0;
        if req.add_bos {
            if let Some(bos) = text.bos_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                    if keep_lp {
                        logprobs.insert(0, None);
                    }
                    head_shift = 1;
                }
            }
        }

        // The thinking switch on the generation prompt's span.
        let kwarg = kwargs.get(ENABLE_THINKING).and_then(Value::as_bool);
        let mut shown_text = rendered;
        let mode = c.mode(kwarg, req.want);
        let (thinking, gen_len) = match mode {
            Some(Ok(m)) if ids.ends_with(&m.template) => {
                let n = ids.len() - m.template.len();
                ids.truncate(n);
                ids.extend_from_slice(&m.gen);
                if m.thinking.by == By::Engine {
                    for &id in &m.gen[m.template.len()..] {
                        shown_text.push_str(&engine.token_str_checked(id)?);
                    }
                }
                (m.thinking, Some(m))
            }
            Some(Err(why)) if req.want == Some(true) => {
                return Err(cannot_think(engine, why));
            }
            _ => {
                let before = ids.len();
                let th = thinking::apply(req.want, c.tags.as_ref(), &mut ids)
                    .map_err(|why| cannot_think(engine, why))?;
                for &id in &ids[before..] {
                    shown_text.push_str(&engine.token_str_checked(id)?);
                }
                (th, None)
            }
        };
        if keep_lp {
            logprobs.resize(ids.len(), None);
        }

        let blocks = text_blocks.map(|tb| {
            let at = |t: usize| {
                cut_ids
                    .iter()
                    .find(|(c, _)| *c == t)
                    .map(|(_, i)| i + head_shift)
            };
            Blocks {
                head: tb.head.and_then(at).unwrap_or(ids.len()),
                starts: tb.message.iter().map(|s| s.and_then(at)).collect(),
                roles: shown
                    .iter()
                    .map(|m| {
                        m.get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect(),
            }
        });

        // The turn checkpoint: where the next turn's rendering of this one
        // diverges, from the constant; the last user turn's start else.
        let persist = kwargs.get(PRESERVE_THINKING).and_then(Value::as_bool);
        let turn_tail = match gen_len {
            Some(m) => m.keep[slot(persist)]
                .or(m.keep[0])
                .map_or(m.gen.len(), |k| m.gen.len() - k),
            None => blocks
                .as_ref()
                .and_then(|b| {
                    b.starts
                        .iter()
                        .zip(&b.roles)
                        .filter(|(_, r)| r.as_str() == "user")
                        .filter_map(|(s, _)| *s)
                        .next_back()
                })
                .map_or(0, |s| ids.len() - s),
        };
        Ok(Prompt {
            ids,
            text: shown_text,
            kwargs,
            thinking: Some(thinking),
            turn_tail,
            blocks,
            spliced,
            logprobs: keep_lp.then_some(logprobs),
            renders,
        })
    }

    /// A family without a text side: render (with the effort's check),
    /// encode and switch through the seam, as before.
    fn plain(&self, engine: &dyn Served, req: Request<'_>) -> Result<Prompt> {
        let before = engine.template_renders();
        let rendered = engine.render_checked(req.messages, &req.kwargs, &[REASONING_EFFORT])?;
        let renders = engine.template_renders() - before;
        if let Some(i) = rendered.ignored.first() {
            let family = engine.capabilities().family;
            let why = if i.reason == NOT_READ {
                format!("{family}'s chat template does not read {REASONING_EFFORT}")
            } else {
                format!(
                    "{family}'s chat template renders the same prompt for any {REASONING_EFFORT} \
                     with {ENABLE_THINKING}={}",
                    req.kwargs.get(ENABLE_THINKING).unwrap_or(&Value::Null)
                )
            };
            return Err(anyhow::Error::new(Refused(format!(
                "{REASONING_EFFORT}={}: {why}",
                i.value
            ))));
        }
        let rendered = match req.expand {
            Some(f) => f(&rendered.text)?,
            None => rendered.text,
        };
        let mut ids = engine.encode(&rendered, req.add_bos)?;
        self.tokenizations.fetch_add(1, Ordering::Relaxed);
        let tags = engine.think_tags();
        let mut shown = rendered;
        let before = ids.len();
        let th = thinking::apply(req.want, tags.as_ref(), &mut ids)
            .map_err(|why| cannot_think(engine, why))?;
        for &id in &ids[before..] {
            shown.push_str(&engine.token_str_checked(id)?);
        }
        let turn_tail = thinking::checkpoint_tail_checked(engine, &req.kwargs, req.want, &ids)?;
        Ok(Prompt {
            ids,
            text: shown,
            kwargs: req.kwargs,
            thinking: Some(th),
            turn_tail,
            blocks: None,
            spliced: Spliced::default(),
            logprobs: None,
            renders,
        })
    }

    /// Why `session.persist_thinking` cannot apply on this template (it has
    /// no switch that changes a past reply's reasoning), when it is set.
    pub fn persist_unsupported(&self) -> Option<String> {
        self.options.persist?;
        match &self.constants {
            Some(c) if c.persist_switch => None,
            Some(c) if c.persists.is_none() => {
                Some("the chat template writes no reasoning into the history".into())
            }
            _ => Some(format!(
                "the chat template has no {PRESERVE_THINKING} switch that changes a past reply's reasoning"
            )),
        }
    }

    /// `prompt` fitted to `budget` ids by dropping its oldest turns whole —
    /// the leading system message kept, the kept part starting at a user
    /// turn — as spans of its ids. `None` where the prompt's spans are not
    /// known (the caller renders the shorter conversations instead) or when
    /// even the last user turn does not fit (`Some(None)`).
    pub fn fit(
        &self,
        prompt: &Prompt,
        messages: &[Value],
        budget: usize,
    ) -> Option<Option<Fitted>> {
        let b = prompt.blocks.as_ref()?;
        if b.starts.len() != messages.len() {
            return None;
        }
        let role = |m: &Value| {
            m.get("role")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        let head = usize::from(messages.first().is_some_and(|m| role(m) == "system"));
        for drop in 1..messages.len().saturating_sub(head) {
            let first = head + drop;
            if role(&messages[first]) != "user" {
                continue;
            }
            let start = b.starts[first]?;
            let mut ids = prompt.ids[..b.head].to_vec();
            ids.extend_from_slice(&prompt.ids[start..]);
            if ids.len() <= budget {
                let mut kept = messages[..head].to_vec();
                kept.extend_from_slice(&messages[first..]);
                let dropped = Dropped {
                    tokens: prompt.ids.len() - ids.len(),
                    messages: drop,
                };
                return Some(Some(Fitted {
                    messages: kept,
                    ids,
                    dropped,
                }));
            }
        }
        Some(None)
    }

    /// Keep a reply in the session log: the ids `decoded` sampled after a
    /// prompt whose thinking was `thinking`, with what `capture` kept of
    /// their logprobs. A reply whose text is not its ids' decoding (a stop
    /// string cut it) is not kept. Returns whether it was.
    pub fn record(
        &self,
        engine: &dyn Served,
        thinking: Option<Thinking>,
        decoded: &Decoded,
    ) -> bool {
        let Some(text) = engine.text() else {
            return false;
        };
        if decoded.tokens.is_empty() {
            return false;
        }
        let Some(full) = decode(text, &decoded.tokens) else {
            return false;
        };
        // The ids the client's text is: all of them, or — where the text
        // stops short (a stop string, a character the run ended inside) —
        // the longest prefix of whole units that decodes to it.
        let n = if full == decoded.text {
            decoded.tokens.len()
        } else {
            let Some(u) = units(text, &decoded.tokens, 0) else {
                return false;
            };
            let mut acc = String::new();
            let mut kept = None;
            for (r, s) in &u {
                acc.push_str(s);
                if !decoded.text.starts_with(acc.as_str()) {
                    break;
                }
                if acc.len() == decoded.text.len() {
                    kept = Some(r.end);
                    break;
                }
            }
            match kept {
                Some(k) => k,
                None => return false,
            }
        };
        let full = decoded.text.clone();
        let tags = self.constants.as_ref().and_then(|c| c.tags.as_ref());
        let ids = &decoded.tokens[..n];
        let open = thinking.is_some_and(|t| t.on);
        // A reasoning block is the reply's only once it closed: a reply cut
        // inside it (`max_tokens`) is text, as the template takes it.
        let (reasoning, content) = match tags {
            Some(t) if open => match ids.iter().position(|&id| id == t.end) {
                Some(e) => (Some(0..e), e + 1),
                None => (None, 0),
            },
            Some(t) => match ids.iter().position(|&id| id == t.start) {
                Some(s) => match ids[s + 1..].iter().position(|&id| id == t.end) {
                    Some(e) => (Some(s + 1..s + 1 + e), s + 2 + e),
                    None => (None, 0),
                },
                None => (None, 0),
            },
            None => (None, 0),
        };
        let reasoning_text = match &reasoning {
            Some(r) => decode(text, &ids[r.clone()]).unwrap_or_default(),
            None => String::new(),
        };
        let content = content.min(ids.len());
        // Past the last end marker: where a stripping template's split of
        // the whole text starts the content it keeps.
        let tail = match (tags, &reasoning) {
            (Some(t), Some(_)) => ids
                .iter()
                .rposition(|&id| id == t.end)
                .map_or(content, |e| e + 1)
                .max(content),
            _ => content,
        };
        let content_text = decode(text, &ids[content..]).unwrap_or_default();
        let keep = self.options.keep;
        let (logprobs, rows) = match (&decoded.logprobs, keep) {
            (Some(lp), KeepLogprobs::Sampled) if lp.sampled.len() >= n => {
                (Some(lp.sampled[..n].to_vec()), None)
            }
            (Some(lp), KeepLogprobs::Full) if lp.sampled.len() >= n => (
                Some(lp.sampled[..n].to_vec()),
                lp.rows.as_ref().map(|r| r[..n.min(r.len())].to_vec()),
            ),
            _ => (None, None),
        };
        self.log.lock().insert(Reply {
            ids: ids.to_vec(),
            text: full,
            reasoning,
            content,
            tail,
            reasoning_text,
            content_text,
            logprobs,
            rows,
        });
        true
    }

    /// The reply the log holds for `content`, for a test or a report.
    pub fn reply(&self, content: &str) -> Option<Arc<Reply>> {
        self.log.lock().find(content).map(|(r, _)| r)
    }
}

/// Thinking on where the prompt cannot think: the client's.
fn cannot_think(engine: &dyn Served, why: &str) -> anyhow::Error {
    anyhow::Error::new(Refused(format!(
        "{ENABLE_THINKING}=true: {}: {why}",
        engine.capabilities().family
    )))
}

/// Why this process cannot keep per-token logprobs: the body it decodes
/// through reads no row of logits on the host (`lloom::nn::refuse`). On the
/// K=1 body a logprobs run takes the blocking arm, so only the speculative
/// body and the batch scheduler refuse.
pub fn logprobs_refused(engine: &dyn Served) -> Option<String> {
    logprobs_refused_on(engine.decode_body())
}

/// [`logprobs_refused`] for a process decoding through `body`.
pub fn logprobs_refused_on(body: lloom::nn::Body) -> Option<String> {
    use lloom::nn::Body;
    let (body, name) = match body {
        Body::Step { .. } => (Body::Step { ring: false }, "the K=1 body"),
        Body::Window => (Body::Window, "the speculative body (--dspark, --mtp)"),
        b @ (Body::BatchedStep | Body::BatchedWindow) => (b, "the batch scheduler (--batch N)"),
    };
    let asks = lloom::nn::Asks {
        sampling: &lloom::generation::Sampling::ArgMax,
        processors: false,
        budget: false,
        digest: false,
        logprobs: true,
    };
    lloom::nn::refuse(body, &asks)
        .err()
        .map(|e| format!("this process decodes through {name}: {e}"))
}

fn decode(text: &Text, ids: &[u32]) -> Option<String> {
    text.tokenizer.decode(ids, false).ok()
}

fn encode_into(text: &Text, s: &str, ids: &mut Vec<u32>) -> Result<()> {
    if s.is_empty() {
        return Ok(());
    }
    let e = text
        .tokenizer
        .encode(s, false)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    ids.extend_from_slice(e.get_ids());
    Ok(())
}

/// Where each message's rendering starts in the rendered text.
struct TextBlocks {
    /// Every opening in the text, in order.
    starts: Vec<usize>,
    /// The first message span's start: where the head ends.
    head: Option<usize>,
    /// Per message, the opening its rendering starts at.
    message: Vec<Option<usize>>,
}

impl TextBlocks {
    /// The text of message `i`'s span: its opening to the next one.
    fn region(&self, i: usize, end: usize) -> Option<Range<usize>> {
        let start = (*self.message.get(i)?)?;
        let next = self
            .starts
            .iter()
            .copied()
            .find(|&s| s > start)
            .unwrap_or(end);
        Some(start..next)
    }
}

/// The openings of `rendered` attributed to `messages`: a leading system
/// message is in the head, a tool result after another shares its span where
/// the template groups them, every other message opens one, and the
/// generation prompt opens the last. Checked: every message's text must lie in
/// its span, or nothing is attributed.
fn text_blocks(
    rendered: &str,
    open: &str,
    messages: &[Value],
    groups_tools: bool,
) -> Option<TextBlocks> {
    fn role(m: &Value) -> &str {
        m.get("role").and_then(Value::as_str).unwrap_or("")
    }
    let starts: Vec<usize> = rendered.match_indices(open).map(|(i, _)| i).collect();
    let opens: Vec<bool> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            !(i == 0 && role(m) == "system"
                || groups_tools && role(m) == "tool" && i > 0 && role(&messages[i - 1]) == "tool")
        })
        .collect();
    let n_open = opens.iter().filter(|&&o| o).count();
    let head = starts.len().checked_sub(n_open + 1)?;
    let mut message = Vec::with_capacity(messages.len());
    let mut k = head;
    for (i, &o) in opens.iter().enumerate() {
        if o {
            message.push(Some(starts[k]));
            k += 1;
        } else if i == 0 {
            message.push(None);
        } else {
            message.push(message[i - 1]);
        }
    }
    let tb = TextBlocks {
        head: starts.get(head).copied(),
        starts: starts.clone(),
        message,
    };
    for (i, m) in messages.iter().enumerate() {
        let Some(content) = m.get("content").and_then(Value::as_str) else {
            continue;
        };
        let probe = content.rsplit("</think>").next().unwrap_or("").trim();
        if probe.is_empty() || !opens[i] {
            continue;
        }
        let region = tb.region(i, rendered.len())?;
        if !rendered[region].contains(probe) {
            return None;
        }
    }
    Some(tb)
}

/// A run of a reply's ids whose text lies contiguously in the rendered text.
#[derive(Debug, Clone)]
struct Run {
    text: Range<usize>,
    ids: Range<usize>,
    reply: Arc<Reply>,
}

/// A reply's ids cut into units whose decodings concatenate to the reply's
/// (a multi-byte character's tokens stay together).
fn units(text: &Text, ids: &[u32], base: usize) -> Option<Vec<(Range<usize>, String)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let mut j = i + 1;
        let s = loop {
            let s = decode(text, &ids[i..j])?;
            if !s.ends_with('\u{FFFD}') || j == ids.len() || j - i >= 8 {
                break s;
            }
            j += 1;
        };
        out.push((base + i..base + j, s));
        i = j;
    }
    let whole = decode(text, ids)?;
    let joined: String = out.iter().map(|(_, s)| s.as_str()).collect();
    (joined == whole).then_some(out)
}

/// Align `reply`'s segments — anything before its reasoning block, the
/// block's interior, its end marker, the content — inside `region` of
/// `rendered`, pushing a run per stretch whose text lies there contiguously.
fn align_reply(
    text: &Text,
    rendered: &str,
    region: Range<usize>,
    reply: &Arc<Reply>,
    runs: &mut Vec<Run>,
) {
    let n = reply.ids.len();
    let mut segments: Vec<Range<usize>> = Vec::new();
    match &reply.reasoning {
        Some(r) => {
            if r.start > 0 {
                segments.push(0..r.start);
            }
            segments.push(r.clone());
            if r.end < reply.content {
                segments.push(r.end..reply.content);
            }
            // A stripped render keeps only what follows the last marker.
            if reply.content < reply.tail {
                segments.push(reply.content..reply.tail);
            }
            segments.push(reply.tail..n);
        }
        None => segments.push(0..n),
    }
    let mut cursor = region.start;
    for seg in segments {
        if seg.is_empty() {
            continue;
        }
        let Some(u) = units(text, &reply.ids[seg.clone()], seg.start) else {
            continue;
        };
        align_units(rendered, &region, &mut cursor, &u, reply, runs);
    }
}

fn align_units(
    rendered: &str,
    region: &Range<usize>,
    cursor: &mut usize,
    units: &[(Range<usize>, String)],
    reply: &Arc<Reply>,
    runs: &mut Vec<Run>,
) {
    let mut k = 0;
    let mut tries = 0;
    while k < units.len() && tries < MAX_ANCHORS {
        tries += 1;
        let Some(a) = (k..units.len()).find(|&u| !units[u].1.trim().is_empty()) else {
            break;
        };
        let mut anchor = String::new();
        let mut b = a;
        while b < units.len() && anchor.len() < ANCHOR_BYTES {
            anchor.push_str(&units[b].1);
            b += 1;
        }
        let Some(off) = rendered[*cursor..region.end].find(&anchor) else {
            k = a + 1;
            continue;
        };
        let mut p = *cursor + off;
        let mut lo = a;
        // The units skipped to reach the anchor join the run only where they
        // fill exactly the text since this reply's previous run (the
        // whitespace around a reasoning block's end marker); never the
        // template's own text before the reply.
        if a > k && *cursor > region.start {
            let gap: String = units[k..a].iter().map(|(_, s)| s.as_str()).collect();
            if rendered[*cursor..p] == gap {
                p = *cursor;
                lo = k;
            }
        }
        let start = p;
        let mut hi = lo;
        while hi < units.len() && rendered[p..region.end].starts_with(units[hi].1.as_str()) {
            p += units[hi].1.len();
            hi += 1;
        }
        if hi > lo {
            runs.push(Run {
                text: start..p,
                ids: units[lo].0.start..units[hi - 1].0.end,
                reply: Arc::clone(reply),
            });
        }
        *cursor = p;
        k = hi.max(a + 1);
    }
}

#[cfg(test)]
#[path = "front_tests.rs"]
mod tests;
