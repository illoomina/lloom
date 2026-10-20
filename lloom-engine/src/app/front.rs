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
use super::template::{ThinkTags, ENABLE_THINKING, NOT_READ, PROBE_VALUE, REASONING_EFFORT, TOOLS};
use super::thinking::{self, By, Thinking};
use crate::Decoded;
use anyhow::Result;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
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

/// How a past reply of ours whose run ended inside an open reasoning block
/// reaches the next turn's template (`session.incomplete_reasoning`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Incomplete {
    /// As the text the client got: its reasoning is the message's content.
    Text,
    /// As the open reasoning it is: given to the template apart, which
    /// closes the block (or drops it, where it strips reasoning).
    #[default]
    Reasoning,
    /// As `Text`, with the turn checkpoint before the divergence such a
    /// reply's re-render makes.
    Recover,
}

impl Incomplete {
    pub fn parse(s: &str) -> Self {
        match s {
            "text" => Self::Text,
            "recover" => Self::Recover,
            _ => Self::Reasoning,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::Recover => "recover",
        }
    }
}

/// The `session` group as the front end runs it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SessionOptions {
    /// `preserve_thinking` for every request that does not set it; `None`
    /// is `true` where the template has the switch (`Front::persist`).
    pub persist: Option<bool>,
    /// A reply of ours sent back without its reasoning takes it from the log.
    pub restore_reasoning: bool,
    pub keep: KeepLogprobs,
    pub incomplete: Incomplete,
}

impl SessionOptions {
    pub fn of(c: &crate::options::SessionConfig) -> Self {
        Self {
            persist: c.persist_thinking,
            restore_reasoning: c.missing_reasoning == "log",
            keep: KeepLogprobs::parse(&c.keep_logprobs),
            incomplete: Incomplete::parse(&c.incomplete_reasoning),
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
    /// As `keep`, for a reply whose run ended inside the block this prompt
    /// opens, rendered as the text it is (`Incomplete::Recover`).
    pub keep_open: [Option<usize>; 3],
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
    /// The text the template writes to end an assistant turn, and the single
    /// id it encodes to: the end-of-turn token. `None` where the template's
    /// ending is not one token, or does not encode to the id the model would
    /// have sampled. A matched turn's span runs through it
    /// ([`emit_turn`]).
    eot: Option<(u32, String)>,
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
            eot: None,
        }
    }

    /// The end-of-turn token: its id and the text the template writes for it.
    pub fn eot(&self) -> Option<(u32, &str)> {
        self.eot.as_ref().map(|(id, s)| (*id, s.as_str()))
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
                    let mut keep_open = [None; 3];
                    let kept = |reply: &Value, kwp: &Map<String, Value>| -> Option<usize> {
                        let two = json!([user, reply, next]);
                        let next_ids = render(&two, kwp, true).and_then(|s| enc(&s))?;
                        let lcp = with
                            .iter()
                            .zip(&next_ids)
                            .take_while(|(a, b)| a == b)
                            .count();
                        Some(lcp.saturating_sub(bare.len()).min(gen.len()))
                    };
                    for p in MODES {
                        let Some(kwp) = with_persist(&kw, p) else {
                            continue;
                        };
                        keep[slot(p)] = kept(&reasoned(thinking.on), &kwp);
                        keep_open[slot(p)] = kept(&reasoned(false), &kwp);
                    }
                    Some(Ok(Mode {
                        template,
                        gen,
                        thinking,
                        keep,
                        keep_open,
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

        // The end-of-turn token: what the template writes after an assistant
        // turn's body. Read by rendering one assistant turn whose content is
        // a probe, and taking what follows the probe up to the next message's
        // opening. It counts only if it encodes to exactly one id, and that
        // id is one of the checkpoint's EOS ids — then it is the token the
        // model itself samples to end a turn, so a matched turn's span may run
        // through it (`emit_turn`).
        let eot = (|| {
            let two = json!([
                user,
                json!({ "role": "assistant", "content": PROBE_ANSWER })
            ]);
            let s = render(&two, &Map::new(), false)?;
            let at = s.rfind(PROBE_ANSWER)? + PROBE_ANSWER.len();
            let rest = &s[at..];
            // The template's own ending for the turn, before any next
            // message's opening and without the whitespace that follows it.
            let marker = match open.as_deref().and_then(|o| rest.find(o)) {
                Some(i) => &rest[..i],
                None => rest,
            }
            .trim_end();
            let ids = enc(marker)?;
            match ids.as_slice() {
                [id] if text.eos_ids.contains(id) => Some((*id, marker.to_string())),
                _ => None,
            }
        })();

        Self {
            modes,
            effort,
            persist_switch,
            persists,
            open,
            groups_tools,
            tags,
            eot,
        }
    }
}

/// Where a reply's reasoning block stands in its ids, as the model produced
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// The reply has no reasoning block.
    None,
    /// Opened (by the prompt, or a start marker the model sampled) and not
    /// closed: the run ended inside it. `interior` is the reasoning.
    Open { interior: Range<usize> },
    /// Closed by the end marker at `end`, an index into the ids.
    Closed { interior: Range<usize>, end: usize },
}

impl Block {
    /// The interior of a block the reply closed.
    pub fn closed(&self) -> Option<&Range<usize>> {
        match self {
            Self::Closed { interior, .. } => Some(interior),
            _ => None,
        }
    }

    /// The interior, closed or open.
    pub fn interior(&self) -> Option<&Range<usize>> {
        match self {
            Self::None => None,
            Self::Open { interior } | Self::Closed { interior, .. } => Some(interior),
        }
    }
}

/// One reply the session log keeps: the ids the model sampled (the EOS
/// excluded), the text they decode to, and where the reasoning sits in them.
#[derive(Debug, Clone)]
pub struct Reply {
    pub ids: Vec<u32>,
    pub text: String,
    pub block: Block,
    /// Why the run ended.
    pub stop: crate::Stop,
    /// Where the content starts in `ids` (past the block's end marker; the
    /// end of the ids for an open block).
    pub content: usize,
    /// Past the last end marker in `ids`: `content`, unless the reply wrote
    /// the marker again after its block closed.
    pub tail: usize,
    pub reasoning_text: String,
    pub content_text: String,
    /// The delivered text's bytes past `ids`' text (a stop or a
    /// cancellation inside a token): text, never ids.
    pub partial: usize,
    /// Every id the run generated; `ids` are the accepted ones.
    pub generated: usize,
    /// `session.keep_logprobs = sampled | full`: the sampled token's
    /// log-probability per id.
    pub logprobs: Option<Vec<f32>>,
    /// `session.keep_logprobs = full`: each position's log-probabilities.
    pub rows: Option<Vec<Vec<f32>>>,
}

/// A reply's accepted conversational frontier (`sequence-reuse-design.md`
/// §2.3f): of the ids the run generated, the longest prefix of whole units
/// whose text the delivered text begins with, and the delivered bytes past
/// it — where a stop string or a cancellation cut inside a token. The
/// delivered text is what the server accepted from the run and handed on,
/// not what a client acknowledged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Frontier {
    /// Every id the run generated (the session's log keeps them all).
    pub generated: usize,
    /// The whole ids the delivered text begins with.
    pub accepted: usize,
    /// The delivered text's bytes past those ids' text.
    pub partial: usize,
    /// The delivered text's bytes.
    pub delivered: usize,
    /// Whether the reply log keeps the reply: the delivered text is those
    /// ids' text and a beginning of the next unit's.
    pub logged: bool,
}

impl Frontier {
    pub fn json(&self) -> Value {
        json!({
            "generated": self.generated,
            "accepted": self.accepted,
            "partial_bytes": self.partial,
            "delivered_bytes": self.delivered,
            "logged": self.logged,
        })
    }
}

/// How a message's text matched a logged reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Matched {
    /// The content is the whole reply as we returned it (surrounding
    /// whitespace aside).
    Full,
    /// The content is the reply's content: the reasoning was sent apart or
    /// dropped.
    Content,
}

/// The conversation a reply answers: the messages before it as the client
/// sent them, the tools and the template it was rendered under, and the
/// model. A reply of ours in a later history takes the logged ids and
/// reasoning only where the history before it hashes to the same scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scope([u8; 32]);

/// What a reply is logged against: its scope, and the thinking of the
/// prompt it followed (where its reasoning block starts).
#[derive(Debug, Clone, Copy)]
pub struct Origin {
    pub scope: Scope,
    pub thinking: Option<Thinking>,
}

/// What a history message says about its reasoning, across the fields a
/// template reads it from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Given {
    /// No reasoning field: the client sent none.
    Absent,
    /// A field set to `null`: the client declares none.
    Null,
    /// A field set to `""`: the reply's reasoning, as the client has it, is
    /// empty.
    Empty,
    /// Reasoning text (or any other value): the client's own.
    Text,
}

fn given_reasoning(m: &Value) -> Given {
    let mut given = Given::Absent;
    for f in REASONING_FIELDS {
        given = match (m.get(f), given) {
            (None, g) => g,
            (Some(Value::Null), Given::Absent) => Given::Null,
            (Some(Value::Null), g) => g,
            (Some(Value::String(s)), _) if s.is_empty() => Given::Empty,
            (Some(_), _) => return Given::Text,
        };
    }
    given
}

/// Whether nothing of a reply comes before its reasoning block but the start
/// marker it sampled: a template writes the reasoning first, so only then can
/// the reply go to it as reasoning and content apart without losing text.
fn block_leads(block: &Block) -> bool {
    block.interior().is_some_and(|r| r.start <= 1)
}

/// The reasoning a message gives apart from its content: the text every
/// reasoning field set to one carries, `None` where none does or two differ.
fn reasoning_apart(m: &Value) -> Option<&str> {
    let mut apart: Option<&str> = None;
    for f in REASONING_FIELDS {
        match m.get(f).and_then(Value::as_str) {
            Some(s) if !s.trim().is_empty() => match apart {
                Some(a) if a != s => return None,
                _ => apart = Some(s),
            },
            _ => {}
        }
    }
    apart
}

/// The texts a reply is found by: the whole reply, and its content where it
/// has a reasoning block apart from it, each without surrounding whitespace.
fn keys(reply: &Reply) -> impl Iterator<Item = &str> {
    let whole = reply.text.trim();
    let content = reply.content_text.trim();
    std::iter::once(whole)
        .chain((content != whole).then_some(content))
        .filter(|k| !k.is_empty())
}

/// The replies one key of one scope names. More than one distinct reply (a
/// regeneration that sampled other ids or other reasoning) leaves the text
/// ambiguous for as long as any of them is logged.
#[derive(Default)]
struct Candidates {
    replies: Vec<Arc<Reply>>,
    ambiguous: bool,
}

/// What a history message's text found in the log.
enum Found {
    /// One reply of this conversation.
    One(Arc<Reply>, Matched),
    /// Several distinct replies of this conversation carry the text.
    Ambiguous,
    /// Only replies of other conversations carry it.
    Elsewhere,
    None,
}

/// A logged reply and the scope it answers.
struct Logged {
    scope: Scope,
    reply: Arc<Reply>,
}

/// The replies of ours a front end has returned, found by the conversation
/// they answer and the text the client may send back. Bounded; the oldest
/// go first.
#[derive(Default)]
struct ReplyLog {
    entries: VecDeque<Logged>,
    by_scope: HashMap<Scope, HashMap<String, Candidates>>,
    /// Logged replies per text, in any scope: a text known only elsewhere is
    /// counted as a provenance miss, never used.
    texts: HashMap<String, usize>,
    tokens: usize,
}

impl ReplyLog {
    fn insert(&mut self, scope: Scope, reply: Reply) {
        let reply = Arc::new(reply);
        self.tokens += reply.ids.len();
        let texts = self.by_scope.entry(scope).or_default();
        for key in keys(&reply) {
            *self.texts.entry(key.to_string()).or_default() += 1;
            let c = texts.entry(key.to_string()).or_default();
            match c
                .replies
                .iter()
                .position(|r| r.ids == reply.ids && r.block == reply.block)
            {
                // The same reply again (a regeneration that sampled the same
                // ids): one provenance.
                Some(i) => c.replies[i] = Arc::clone(&reply),
                None => {
                    c.replies.push(Arc::clone(&reply));
                    c.ambiguous |= c.replies.len() > 1;
                }
            }
        }
        self.entries.push_back(Logged { scope, reply });
        while self.entries.len() > LOG_REPLIES
            || (self.tokens > LOG_TOKENS && self.entries.len() > 1)
        {
            let Some(old) = self.entries.pop_front() else {
                break;
            };
            self.forget(old);
        }
    }

    fn forget(&mut self, old: Logged) {
        self.tokens -= old.reply.ids.len();
        for key in keys(&old.reply) {
            if let Some(n) = self.texts.get_mut(key) {
                *n -= 1;
                if *n == 0 {
                    self.texts.remove(key);
                }
            }
            let Some(texts) = self.by_scope.get_mut(&old.scope) else {
                continue;
            };
            if let Some(c) = texts.get_mut(key) {
                c.replies.retain(|r| !Arc::ptr_eq(r, &old.reply));
                if c.replies.is_empty() {
                    texts.remove(key);
                }
            }
            if texts.is_empty() {
                self.by_scope.remove(&old.scope);
            }
        }
    }

    fn find(&self, scope: Scope, content: &str) -> Found {
        let key = content.trim();
        if key.is_empty() {
            return Found::None;
        }
        match self.by_scope.get(&scope).and_then(|t| t.get(key)) {
            Some(c) if c.ambiguous => Found::Ambiguous,
            Some(c) => {
                let r = &c.replies[0];
                let how = if r.text.trim() == key {
                    Matched::Full
                } else {
                    Matched::Content
                };
                Found::One(Arc::clone(r), how)
            }
            None if self.texts.contains_key(key) => Found::Elsewhere,
            None => Found::None,
        }
    }

    /// The newest reply whose whole text or content is `content`, in any
    /// scope: a test's or a report's lookup, never a prompt's.
    fn any(&self, content: &str) -> Option<Arc<Reply>> {
        self.entries
            .iter()
            .rev()
            .find(|l| l.reply.text == content || l.reply.content_text.trim() == content.trim())
            .map(|l| Arc::clone(&l.reply))
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
    /// The messages as the client sent them, when `messages` is a rewrite
    /// of them (an image's pixels replaced by its placeholder): what a
    /// reply's scope is hashed over. `None` is `messages`.
    pub history: Option<&'a Value>,
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
    /// Assistant messages whose text several distinct replies of this
    /// conversation carry: tokenized as sent.
    pub ambiguous: usize,
    /// Assistant messages whose text only another conversation's reply
    /// carries: tokenized as sent, nothing restored.
    pub unverified: usize,
    /// Replies of ours whose reasoning the log would have restored, kept as
    /// sent because the message says null or empty or the request strips.
    pub declined: usize,
    /// Matched replies whose run ended inside their reasoning block.
    pub open: usize,
    /// Of those, the ones given to the template as reasoning, apart from
    /// their content (by `Incomplete::Reasoning`, or by the client's own
    /// reasoning field): whatever closes the block in the prompt is the
    /// template's text, not a sampled id.
    pub open_apart: usize,
    /// Matched replies rendered from the ids the model emitted — the whole
    /// turn, in place of the template's rendering of it (decision 51 Q1).
    pub emitted: usize,
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
    /// The prompt's ids taken from logged replies, as sampled; every other
    /// id is the template's or the client's text, encoded.
    pub sampled: Vec<Range<usize>>,
    /// `session.keep_logprobs`: the logged log-probability of each id the
    /// prompt took from a reply, `None` elsewhere.
    pub logprobs: Option<Vec<Option<f32>>>,
    /// The template renders this prompt took (1 by construction).
    pub renders: u64,
    /// What the reply to this prompt is logged against.
    pub origin: Origin,
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
    /// Normal request kwargs, resolved once by the frontend thinking policy.
    thinking_kwargs: [Option<bool>; 3],
    log: Mutex<ReplyLog>,
    tokenizations: AtomicU64,
    /// The model and the template every scope is hashed under.
    context: [u8; 32],
}

impl Front {
    /// Read `engine`'s constants (once: this is the load-time read).
    pub fn new(engine: &dyn Served, options: SessionOptions) -> Self {
        let mut h = Sha256::new();
        h.update(b"lloom front scope 1\0");
        h.update(engine.model_name().as_bytes());
        h.update(b"\0");
        if let Some(text) = engine.text() {
            if let Some(t) = &text.template {
                h.update(t.digest());
            }
            h.update(text.bos_token.as_bytes());
        }
        Self {
            constants: engine.text().map(Constants::read),
            options,
            thinking_kwargs: MODES.map(|want| {
                let mut kwargs = Map::new();
                thinking::set_kwarg(engine, want, &mut kwargs);
                kwargs.get(ENABLE_THINKING).and_then(Value::as_bool)
            }),
            log: Mutex::new(ReplyLog::default()),
            tokenizations: AtomicU64::new(0),
            context: h.finalize().into(),
        }
    }

    /// The scope before each of `messages` and after the last (`n + 1`
    /// entries): a chain of digests over the context, the request's tools
    /// and each message as sent.
    fn scopes(&self, messages: &[Value], kwargs: &Map<String, Value>) -> Vec<Scope> {
        let mut h = Sha256::new();
        h.update(self.context);
        h.update(
            kwargs
                .get(TOOLS)
                .unwrap_or(&Value::Null)
                .to_string()
                .as_bytes(),
        );
        let mut at = Scope(h.finalize().into());
        let mut out = Vec::with_capacity(messages.len() + 1);
        out.push(at);
        for m in messages {
            let mut h = Sha256::new();
            h.update(at.0);
            h.update(m.to_string().as_bytes());
            at = Scope(h.finalize().into());
            out.push(at);
        }
        out
    }

    /// What a reply to `messages` (as the next request will carry them,
    /// rendered under `kwargs`) is logged against: for a caller that keeps a
    /// history other than the request's (a prompt budget that dropped turns
    /// from the conversation itself).
    pub fn origin(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        thinking: Option<Thinking>,
    ) -> Origin {
        let list = messages.as_array().map_or(&[][..], Vec::as_slice);
        let scope = *self.scopes(list, kwargs).last().expect("n + 1 scopes");
        Origin { scope, thinking }
    }

    pub fn constants(&self) -> Option<&Constants> {
        self.constants.as_ref()
    }

    pub fn options(&self) -> SessionOptions {
        self.options
    }

    /// The `preserve_thinking` every request that does not set it renders
    /// with: `session.persist_thinking` where set, else `true` — the history
    /// keeps what the model generated, stripping is opt-in. `None` where the
    /// template has no switch that changes a past reply's reasoning: its own
    /// rendering stands.
    fn persist(&self, c: &Constants) -> Option<bool> {
        c.persist_switch
            .then_some(self.options.persist.unwrap_or(true))
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
        // The tail each mode's checkpoint takes under the persistence a
        // request that sets none renders with.
        let persist = self.persist(c);
        let gen = |w| match c.mode(self.thinking_kwargs[slot(w)], w) {
            Some(Ok(m)) => format!(
                "{}/{}",
                m.gen.len(),
                m.keep[slot(persist)]
                    .or(m.keep[0])
                    .map_or("-".into(), |k| (m.gen.len() - k).to_string())
            ),
            Some(Err(_)) => "refused".into(),
            None => "-".into(),
        };
        format!(
            "reasoning_default={persists} switch={} persist_thinking={} missing_reasoning={} keep_logprobs={} \
             incomplete_reasoning={} open={:?} groups_tools={} gen/tail[unset,on,off]=[{}, {}, {}]",
            c.persist_switch,
            match (self.options.persist, self.persist(c)) {
                (Some(p), _) => p.to_string(),
                (None, Some(p)) => format!("{p}(default)"),
                (None, None) => "template".into(),
            },
            if self.options.restore_reasoning {
                "log"
            } else {
                "as-sent"
            },
            self.options.keep.name(),
            self.options.incomplete.name(),
            c.open,
            c.groups_tools,
            gen(None),
            gen(Some(true)),
            gen(Some(false)),
        )
    }

    /// The session policy this front end applies, resolved: `/health`'s
    /// `session`.
    pub fn session_json(&self) -> Value {
        let persist = self.constants.as_ref().and_then(|c| self.persist(c));
        json!({
            "persist_thinking": persist,
            "persist_thinking_set": self.options.persist,
            "incomplete_reasoning": self.options.incomplete.name(),
            "missing_reasoning": if self.options.restore_reasoning { "log" } else { "as-sent" },
            "keep_logprobs": self.options.keep.name(),
        })
    }

    /// Whether the history's last message is a reply of ours whose run ended
    /// inside its reasoning block (decision 51 Q3): then an opt-in strip is
    /// suspended for that one request, so the model sees the thought it has
    /// not finished. Looked up as the prompt looks a reply up — in the
    /// message's own scope, by its text or the reasoning it gives apart — so a
    /// reply the log does not hold, or another conversation's, does not
    /// suspend anything.
    fn last_reply_is_open(&self, c: &Constants, list: &[Value], scopes: &[Scope]) -> bool {
        if c.tags.is_none() {
            return false;
        }
        let Some((i, m)) = list.iter().enumerate().next_back() else {
            return false;
        };
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            return false;
        }
        let Some(&scope) = scopes.get(i) else {
            return false;
        };
        let content = m.get("content").and_then(Value::as_str).unwrap_or("");
        let key = if content.trim().is_empty() {
            match reasoning_apart(m) {
                Some(r) => r,
                None => return false,
            }
        } else {
            content
        };
        matches!(
            self.log.lock().find(scope, key),
            Found::One(reply, _) if matches!(reply.block, Block::Open { .. })
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
            Effort::NotRead
                if family == "lfm2-gguf" && matches!(value.as_str(), Some("none" | "default")) =>
            {
                eprintln!("[{family}: {REASONING_EFFORT}={value} ignored: the chat template does not read it]");
                return Ok(());
            }
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
        let empty = Vec::new();
        let list = req.messages.as_array().unwrap_or(&empty);
        let sent = req
            .history
            .and_then(Value::as_array)
            .filter(|h| h.len() == list.len())
            .unwrap_or(list);
        let scopes = self.scopes(sent, &req.kwargs);
        let mut kwargs = req.kwargs;
        if let (Some(p), false) = (self.persist(c), kwargs.contains_key(PRESERVE_THINKING)) {
            // Decision 51 Q3: a request whose history ends in a thought the
            // model did not finish renders with the reasoning kept, even under
            // an opt-in strip — a stripped prompt would show the model an
            // empty reply where its own unfinished thinking should be, and a
            // "continue" turn would have lost it. No template has a per-message
            // switch, so this keeps every past reply's reasoning, for this
            // request alone.
            let p = p || self.last_reply_is_open(c, list, &scopes);
            kwargs.insert(PRESERVE_THINKING.into(), Value::Bool(p));
        }

        // The replies of ours, found by the conversation before them and
        // their text; a matched reply with a reasoning block is given to the
        // template as the model wrote it, reasoning and content apart.
        let mut spliced = Spliced::default();
        // Each matched reply, and whether its open block went to the template
        // apart (its ids are then aligned as reasoning).
        let mut matched: Vec<(usize, Arc<Reply>, bool)> = Vec::new();
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
        // An explicit strip (the request's or the option's) restores
        // nothing, whatever the template keeps.
        let strips = kwargs.get(PRESERVE_THINKING).and_then(Value::as_bool) == Some(false);
        {
            let log = self.log.lock();
            for (i, m) in list.iter().enumerate() {
                if m.get("role").and_then(Value::as_str) != Some("assistant") {
                    continue;
                }
                spliced.replies += 1;
                let content = m.get("content").and_then(Value::as_str).unwrap_or("");
                if content.trim().is_empty() {
                    // No content to find it by: a reply cut inside its
                    // reasoning given back with that reasoning apart, as
                    // the template reads it, is found by the reasoning.
                    // It goes as sent; only its ids are the logged ones.
                    let Some(reasoning) = reasoning_apart(m) else {
                        continue;
                    };
                    match log.find(scopes[i], reasoning) {
                        Found::One(reply, Matched::Full)
                            if matches!(reply.block, Block::Open { .. }) =>
                        {
                            spliced.matched += 1;
                            spliced.reply_tokens += reply.ids.len();
                            spliced.open += 1;
                            spliced.open_apart += 1;
                            matched.push((i, reply, true));
                        }
                        Found::Ambiguous => spliced.ambiguous += 1,
                        Found::Elsewhere => spliced.unverified += 1,
                        Found::One(..) | Found::None => {}
                    }
                    continue;
                }
                let (reply, how) = match log.find(scopes[i], content) {
                    Found::One(reply, how) => (reply, how),
                    Found::Ambiguous => {
                        spliced.ambiguous += 1;
                        continue;
                    }
                    Found::Elsewhere => {
                        spliced.unverified += 1;
                        continue;
                    }
                    Found::None => continue,
                };
                spliced.matched += 1;
                spliced.reply_tokens += reply.ids.len();
                let given = given_reasoning(m);
                // Kept, the reply is rendered as the model wrote it,
                // reasoning and content apart. Stripped, the message goes to
                // the template as sent: what survives of the whole text is
                // the template's own split (`content.split('</think>')[-1]`
                // keeps what follows the *last* marker, and a reply can write
                // one again after the engine closed its block), and the
                // sampled ids are spliced wherever its text is theirs.
                // Full, the reasoning is in the text the client sent and is
                // only given apart; by content, it is restored from the log,
                // and only for a message that left it absent.
                //
                // A reply whose run ended inside its block is text unless
                // `Incomplete::Reasoning`: then, sent back whole, it is given
                // apart whatever the template keeps, since no marker in its
                // text lets the template split it itself.
                let kept = persists || last_user.is_some_and(|u| i > u);
                let open = matches!(reply.block, Block::Open { .. });
                let leads = block_leads(&reply.block);
                let open_apart = open && leads && self.options.incomplete == Incomplete::Reasoning;
                let wanted = (reply.block.closed().is_some() && leads || open_apart)
                    && (kept || open_apart && how == Matched::Full)
                    && (how == Matched::Full || self.options.restore_reasoning);
                let restore = wanted && given == Given::Absent && (how == Matched::Full || !strips);
                if wanted && !restore && how == Matched::Content {
                    spliced.declined += 1;
                }
                spliced.open += usize::from(open);
                if restore {
                    if how == Matched::Content {
                        spliced.restored += 1;
                    }
                    spliced.open_apart += usize::from(open_apart);
                    let msgs = rewritten.get_or_insert_with(|| list.clone());
                    let mut msg = m.clone();
                    msg["content"] = json!(reply.content_text);
                    for f in REASONING_FIELDS {
                        msg[f] = json!(reply.reasoning_text);
                    }
                    msgs[i] = msg;
                }
                matched.push((i, reply, restore && open_apart));
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

        // The logged replies' ids in their own spans: the whole turn the model
        // emitted (decision 51 Q1), or — where that span cannot be identified
        // — aligned segment by segment inside the template's rendering of it.
        let mut runs: Vec<Run> = Vec::new();
        if let Some(tb) = &text_blocks {
            for (i, reply, apart) in &matched {
                let Some(region) = tb.region(*i, rendered.len()) else {
                    continue;
                };
                if emit_turn(text, &rendered, region.clone(), reply, c.eot(), &mut runs) {
                    spliced.emitted += 1;
                    continue;
                }
                align_reply(text, &rendered, region, reply, *apart, &mut runs);
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
        let mut sampled: Vec<Range<usize>> = Vec::with_capacity(runs.len());
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
            sampled.push(ids.len()..ids.len() + run.ids.len());
            ids.extend_from_slice(&run.reply.ids[run.ids.clone()]);
            spliced.spliced_tokens += run.ids.len();
            if keep_lp {
                // A run can carry one id the model did not sample — the
                // end-of-turn token, which is the template's bytes for a
                // position the session fed (`emit_turn`) — and that id has no
                // logprob of its own. Take what the reply kept and leave the
                // rest `None`, as every id the prompt encoded carries.
                match &run.reply.logprobs {
                    Some(lp) => {
                        let have = run.ids.start.min(lp.len())..run.ids.end.min(lp.len());
                        logprobs.extend(lp[have].iter().map(|&v| Some(v)));
                        logprobs.resize(ids.len(), None);
                    }
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
        for r in &mut sampled {
            *r = r.start + head_shift..r.end + head_shift;
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
            Some(m) => {
                let keep = m.keep[slot(persist)].or(m.keep[0]);
                // Before whichever divergence comes first: a reply that
                // closes its block, or one cut inside it and given back as
                // text.
                let keep = match self.options.incomplete {
                    Incomplete::Recover if thinking.on => {
                        let open = m.keep_open[slot(persist)].or(m.keep_open[0]);
                        match (keep, open) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (a, b) => a.or(b),
                        }
                    }
                    _ => keep,
                };
                keep.map_or(m.gen.len(), |k| m.gen.len() - k)
            }
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
            sampled,
            logprobs: keep_lp.then_some(logprobs),
            renders,
            origin: Origin {
                scope: *scopes.last().expect("n + 1 scopes"),
                thinking: Some(thinking),
            },
        })
    }

    /// A family without a text side: render (with the effort's check),
    /// encode and switch through the seam, as before.
    fn plain(&self, engine: &dyn Served, req: Request<'_>) -> Result<Prompt> {
        let scope = self
            .origin(req.history.unwrap_or(req.messages), &req.kwargs, None)
            .scope;
        let before = engine.template_renders();
        let rendered = engine.render_checked(req.messages, &req.kwargs, &[REASONING_EFFORT])?;
        let renders = engine.template_renders() - before;
        for ignored in &rendered.ignored {
            if engine.capabilities().family == "lfm2-gguf"
                && ignored.reason == NOT_READ
                && matches!(ignored.value.as_str(), Some("none" | "default"))
            {
                eprintln!(
                    "[{}: {REASONING_EFFORT}={} ignored: the chat template does not read it]",
                    engine.capabilities().family,
                    ignored.value
                );
                continue;
            }
            let i = ignored;
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
            sampled: Vec::new(),
            logprobs: None,
            renders,
            origin: Origin {
                scope,
                thinking: Some(th),
            },
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

    /// Keep a reply in the session log against `origin`: the ids `decoded`
    /// sampled, with what the run kept of their logprobs. Returns whether it
    /// was kept ([`Self::record_frontier`]).
    pub fn record(&self, engine: &dyn Served, origin: Origin, decoded: &Decoded) -> bool {
        self.record_frontier(engine, origin, decoded).logged
    }

    /// [`Self::record`], returning the reply's accepted frontier. The ids
    /// kept are the longest prefix of whole units whose text the delivered
    /// text begins with, and the rest of the delivered text — a beginning of
    /// the next unit's — is kept as text. A delivered text that is not that
    /// (nothing of it whole, or not the run's) is not kept.
    pub fn record_frontier(
        &self,
        engine: &dyn Served,
        origin: Origin,
        decoded: &Decoded,
    ) -> Frontier {
        let thinking = origin.thinking;
        let mut frontier = Frontier {
            generated: decoded.tokens.len(),
            delivered: decoded.text.len(),
            ..Frontier::default()
        };
        let Some(text) = engine.text() else {
            return frontier;
        };
        if decoded.tokens.is_empty() {
            return frontier;
        }
        let Some((n, partial)) = accepted(text, &decoded.tokens, &decoded.text) else {
            return frontier;
        };
        frontier.accepted = n;
        frontier.partial = partial;
        if n == 0 {
            return frontier;
        }
        let partial_text = &decoded.text[decoded.text.len() - partial..];
        let full = decoded.text.clone();
        let tags = self.constants.as_ref().and_then(|c| c.tags.as_ref());
        let ids = &decoded.tokens[..n];
        let open = thinking.is_some_and(|t| t.on);
        // The block as the model produced it: opened by the prompt or by a
        // start marker it sampled, closed by the first end marker after
        // that; a run that ended before one leaves it open. No end marker is
        // assumed that the model did not sample.
        let block = match tags {
            Some(t) => {
                let start = if open {
                    Some(0)
                } else {
                    ids.iter().position(|&id| id == t.start).map(|s| s + 1)
                };
                match start {
                    None => Block::None,
                    Some(s) => match ids[s..].iter().position(|&id| id == t.end) {
                        Some(e) => Block::Closed {
                            interior: s..s + e,
                            end: s + e,
                        },
                        None => Block::Open {
                            interior: s..ids.len(),
                        },
                    },
                }
            }
            None => Block::None,
        };
        let mut reasoning_text = match block.interior() {
            Some(r) => decode(text, &ids[r.clone()]).unwrap_or_default(),
            None => String::new(),
        };
        let content = match &block {
            Block::None => 0,
            Block::Open { .. } => ids.len(),
            Block::Closed { end, .. } => end + 1,
        };
        // Past the last end marker: where a stripping template's split of
        // the whole text starts the content it keeps.
        let tail = match (tags, block.closed()) {
            (Some(t), Some(_)) => ids
                .iter()
                .rposition(|&id| id == t.end)
                .map_or(content, |e| e + 1)
                .max(content),
            _ => content,
        };
        let mut content_text = decode(text, &ids[content..]).unwrap_or_default();
        // The delivered text past the whole ids is the block's while it is
        // open, else the content's.
        match block {
            Block::Open { .. } => reasoning_text.push_str(partial_text),
            _ => content_text.push_str(partial_text),
        }
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
        self.log.lock().insert(
            origin.scope,
            Reply {
                ids: ids.to_vec(),
                text: full,
                block,
                stop: decoded.stop,
                content,
                tail,
                reasoning_text,
                content_text,
                partial,
                generated: decoded.tokens.len(),
                logprobs,
                rows,
            },
        );
        frontier.logged = true;
        frontier
    }

    /// The newest reply the log holds whose text or content is `content`,
    /// in any conversation: for a test or a report, never a prompt.
    pub fn reply(&self, content: &str) -> Option<Arc<Reply>> {
        self.log.lock().any(content)
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
        logprobs: Some(0),
    };
    lloom::nn::refuse(body, &asks).err().map(|e| {
        format!(
            "this process decodes through {name}: {}",
            e.without_backtrace()
        )
    })
}

/// Of `tokens`, the longest prefix of whole units whose text `delivered`
/// begins with, and `delivered`'s bytes past it — which must begin the next
/// unit's text: the accepted frontier ([`Frontier`]). `None` where
/// `delivered` is not that (not the run's text).
pub fn accepted(text: &Text, tokens: &[u32], delivered: &str) -> Option<(usize, usize)> {
    let full = decode(text, tokens)?;
    if full == delivered {
        return Some((tokens.len(), 0));
    }
    let u = units(text, tokens, 0)?;
    let mut at = 0;
    let mut kept = 0;
    let mut next = None;
    for (r, s) in &u {
        if !delivered[at..].starts_with(s.as_str()) {
            next = Some(s.as_str());
            break;
        }
        at += s.len();
        kept = r.end;
        if at == delivered.len() {
            break;
        }
    }
    let rest = &delivered[at..];
    if !rest.is_empty() && !next.is_some_and(|s| s.starts_with(rest)) {
        return None;
    }
    Some((kept, rest.len()))
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

/// Render a past assistant turn from the ids the model emitted (decision 51
/// Q1): the whole turn — anything before its reasoning block, the block's
/// markers and interior with its own whitespace, the content — in place of
/// whatever the template rendered for it. The template's own role header and
/// everything else it writes stay: the model did not write them.
///
/// The run ends at the turn's end: **through the end-of-turn token** for a
/// reply the model ended itself (that position was fed, and the token is the
/// template's own bytes for it, since the log does not keep the sampled EOS),
/// and at the reply's last id for one a limit, a stop string or a cancellation
/// cut, which fed no end-of-turn position.
///
/// So the next prompt's ids are the session's log wherever a reply is matched,
/// whatever the template would have rewritten — Qwen's `reasoning_content|trim`,
/// a re-split of an evicted reply, an engine-closed empty block. No restore, no
/// replay, no tail rewind.
///
/// `eot` is the end-of-turn token the template writes after an assistant turn
/// and the id it encodes to, read at load ([`Constants::eot`]).
///
/// `false` — the caller's segment alignment takes the span instead, which is
/// the pre-Q1 path and still splices what it can — wherever this cannot speak
/// for the span exactly:
/// - the reply's text is not in the template's rendering of its message
///   verbatim (the template trimmed, stripped or re-split it; the client
///   edited it);
/// - it is there more than once, so which occurrence the ids belong to would
///   be a guess;
/// - the ids do not decode to exactly the text they would replace.
fn emit_turn(
    text: &Text,
    rendered: &str,
    region: Range<usize>,
    reply: &Arc<Reply>,
    eot: Option<(u32, &str)>,
    runs: &mut Vec<Run>,
) -> bool {
    let span = &rendered[region.clone()];
    // The turn's text as the model emitted it, and the template's rendering
    // of this message. The emitted text must be there verbatim: anything else
    // — a trim, a strip, a re-split, a client's edit — is a span this cannot
    // speak for, and the caller's segment alignment takes it.
    let emitted = reply.text.as_str();
    if emitted.is_empty() {
        return false;
    }
    // Where the turn's own bytes begin in the template's rendering of it, and
    // where they end. The template may have rewritten the stretch between —
    // Qwen trims the reasoning it is given, deleting whitespace the model
    // emitted, which no field can carry back — so the match is on the bytes
    // the rewriting cannot touch: the turn's first non-whitespace run, and the
    // end-of-turn token the template wrote. Everything between is replaced by
    // what the model emitted, which is Q1's instruction.
    let head = emitted.trim_start();
    let lead: String = head.chars().take_while(|c| !c.is_whitespace()).collect();
    if lead.is_empty() {
        return false;
    }
    let Some(off) = span.find(lead.as_str()) else {
        return false;
    };
    // Exactly one place it can begin, or which one the ids belong to is a
    // guess.
    if span[off + lead.len()..].contains(lead.as_str()) {
        return false;
    }
    let start = region.start + off;
    // The turn ends where the template ended it: at the end-of-turn token it
    // wrote after the body, or — with no such constant — at the span's end.
    let body_end = match eot {
        Some((_, marker)) => match span[off..].rfind(marker) {
            Some(i) => region.start + off + i,
            None => return false,
        },
        None => region.end,
    };
    if body_end <= start {
        return false;
    }
    // Through the end-of-turn token where the model ended the turn itself:
    // that position was fed, and the token is the template's own bytes for it
    // (the log does not keep the sampled EOS). Only when the template writes
    // it immediately after the reply, with nothing of its own between.
    // Bounded by the span: a run never reaches into the next message's text,
    // nor into the generation prompt.
    let through = match eot {
        Some((id, marker))
            if reply.stop == crate::Stop::Eos && body_end + marker.len() <= region.end =>
        {
            Some((id, marker.len()))
        }
        _ => None,
    };
    let (text_end, ids) = match through {
        Some((id, len)) => {
            let mut ids = reply.ids.clone();
            ids.push(id);
            (body_end + len, ids)
        }
        None => (body_end, reply.ids.clone()),
    };
    if ids.is_empty() {
        return false;
    }
    // Two guards, and both are needed.
    //
    // The ids must be this reply's turn: what the log holds, with the
    // end-of-turn token where the model ended the turn itself. Trailing
    // whitespace aside, because a template normalises the whitespace after
    // the body.
    let want = match through {
        Some(_) => format!("{}{}", emitted.trim_end(), eot.map_or("", |(_, m)| m)),
        None => emitted.trim_end().to_string(),
    };
    match decode(text, &ids) {
        Some(d) if d.trim_end() == want.trim_end() => {}
        _ => return false,
    }
    // And nothing the template wrote of its own may be inside the run. The
    // stretch being replaced is allowed to differ from the model's bytes only
    // in whitespace the template normalised: a closing it wrote for a block
    // the model left open (`\n</think>\n\n`), a wrapper, a re-split, are all
    // text of the template's, and a run that swallowed them would put the
    // model's ids where the template's text stands.
    let replaced = &rendered[start..text_end];
    let same = |a: &str, b: &str| -> bool {
        let f = |s: &str| -> Vec<String> { s.split_whitespace().map(str::to_string).collect() };
        f(a) == f(b)
    };
    if !same(replaced, &want) {
        return false;
    }
    let n = ids.len();
    runs.push(Run {
        text: start..text_end,
        ids: 0..n,
        reply: Arc::new(Reply {
            ids,
            ..Reply::clone(reply)
        }),
    });
    true
}

/// Align `reply`'s segments — anything before its reasoning block, the
/// block's interior, its end marker, the content — inside `region` of
/// `rendered`, pushing a run per stretch whose text lies there contiguously.
/// An open block is a segment of its own only where it went to the template
/// as reasoning (`apart`); as text, the reply is one segment.
fn align_reply(
    text: &Text,
    rendered: &str,
    region: Range<usize>,
    reply: &Arc<Reply>,
    apart: bool,
    runs: &mut Vec<Run>,
) {
    let n = reply.ids.len();
    let mut segments: Vec<Range<usize>> = Vec::new();
    match &reply.block {
        Block::Open { interior: r } if apart => {
            if r.start > 0 {
                segments.push(0..r.start);
            }
            segments.push(r.clone());
        }
        Block::Closed { interior: r, .. } => {
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
        Block::None | Block::Open { .. } => segments.push(0..n),
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
