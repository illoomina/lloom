//! The tiny models' chat templates (`lloom::testing::tiny`,
//! `docs/porting-models.md` §21.8): lloom's own minimal templates, one per
//! family, rendered through the engine's template machinery
//! (`app::template::Template`, what `lloom chat` and `lloom serve` render
//! with).
//!
//! - **One test per path.** Each renders one conversation through the tiny
//!   template and checks the special tokens it encodes to, in order, through
//!   the family's tiny tokenizer, against the list written here. A template
//!   edit that drops a path fails that path's test.
//! - **The path-coverage check** (`real_templates_*`). When a checkpoint's own
//!   template is in the Hugging Face cache, each path renders through it too
//!   and must give the same tokens, or differ where a gap is listed with its
//!   reason; every variable it reads and every branch condition it has is a
//!   path here or a listed gap. Conditions are keyed by a hash of their text,
//!   so no checkpoint's text lives in this tree. Skipped, saying so, when the
//!   cache has no template of the family.
#![cfg(feature = "serve")]

use lloom::testing::tiny::TinyTokenizer;
use lloom_engine::app::template::{Rejected, Template};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Family {
    Lfm2,
    Qwen4Exp,
}

impl Family {
    fn tokenizer(self) -> TinyTokenizer {
        match self {
            Family::Lfm2 => TinyTokenizer::lfm2(),
            Family::Qwen4Exp => TinyTokenizer::qwen4exp(),
        }
    }

    /// The `bos_token` the engine renders with (`app/lfm2.rs` takes the
    /// tokenizer's `<|startoftext|>`; `app/qwen4exp.rs` renders none).
    fn bos(self) -> &'static str {
        match self {
            Family::Lfm2 => "<|startoftext|>",
            Family::Qwen4Exp => "",
        }
    }
}

/// What a rendering came to: the special tokens of its encoding, in order,
/// each `<|im_start|>` with the role after it; or the template's own refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Tokens(Vec<String>),
    Rejected,
    /// A render failure that is not the template's `raise_exception` (on
    /// `lloom serve` a 400 like a rejection when the template renders a
    /// one-message conversation, `template::Unrenderable`).
    Failed(String),
}

struct Path {
    family: Family,
    name: &'static str,
    messages: Value,
    kwargs: Value,
    want: Option<&'static [&'static str]>,
    /// Words the tiny template's rendering must carry, and must not.
    has: &'static [&'static str],
    lacks: &'static [&'static str],
    /// Why a checkpoint's template renders this path differently, when it
    /// does. The coverage check asserts that it still does.
    real_differs: Option<&'static str>,
}

const S: &str = "<|startoftext|>";
const SYS: &str = "<|im_start|>system";
const USER: &str = "<|im_start|>user";
const ASST: &str = "<|im_start|>assistant";
const TOOL: &str = "<|im_start|>tool";
const END: &str = "<|im_end|>";
const THINK: &str = "<think>";
const UNTHINK: &str = "</think>";
const CALL: &str = "<tool_call>";
const CALLS: &str = "<|tool_call_start|>";
const UNCALLS: &str = "<|tool_call_end|>";
const RESP: &str = "<tool_response>";
const UNRESP: &str = "</tool_response>";
const IMAGE: &str = "<image>";
const UNCALL: &str = "</tool_call>";

fn user(text: &str) -> Value {
    json!({ "role": "user", "content": text })
}

fn system(text: &str) -> Value {
    json!({ "role": "system", "content": text })
}

fn assistant(text: &str) -> Value {
    json!({ "role": "assistant", "content": text })
}

fn reasoned(reasoning: &str, text: &str) -> Value {
    json!({ "role": "assistant", "reasoning_content": reasoning, "content": text })
}

fn tool_result(text: &str) -> Value {
    json!({ "role": "tool", "content": text })
}

fn weather_call() -> Value {
    json!({ "type": "function", "function": {
        "name": "get_weather", "arguments": { "city": "Paris", "days": 2 } } })
}

fn tools() -> Value {
    json!([
        { "type": "function", "function": { "name": "get_weather",
          "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } },
        { "type": "function", "function": { "name": "get_time", "parameters": {} } },
    ])
}

fn path(family: Family, name: &'static str, messages: Value) -> Path {
    Path {
        family,
        name,
        messages,
        kwargs: json!({}),
        want: None,
        has: &[],
        lacks: &[],
        real_differs: None,
    }
}

impl Path {
    fn kwargs(mut self, kwargs: Value) -> Self {
        self.kwargs = kwargs;
        self
    }
    fn tokens(mut self, want: &'static [&'static str]) -> Self {
        self.want = Some(want);
        self
    }
    fn rejected(mut self) -> Self {
        self.want = None;
        self
    }
    fn has(mut self, has: &'static [&'static str]) -> Self {
        self.has = has;
        self
    }
    fn lacks(mut self, lacks: &'static [&'static str]) -> Self {
        self.lacks = lacks;
        self
    }
    fn real_differs(mut self, why: &'static str) -> Self {
        self.real_differs = Some(why);
        self
    }
}

/// Every template path the engine drives, per family. `lloom chat` renders
/// a system message (`--system`), the user's turns and its own replies as
/// `content` (thinking text and all), with `enable_thinking` set where the
/// template reads it (`--think` / `--no-think`, off by default) and
/// `reasoning_effort` when given, and an image as a `{"type": "image"}` part.
/// `lloom serve` passes a client's messages through as sent (image URLs
/// become image parts), with `chat_template_kwargs` as the template's
/// variables (`tools`, `preserve_thinking`, …) and the top-level
/// `reasoning_effort`, `enable_thinking` and `tools` folded in. Every render
/// sets `add_generation_prompt`.
fn paths() -> Vec<Path> {
    use Family::*;
    let off = json!({ "enable_thinking": false });
    vec![
        // LFM2: its templates read no thinking variable; the tiny one opens
        // the think block as LFM2.5-2.6B's does.
        path(Lfm2, "user", json!([user("Hi")])).tokens(&[S, USER, END, ASST, THINK]),
        path(Lfm2, "system", json!([system("Be brief."), user("Hi")]))
            .tokens(&[S, SYS, END, USER, END, ASST, THINK])
            .has(&["Be brief."]),
        path(Lfm2, "empty_system", json!([system(""), user("Hi")]))
            .tokens(&[S, USER, END, ASST, THINK]),
        path(
            Lfm2,
            "text_parts",
            json!([{ "role": "user", "content": [
            { "type": "text", "text": "Hi " }, { "type": "text", "text": "there" } ] }]),
        )
        .tokens(&[S, USER, END, ASST, THINK])
        .has(&["Hi there"]),
        path(
            Lfm2,
            "image",
            json!([{ "role": "user", "content": [
            { "type": "image" }, { "type": "text", "text": "What is this?" } ] }]),
        )
        .tokens(&[S, USER, IMAGE, END, ASST, THINK]),
        path(
            Lfm2,
            "multi_turn",
            json!([user("Hi"), assistant("Hello"), user("Bye")]),
        )
        .tokens(&[S, USER, END, ASST, END, USER, END, ASST, THINK])
        .has(&["Hello"]),
        // `lloom chat`'s history: the reply as decoded, its thinking in the
        // content, dropped once a later user turn exists.
        path(
            Lfm2,
            "thinking_in_content_dropped",
            json!([user("Hi"), assistant("mull</think>Hello"), user("Bye")]),
        )
        .tokens(&[S, USER, END, ASST, END, USER, END, ASST, THINK])
        .has(&["Hello"])
        .lacks(&["mull"]),
        path(
            Lfm2,
            "thinking_in_content_kept_last",
            json!([user("Hi"), assistant("mull</think>Hello")]),
        )
        .tokens(&[S, USER, END, ASST, UNTHINK, END, ASST, THINK])
        .has(&["mull"]),
        path(
            Lfm2,
            "reasoning_content_dropped",
            json!([user("Hi"), reasoned("mull", "Hello"), user("Bye")]),
        )
        .tokens(&[S, USER, END, ASST, END, USER, END, ASST, THINK])
        .lacks(&["mull"]),
        path(
            Lfm2,
            "reasoning_content_kept_last",
            json!([user("Hi"), reasoned("mull", "Hello")]),
        )
        .tokens(&[S, USER, END, ASST, THINK, UNTHINK, END, ASST, THINK])
        .has(&["mull"]),
        path(
            Lfm2,
            "preserve_thinking",
            json!([user("Hi"), reasoned("mull", "Hello"), user("Bye")]),
        )
        .kwargs(json!({ "preserve_thinking": true }))
        .tokens(&[
            S, USER, END, ASST, THINK, UNTHINK, END, USER, END, ASST, THINK,
        ])
        .has(&["mull"]),
        path(Lfm2, "tools", json!([user("Weather?")]))
            .kwargs(json!({ "tools": tools() }))
            .tokens(&[S, SYS, END, USER, END, ASST, THINK])
            .has(&["get_weather", "get_time"]),
        path(
            Lfm2,
            "tools_and_system",
            json!([system("Be brief."), user("Weather?")]),
        )
        .kwargs(json!({ "tools": tools() }))
        .tokens(&[S, SYS, END, USER, END, ASST, THINK])
        .has(&["Be brief.", "get_weather"]),
        path(
            Lfm2,
            "tool_call_and_result",
            json!([user("Weather?"),
            { "role": "assistant", "content": null, "tool_calls": [weather_call()] },
            tool_result("sunny")]),
        )
        .tokens(&[
            S, USER, END, ASST, CALLS, UNCALLS, END, TOOL, END, ASST, THINK,
        ])
        .has(&["get_weather", "Paris", "sunny"]),
        path(
            Lfm2,
            "bare_tool_calls",
            json!([user("Weather?"),
            { "role": "assistant", "content": "Checking.", "tool_calls": [
                { "name": "get_weather", "arguments": { "city": "Paris" } },
                { "name": "get_time", "arguments": {} } ] }]),
        )
        .tokens(&[S, USER, END, ASST, CALLS, UNCALLS, END, ASST, THINK])
        .has(&["Checking.", "get_weather", "get_time"]),
        // OpenAI sends the arguments as a JSON string; the family's templates
        // refuse them (a 400 on `lloom serve`).
        path(
            Lfm2,
            "tool_call_arguments_as_a_string",
            json!([user("Weather?"),
            { "role": "assistant", "content": null, "tool_calls": [
                { "type": "function", "function": { "name": "get_weather",
                  "arguments": "{\"city\": \"Paris\"}" } } ] }]),
        )
        .rejected(),
        // Qwen4-exp: `lloom chat` renders thinking off unless `--think`;
        // `lloom serve` sets nothing unless the request does (thinking on).
        path(Qwen4Exp, "user", json!([user("Hi")]))
            .kwargs(off.clone())
            .tokens(&[USER, END, ASST, THINK, UNTHINK])
            .lacks(&["effort"]),
        path(Qwen4Exp, "thinking_by_default", json!([user("Hi")]))
            .tokens(&[SYS, END, USER, END, ASST, THINK])
            .has(&["Thinking effort: xhigh."]),
        path(Qwen4Exp, "effort_low", json!([user("Hi")]))
            .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "low" }))
            .tokens(&[SYS, END, USER, END, ASST, THINK])
            .has(&["Thinking effort: low."]),
        path(Qwen4Exp, "effort_medium", json!([user("Hi")]))
            .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "medium" }))
            .tokens(&[USER, END, ASST, THINK])
            .lacks(&["effort"]),
        path(Qwen4Exp, "effort_xhigh", json!([user("Hi")]))
            .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "xhigh" }))
            .tokens(&[SYS, END, USER, END, ASST, THINK])
            .has(&["Thinking effort: xhigh."]),
        path(Qwen4Exp, "effort_rejected", json!([user("Hi")]))
            .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "high" }))
            .rejected(),
        path(
            Qwen4Exp,
            "effort_unread_with_thinking_off",
            json!([user("Hi")]),
        )
        .kwargs(json!({ "enable_thinking": false, "reasoning_effort": "high" }))
        .tokens(&[USER, END, ASST, THINK, UNTHINK])
        .lacks(&["effort"]),
        path(Qwen4Exp, "system", json!([system("Be brief."), user("Hi")]))
            .kwargs(off.clone())
            .tokens(&[SYS, END, USER, END, ASST, THINK, UNTHINK])
            .has(&["Be brief."]),
        path(
            Qwen4Exp,
            "system_and_effort",
            json!([system("Be brief."), user("Hi")]),
        )
        .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "low" }))
        .tokens(&[SYS, END, USER, END, ASST, THINK])
        .has(&["Thinking effort: low.", "Be brief."]),
        path(Qwen4Exp, "empty_system", json!([system(""), user("Hi")]))
            .kwargs(off.clone())
            .tokens(&[USER, END, ASST, THINK, UNTHINK]),
        path(
            Qwen4Exp,
            "text_parts",
            json!([{ "role": "user", "content": [
            { "type": "text", "text": "Hi " }, { "type": "text", "text": "there" } ] }]),
        )
        .kwargs(off.clone())
        .tokens(&[USER, END, ASST, THINK, UNTHINK])
        .has(&["Hi there"]),
        path(
            Qwen4Exp,
            "image",
            json!([{ "role": "user", "content": [
            { "type": "image" }, { "type": "text", "text": "What is this?" } ] }]),
        )
        .kwargs(off.clone())
        .rejected()
        .real_differs(
            "the engine has no Qwen4-exp vision: `Served::image_placeholder` \
                refuses every image (served.rs), so the tiny template refuses the part",
        ),
        // History keeps the think block by default, empty when the reply
        // carried none; `preserve_thinking: false` drops it before the last
        // user turn.
        path(
            Qwen4Exp,
            "multi_turn",
            json!([user("Hi"), assistant("Hello"), user("Bye")]),
        )
        .kwargs(off.clone())
        .tokens(&[
            USER, END, ASST, THINK, UNTHINK, END, USER, END, ASST, THINK, UNTHINK,
        ])
        .has(&["Hello"]),
        path(
            Qwen4Exp,
            "reasoning_content_kept",
            json!([user("Hi"), reasoned("mull", "Hello"), user("Bye")]),
        )
        .kwargs(off.clone())
        .tokens(&[
            USER, END, ASST, THINK, UNTHINK, END, USER, END, ASST, THINK, UNTHINK,
        ])
        .has(&["mull"]),
        path(
            Qwen4Exp,
            "reasoning_content_dropped",
            json!([user("Hi"), reasoned("mull", "Hello"), user("Bye")]),
        )
        .kwargs(json!({ "enable_thinking": false, "preserve_thinking": false }))
        .tokens(&[USER, END, ASST, END, USER, END, ASST, THINK, UNTHINK])
        .lacks(&["mull"]),
        path(
            Qwen4Exp,
            "reasoning_content_kept_last",
            json!([user("Hi"), reasoned("mull", "Hello")]),
        )
        .kwargs(json!({ "enable_thinking": false, "preserve_thinking": false }))
        .tokens(&[USER, END, ASST, THINK, UNTHINK, END, ASST, THINK, UNTHINK])
        .has(&["mull"]),
        path(Qwen4Exp, "tools", json!([user("Weather?")]))
            .kwargs(json!({ "enable_thinking": false, "tools": tools() }))
            .tokens(&[
                SYS, CALL, UNCALL, CALL, UNCALL, END, USER, END, ASST, THINK, UNTHINK,
            ])
            .has(&["get_weather", "get_time"]),
        path(
            Qwen4Exp,
            "tools_system_and_effort",
            json!([system("Be brief."), user("Weather?")]),
        )
        .kwargs(json!({ "enable_thinking": true, "reasoning_effort": "low", "tools": tools() }))
        .tokens(&[SYS, CALL, UNCALL, CALL, UNCALL, END, USER, END, ASST, THINK])
        .has(&["Thinking effort: low.", "get_weather", "Be brief."]),
        path(
            Qwen4Exp,
            "tool_call_and_result",
            json!([user("Weather?"),
            { "role": "assistant", "content": null, "tool_calls": [weather_call()] },
            tool_result("sunny")]),
        )
        .kwargs(off.clone())
        .tokens(&[
            USER, END, ASST, THINK, UNTHINK, CALL, UNCALL, END, USER, RESP, UNRESP, END, ASST,
            THINK, UNTHINK,
        ])
        .has(&["get_weather", "Paris", "sunny"]),
        path(
            Qwen4Exp,
            "two_calls_two_results",
            json!([user("Weather?"),
            { "role": "assistant", "content": "Checking.", "tool_calls": [weather_call(),
                { "name": "get_time", "arguments": {} }] },
            tool_result("sunny"), tool_result("noon"), assistant("Sunny at noon."),
            user("Thanks")]),
        )
        .kwargs(off.clone())
        .tokens(&[
            USER, END, ASST, THINK, UNTHINK, CALL, UNCALL, CALL, UNCALL, END, USER, RESP, UNRESP,
            RESP, UNRESP, END, ASST, THINK, UNTHINK, END, USER, END, ASST, THINK, UNTHINK,
        ])
        .has(&["Checking.", "get_time", "noon"]),
        // OpenAI sends the arguments as a JSON string. The checkpoint's
        // template fails on them outside `raise_exception` (a 400 on
        // `lloom serve`, as the request's data); the tiny one renders the
        // string.
        path(
            Qwen4Exp,
            "tool_call_arguments_as_a_string",
            json!([user("Weather?"),
            { "role": "assistant", "content": null, "tool_calls": [
                { "type": "function", "function": { "name": "get_weather",
                  "arguments": "{\"city\": \"Paris\"}" } } ] }]),
        )
        .kwargs(off.clone())
        .tokens(&[
            USER, END, ASST, THINK, UNTHINK, CALL, UNCALL, END, ASST, THINK, UNTHINK,
        ])
        .real_differs(
            "the checkpoint's template iterates the arguments as a mapping and \
                fails on a string outside raise_exception; the tiny template is not built \
                to reproduce a render failure",
        ),
        path(
            Qwen4Exp,
            "system_not_first",
            json!([user("Hi"), system("Be brief.")]),
        )
        .kwargs(off.clone())
        .rejected(),
        path(
            Qwen4Exp,
            "no_user_message",
            json!([system("Be brief."), assistant("Hello")]),
        )
        .kwargs(off.clone())
        .rejected(),
        path(
            Qwen4Exp,
            "unknown_role",
            json!([user("Hi"), { "role": "developer", "content": "x" }]),
        )
        .kwargs(off)
        .rejected(),
    ]
}

/// A branch of a checkpoint's template, keyed by the hash of its condition
/// (`condition_key`): the paths above that take it, or why none does.
enum Branch {
    Paths(&'static [&'static str]),
    Gap(&'static str),
}

const EVERY_PATH: Branch = Branch::Paths(&[]);

/// LFM2.5-2.6B's and LFM2.5-VL-3B's branch conditions (the two differ only
/// in the generation prompt's text).
const LFM2_BRANCHES: &[(&str, &str, Branch)] = &[
    (
        "036350b0",
        "a tool call's argument value is a string",
        Branch::Paths(&["tool_call_and_result"]),
    ),
    (
        "e2398449",
        "a tool call's argument value is structured",
        Branch::Paths(&["tool_call_and_result"]),
    ),
    ("d90613f7", "content is a string", Branch::Paths(&["user"])),
    (
        "441f0074",
        "content is one object",
        Branch::Gap(
            "not an OpenAI content shape; neither front end builds one, and the tiny template \
         refuses what `lloom serve` passes through",
        ),
    ),
    (
        "ae9c0e7d",
        "content is a list of parts",
        Branch::Paths(&["text_parts", "image"]),
    ),
    (
        "e047c997",
        "a part is a bare string",
        Branch::Gap("not an OpenAI part shape; the tiny template refuses it"),
    ),
    ("8a424623", "an image part", Branch::Paths(&["image"])),
    (
        "a062d679",
        "a text part",
        Branch::Paths(&["text_parts", "image"]),
    ),
    (
        "f5ffc375",
        "a tool call wrapped in `function`",
        Branch::Paths(&["tool_call_and_result", "bare_tool_calls"]),
    ),
    (
        "daf593c3",
        "tool call arguments are an object",
        Branch::Paths(&["tool_call_and_result"]),
    ),
    (
        "d99d6d22",
        "tool call arguments are a string",
        Branch::Paths(&["tool_call_arguments_as_a_string"]),
    ),
    (
        "13411b11",
        "a system message first",
        Branch::Paths(&["system", "empty_system"]),
    ),
    (
        "ffb41319",
        "the system message has content",
        Branch::Paths(&["system", "empty_system"]),
    ),
    (
        "f9d35d43",
        "tools given",
        Branch::Paths(&["tools", "tools_and_system"]),
    ),
    (
        "c8f8bcaa",
        "the system prompt is not empty",
        Branch::Paths(&["tools_and_system", "system", "user"]),
    ),
    (
        "f857838f",
        "a tool given as an object",
        Branch::Paths(&["tools"]),
    ),
    ("58be8832", "between two tools", Branch::Paths(&["tools"])),
    (
        "e681e011",
        "a user turn (the last one decides what history keeps)",
        Branch::Paths(&["multi_turn", "reasoning_content_dropped"]),
    ),
    (
        "4f5e9f97",
        "an assistant turn",
        Branch::Paths(&["multi_turn"]),
    ),
    (
        "583fd580",
        "the reply's reasoning is a string",
        Branch::Paths(&["reasoning_content_kept_last"]),
    ),
    (
        "2b319b60",
        "the reply's reasoning is kept",
        Branch::Paths(&[
            "reasoning_content_kept_last",
            "reasoning_content_dropped",
            "preserve_thinking",
        ]),
    ),
    (
        "58808445",
        "a turn has content",
        Branch::Paths(&["multi_turn", "tool_call_and_result"]),
    ),
    (
        "c0bcfec0",
        "thinking in the reply's content is dropped",
        Branch::Paths(&[
            "thinking_in_content_dropped",
            "thinking_in_content_kept_last",
        ]),
    ),
    (
        "d3fd213f",
        "the content ends in transformers' continue-final-message marker",
        Branch::Gap(
            "transformers' `continue_final_message`; the engine always renders with \
         `add_generation_prompt` (template.rs) and never continues a final message",
        ),
    ),
    (
        "f7d97832",
        "the reply has tool calls",
        Branch::Paths(&["tool_call_and_result", "bare_tool_calls"]),
    ),
    (
        "29ebc396",
        "restoring the continue-final-message marker",
        Branch::Gap("as the marker above"),
    ),
    ("d86f4175", "the generation prompt", EVERY_PATH),
];

/// Qwen3.8-Flash-Next's branch conditions.
const QWEN_BRANCHES: &[(&str, &str, Branch)] = &[
    ("d90613f7", "content is a string", Branch::Paths(&["user"])),
    ("f917e1f1", "content is a list of parts", Branch::Paths(&["text_parts"])),
    ("7bd9b1a8", "an image part", Branch::Paths(&["image"])),
    ("c47ba6e8", "a picture or video in the system message", Branch::Gap("no Qwen4-exp vision (the image path above)")),
    ("2e1e5b7e", "counting pictures and videos", Branch::Gap("no Qwen4-exp vision")),
    ("e48de17e", "numbering pictures and videos (`add_vision_id`)", Branch::Gap("no Qwen4-exp vision")),
    ("03ae1314", "a video part", Branch::Gap("the engine has no video input on any family")),
    ("b45c5c74", "a text part", Branch::Paths(&["text_parts"])),
    ("5d8fa582", "no content", Branch::Paths(&["tool_call_and_result"])),
    ("fc0426af", "no messages", Branch::Gap(
        "unreachable: `lloom serve` refuses an empty `messages` before rendering (serve.rs), \
         and `lloom chat` renders a user turn")),
    ("f201a20f", "thinking on, or not given", Branch::Paths(&["thinking_by_default", "effort_low", "effort_unread_with_thinking_off"])),
    ("852591f3", "an effort it does not take", Branch::Paths(&["effort_rejected"])),
    ("89773ac8", "effort xhigh", Branch::Paths(&["effort_xhigh", "thinking_by_default"])),
    ("c4e7b2a9", "effort low", Branch::Paths(&["effort_low"])),
    ("ec63cb04", "tools given", Branch::Paths(&["tools", "tools_system_and_effort"])),
    ("3b0dd4ec", "an effort instruction to render", Branch::Paths(&["tools_system_and_effort", "system_and_effort", "effort_low", "effort_medium"])),
    ("0ba81dc8", "a system message first", Branch::Paths(&["system", "tools_system_and_effort", "empty_system"])),
    ("ed7002b4", "the system message has content", Branch::Paths(&["system", "empty_system"])),
    ("af6225a0", "the last user turn, searched from the end", Branch::Paths(&["multi_turn", "two_calls_two_results"])),
    ("bad247f2", "a user turn that is only a tool response", Branch::Gap(
        "tool results as user text, the older form; a client sends tool results as `role: tool` messages \
         (the tool_call_and_result path), which neither template counts as a user turn")),
    ("2ae62c5b", "no user turn", Branch::Paths(&["no_user_message"])),
    ("2aaadc85", "a system message in the loop", Branch::Paths(&["system", "system_not_first"])),
    ("83f54756", "not the first message", Branch::Paths(&["system_not_first"])),
    ("52158fa7", "a user turn", Branch::Paths(&["user"])),
    ("4f5e9f97", "an assistant turn", Branch::Paths(&["multi_turn"])),
    ("6740d105", "the reply's reasoning is a string", Branch::Paths(&["reasoning_content_kept"])),
    ("d176156a", "the reply's reasoning is kept", Branch::Paths(&["reasoning_content_kept", "reasoning_content_dropped", "reasoning_content_kept_last"])),
    ("1f8db93c", "the reply has tool calls", Branch::Paths(&["tool_call_and_result"])),
    ("ea9bf765", "a tool call wrapped in `function`", Branch::Paths(&["tool_call_and_result", "two_calls_two_results"])),
    ("a72e26ea", "the first tool call", Branch::Paths(&["two_calls_two_results"])),
    ("faa03a05", "the reply has text before its calls", Branch::Paths(&["two_calls_two_results", "tool_call_and_result"])),
    ("e126e51c", "tool call arguments given", Branch::Paths(&["tool_call_and_result", "two_calls_two_results"])),
    ("8e840415", "an argument value is a string", Branch::Paths(&["tool_call_and_result"])),
    ("20efedc9", "a tool result", Branch::Paths(&["tool_call_and_result"])),
    ("fdb2e860", "the first of consecutive tool results", Branch::Paths(&["two_calls_two_results"])),
    ("46e61003", "the last of consecutive tool results", Branch::Paths(&["two_calls_two_results"])),
    ("5b500bf7", "a tool result ends the conversation", Branch::Paths(&["tool_call_and_result"])),
    ("d86f4175", "the generation prompt", EVERY_PATH),
    ("8ac0517e", "thinking off", Branch::Paths(&["user", "effort_unread_with_thinking_off"])),
];

/// The variables a checkpoint's template reads that the tiny one does not.
const VARIABLE_GAPS: &[(Family, &str, &str)] = &[(
    Family::Qwen4Exp,
    "add_vision_id",
    "numbers pictures and videos; no Qwen4-exp vision in the engine",
)];

/// The checkpoints whose templates are checked. `opens_think`: the
/// generation prompt opens the think block (LFM2.5-VL-3B's does not; the
/// tiny LFM2 template follows LFM2.5-2.6B, the tokenizer's source).
const REAL: &[(Family, &str, bool)] = &[
    (Family::Lfm2, "LiquidAI/LFM2.5-2.6B", true),
    (Family::Lfm2, "LiquidAI/LFM2.5-VL-3B", false),
    (Family::Qwen4Exp, "Qwen/Qwen3.8-Flash-Next", true),
];

fn tiny(family: Family) -> Template {
    Template::compile(family.tokenizer().chat_template).expect("the tiny template compiles")
}

fn render(t: &Template, family: Family, p: &Path) -> (Outcome, String) {
    let kwargs: Map<String, Value> = p.kwargs.as_object().cloned().unwrap_or_default();
    match t.render(&p.messages, family.bos(), &kwargs) {
        Ok(text) => (Outcome::Tokens(special_tokens(family, &text)), text),
        Err(e) if e.downcast_ref::<Rejected>().is_some() => (Outcome::Rejected, format!("{e:#}")),
        Err(e) => (Outcome::Failed(format!("{e:#}")), String::new()),
    }
}

/// The special tokens `text` encodes to through the family's tiny tokenizer,
/// in order; an `<|im_start|>` carries the role that follows it.
fn special_tokens(family: Family, text: &str) -> Vec<String> {
    let tt = family.tokenizer();
    let tk = tokenizers::Tokenizer::from_bytes(lloom::testing::tiny::tokenizer::tokenizer_json(
        &tt.added,
    ))
    .expect("the tiny tokenizer loads");
    let ids = tk
        .encode(text, false)
        .expect("the rendering encodes")
        .get_ids()
        .to_vec();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let id = ids[i] as usize;
        i += 1;
        if id < 256 {
            continue;
        }
        let mut name = tt.added[id - 256].0.clone();
        if name == "<|im_start|>" {
            let mut role = Vec::new();
            while i < ids.len() && ids[i] < 256 && ids[i] != u32::from(b'\n') {
                role.push(ids[i] as u8);
                i += 1;
            }
            name.push_str(&String::from_utf8_lossy(&role));
        }
        out.push(name);
    }
    out
}

fn want(p: &Path) -> Outcome {
    match p.want {
        Some(t) => Outcome::Tokens(t.iter().map(|s| s.to_string()).collect()),
        None => Outcome::Rejected,
    }
}

fn find(family: Family, name: &str) -> Path {
    paths()
        .into_iter()
        .find(|p| p.family == family && p.name == name)
        .unwrap_or_else(|| panic!("no path {family:?}/{name}"))
}

/// One path through the tiny template: its tokens (or its refusal) and its
/// words.
fn check_path(family: Family, name: &str) {
    let p = find(family, name);
    let (got, text) = render(&tiny(family), family, &p);
    assert_eq!(got, want(&p), "{family:?}/{name}: {text}");
    for w in p.has {
        assert!(text.contains(w), "{family:?}/{name} lacks {w:?}: {text}");
    }
    for w in p.lacks {
        assert!(!text.contains(w), "{family:?}/{name} carries {w:?}: {text}");
    }
}

macro_rules! path_tests {
    ($family:ident: $($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_path(Family::$family, stringify!($name));
            }
        )*
        pub(super) const TESTED: &[&str] = &[$(stringify!($name)),*];
    };
}

mod lfm2 {
    use super::*;
    path_tests!(Lfm2:
        user, system, empty_system, text_parts, image, multi_turn,
        thinking_in_content_dropped, thinking_in_content_kept_last,
        reasoning_content_dropped, reasoning_content_kept_last, preserve_thinking,
        tools, tools_and_system, tool_call_and_result, bare_tool_calls,
        tool_call_arguments_as_a_string,
    );
}

mod qwen4exp {
    use super::*;
    path_tests!(Qwen4Exp:
        user, thinking_by_default, effort_low, effort_medium, effort_xhigh, effort_rejected,
        effort_unread_with_thinking_off, system, system_and_effort, empty_system, text_parts,
        image, multi_turn, reasoning_content_kept, reasoning_content_dropped,
        reasoning_content_kept_last, tools, tools_system_and_effort, tool_call_and_result,
        two_calls_two_results, tool_call_arguments_as_a_string, system_not_first,
        no_user_message, unknown_role,
    );
}

/// Every path has its test, and no test names a missing path.
#[test]
fn every_path_has_a_test() {
    for (family, tested) in [
        (Family::Lfm2, lfm2::TESTED),
        (Family::Qwen4Exp, qwen4exp::TESTED),
    ] {
        let listed: BTreeSet<&str> = paths()
            .iter()
            .filter(|p| p.family == family)
            .map(|p| p.name)
            .collect();
        let tested: BTreeSet<&str> = tested.iter().copied().collect();
        assert_eq!(listed, tested, "{family:?}");
    }
}

/// The variables the engine reads off each tiny template: the Qwen4-exp one
/// reads `reasoning_effort` (with thinking on) and `enable_thinking`; the
/// LFM2 one reads neither, so the engine's refusal of a variable the
/// template does not read keeps a subject (`tiny_chat_e2e.rs`).
#[test]
fn the_engine_variables_each_tiny_template_reads() {
    let (l, q) = (tiny(Family::Lfm2), tiny(Family::Qwen4Exp));
    for v in ["enable_thinking", "reasoning_effort"] {
        assert_eq!(q.reads(v), Some(true), "{v}");
        assert_eq!(l.reads(v), Some(false), "{v}");
    }
    for v in [
        "messages",
        "tools",
        "preserve_thinking",
        "add_generation_prompt",
    ] {
        assert_eq!(q.reads(v), Some(true), "{v}");
        assert_eq!(l.reads(v), Some(true), "{v}");
    }
    assert_eq!(l.reads("bos_token"), Some(true));
}

// ---- the path-coverage check against the checkpoints' own templates ----

fn hub() -> PathBuf {
    if let Ok(d) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
        });
    home.join("hub")
}

/// Every cached snapshot's `chat_template.jinja` of `repo` (the cache keeps
/// blobs behind symlinks; reading through the link follows it).
fn cached(repo: &str) -> Vec<(String, String)> {
    let dir = hub()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        if let Ok(src) = std::fs::read_to_string(e.path().join("chat_template.jinja")) {
            out.push((e.file_name().to_string_lossy().to_string(), src));
        }
    }
    out.sort();
    out
}

/// The branch conditions of a template's source, in order: each `if` /
/// `elif` tag's test and each inline `a if test else b`, whitespace
/// collapsed. String literals are masked while the keywords are found, so a
/// literal's words are not taken for syntax.
fn conditions(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] != b'{' {
            i += 1;
            continue;
        }
        let close: &[u8] = match b[i + 1] {
            b'%' => b"%}",
            b'{' => b"}}",
            b'#' => {
                i = src[i..].find("#}").map_or(b.len(), |e| i + e + 2);
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let block = b[i + 1] == b'%';
        let start = i + 2;
        let mut j = start;
        let mut masked = Vec::new();
        while j < b.len() && !b[j..].starts_with(close) {
            let c = b[j];
            masked.push(c);
            j += 1;
            if c == b'\'' || c == b'"' {
                while j < b.len() && b[j] != c {
                    let n = if b[j] == b'\\' { 2 } else { 1 };
                    masked.extend(std::iter::repeat_n(b'x', n));
                    j += n;
                }
                masked.push(c);
                j += 1;
            }
        }
        let end = j.min(b.len());
        masked.truncate(end - start);
        let raw = &src[start..end];
        let m = String::from_utf8(masked).expect("the mask keeps ASCII syntax");
        i = end + 2;
        let edge = |c: char| c == '-' || c == '+' || c.is_whitespace();
        let lead = m.len() - m.trim_start_matches(edge).len();
        let tail = m.trim_end_matches(edge).len();
        if tail <= lead {
            continue;
        }
        let (mb, rb) = (&m[lead..tail], &raw[lead..tail]);
        let mut from = 0;
        if block {
            for kw in ["if ", "elif "] {
                if mb.starts_with(kw) {
                    out.push(norm(&rb[kw.len()..]));
                    from = mb.len();
                }
            }
        }
        while let Some(p) = mb[from..].find(" if ").map(|p| from + p) {
            let c0 = p + " if ".len();
            let c1 = mb[c0..].find(" else ").map_or(mb.len(), |e| c0 + e);
            out.push(norm(&rb[c0..c1]));
            from = c1;
        }
    }
    out
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A condition's key: the first 4 bytes of its SHA-256, as hex.
fn condition_key(condition: &str) -> String {
    Sha256::digest(condition.as_bytes())
        .iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn branches(family: Family) -> &'static [(&'static str, &'static str, Branch)] {
    match family {
        Family::Lfm2 => LFM2_BRANCHES,
        Family::Qwen4Exp => QWEN_BRANCHES,
    }
}

/// The extractor on a source written here: the block tests, the inline
/// tests, nothing from inside a literal or a comment.
#[test]
fn the_condition_extractor_finds_block_and_inline_tests() {
    let src = "{#- if not this -#}{%- if a and b -%}x{%- elif c is string %}\
               {{ 'd' if e else ' if f else ' }}{% set g = h if i else j %}{% endif %}";
    assert_eq!(
        conditions(src),
        ["a and b", "c is string", "e", "i"]
            .map(String::from)
            .to_vec()
    );
}

/// The check against the cached checkpoint templates: each path renders to
/// the tiny template's tokens (or its refusal) through the real one too,
/// or differs as its gap says; every variable and every branch condition of
/// the real one is a path or a listed gap, and no listed entry is stale.
#[test]
fn real_templates_have_every_path_and_nothing_unlisted() {
    let paths = paths();
    let mut checked = BTreeMap::<Family, usize>::new();
    let mut seen_keys = BTreeMap::<Family, BTreeSet<String>>::new();
    let mut failures = Vec::new();
    for &(family, repo, opens_think) in REAL {
        for (rev, src) in cached(repo) {
            let at = format!("{repo}@{}", &rev[..rev.len().min(8)]);
            let real = match Template::compile(&src) {
                Ok(t) => t,
                Err(e) => {
                    failures.push(format!("{at}: does not compile: {e:#}"));
                    continue;
                }
            };
            *checked.entry(family).or_default() += 1;

            let tiny_vars = tiny(family)
                .variables()
                .cloned()
                .expect("the tiny template analyses");
            let Some(real_vars) = real.variables() else {
                failures.push(format!("{at}: its variables could not be analysed"));
                continue;
            };
            eprintln!("{at}: reads {real_vars:?}");
            for v in real_vars {
                let gap = VARIABLE_GAPS.iter().any(|g| g.0 == family && g.1 == v);
                if !tiny_vars.contains(v) && !gap {
                    failures.push(format!("{at}: reads `{v}`; the tiny template does not"));
                }
            }
            for &(f, v, _) in VARIABLE_GAPS {
                if f == family && (!real_vars.contains(v) || tiny_vars.contains(v)) {
                    failures.push(format!("{at}: the listed variable gap `{v}` is stale"));
                }
            }

            let listed: BTreeSet<&str> = branches(family).iter().map(|b| b.0).collect();
            for c in conditions(&src) {
                let key = condition_key(&c);
                if !listed.contains(key.as_str()) {
                    failures.push(format!(
                        "{at}: an unlisted branch {key} `{c}`: add a path or a gap"
                    ));
                }
                seen_keys.entry(family).or_default().insert(key);
            }

            for p in paths.iter().filter(|p| p.family == family) {
                let (mut want, _) = render(&tiny(family), family, p);
                if let (false, Outcome::Tokens(t)) = (opens_think, &mut want) {
                    if t.last().map(String::as_str) == Some(THINK) {
                        t.pop();
                    }
                }
                let (got, text) = render(&real, family, p);
                match (p.real_differs, got == want) {
                    (None, false) => failures.push(format!(
                        "{at}: path {}: the checkpoint renders {got:?}, the tiny template \
                         {want:?}\n{text}",
                        p.name
                    )),
                    (Some(why), true) => failures.push(format!(
                        "{at}: path {}: listed as differing ({why}) and renders the same",
                        p.name
                    )),
                    (Some(why), false) => {
                        eprintln!("{at}: path {}: differs as listed: {got:?} ({why})", p.name)
                    }
                    (None, true) => {}
                }
            }
        }
    }
    for (family, keys) in &seen_keys {
        for (key, what, branch) in branches(*family) {
            if !keys.contains(*key) {
                failures.push(format!(
                    "{family:?}: the listed branch {key} ({what}) is stale"
                ));
            }
            match branch {
                Branch::Paths(names) => {
                    for n in *names {
                        if !paths.iter().any(|p| p.family == *family && p.name == *n) {
                            failures.push(format!("{family:?}: branch {key} names no path {n}"));
                        }
                    }
                }
                Branch::Gap(why) => eprintln!("{family:?}: branch {key} ({what}) is a gap: {why}"),
            }
        }
    }
    for &(family, repo, _) in REAL {
        if !checked.contains_key(&family) {
            eprintln!(
                "skipped {family:?}: no {repo} chat_template.jinja under {:?}",
                hub()
            );
        }
    }
    eprintln!(
        "checked: {checked:?}, branches seen: {:?}",
        seen_keys
            .iter()
            .map(|(f, k)| (f, k.len()))
            .collect::<Vec<_>>()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
