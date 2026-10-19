//! The front end against the tiny templates (and the checkpoints' own, when
//! they are in the Hugging Face cache): one render per prompt, the ids a
//! fresh render gives when no reply of ours is in the history, the logged
//! ids where one is, the turn checkpoint where the next turn diverges, and
//! the prompt budget's spans against the re-rendered reference.

use super::*;
use crate::app::budgets;
use crate::app::common::{DecodeTiming, PrefillTiming};
use crate::app::family::{
    Capabilities, LFM2_CAPABILITIES, QWEN3_5_CAPABILITIES, QWEN4EXP_CAPABILITIES,
};
use crate::app::template::{Rendered, Template};
use crate::{DecodeParams, ReadbackArm, Stop};
use lloom::load_warning::LoadWarning;
use lloom::nn::ReuseAction;
use lloom::testing::tiny::TinyTokenizer;
use lloom_tensor::Device;

/// A family reached only through the seam: the text side and the template,
/// no model.
struct Stub {
    text: Text,
    caps: Capabilities,
    device: Device,
}

impl Stub {
    fn new(
        tokenizer: tokenizers::Tokenizer,
        template: &str,
        bos: &str,
        caps: Capabilities,
    ) -> Self {
        let bos_id = (!bos.is_empty())
            .then(|| tokenizer.token_to_id(bos))
            .flatten();
        Self {
            text: Text {
                tokenizer,
                template: Some(Template::compile(template).unwrap()),
                bos_token: bos.to_string(),
                bos_id,
                eos_ids: vec![0],
                seed_tokens: vec![1],
                think: Default::default(),
                gen_prompts: Default::default(),
            },
            caps,
            device: Device::Cpu,
        }
    }

    fn tiny(tok: &TinyTokenizer, bos: &str, caps: Capabilities) -> Self {
        Self::tiny_with(tok, tok.chat_template, bos, caps)
    }

    fn tiny_with(tok: &TinyTokenizer, template: &str, bos: &str, caps: Capabilities) -> Self {
        let json = lloom::testing::tiny::tokenizer::tokenizer_json(&tok.added);
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        Self::new(tokenizer, template, bos, caps)
    }

    /// The tiny Qwen3.5 template with Qwen3.6's rule for a past reply: the
    /// content is split at the end marker only when no `reasoning_content`
    /// came beside it.
    fn qwen3_6_like() -> Self {
        let tok = TinyTokenizer::qwen3_5();
        let split = r#"{%- if "</think>" in body -%}"#;
        assert_eq!(tok.chat_template.matches(split).count(), 1);
        let src = tok.chat_template.replace(
            split,
            r#"{%- if turn.reasoning_content is not string and "</think>" in body -%}"#,
        );
        Self::tiny_with(&tok, &src, "", QWEN3_5_CAPABILITIES)
    }

    fn lfm2() -> Self {
        Self::tiny(&TinyTokenizer::lfm2(), "<|startoftext|>", LFM2_CAPABILITIES)
    }

    fn qwen3_5() -> Self {
        Self::tiny(&TinyTokenizer::qwen3_5(), "", QWEN3_5_CAPABILITIES)
    }

    fn qwen4exp() -> Self {
        Self::tiny(&TinyTokenizer::qwen4exp(), "", QWEN4EXP_CAPABILITIES)
    }

    fn enc(&self, s: &str) -> Vec<u32> {
        self.text
            .tokenizer
            .encode(s, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    fn renders(&self) -> u64 {
        self.text.template.as_ref().unwrap().renders()
    }

    /// The prompt as the engine built it before the front end: render,
    /// encode with the BOS, the thinking switch.
    fn reference(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        want: Option<bool>,
    ) -> Vec<u32> {
        let t = self.text.template.as_ref().unwrap();
        let rendered = t.render(messages, &self.text.bos_token, kwargs).unwrap();
        let mut ids = self.enc(&rendered);
        if let Some(bos) = self.text.bos_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        thinking::apply(want, self.text.think_tags(), &mut ids).unwrap();
        ids
    }
}

impl Served for Stub {
    fn device(&self) -> &Device {
        &self.device
    }
    fn model_name(&self) -> &str {
        "stub"
    }
    fn kv_capacity(&self) -> usize {
        1 << 16
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
        &self.text.eos_ids
    }
    fn bos_id(&self) -> Option<u32> {
        self.text.bos_id
    }
    fn health(&self) -> Value {
        Value::Null
    }
    fn describe(&self) -> String {
        String::new()
    }
    fn capabilities(&self) -> Capabilities {
        self.caps
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        Vec::new()
    }
    fn note_load_warnings(&mut self, _: Vec<LoadWarning>) {}
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let mut ids = self.enc(text);
        if let (true, Some(bos)) = (add_special, self.text.bos_id) {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        Ok(ids)
    }
    fn token_str(&self, id: u32) -> String {
        self.text.tokenizer.decode(&[id], false).unwrap()
    }
    fn render_checked(&self, m: &Value, kw: &Map<String, Value>, c: &[&str]) -> Result<Rendered> {
        self.text
            .template
            .as_ref()
            .unwrap()
            .render_checked(m, &self.text.bos_token, kw, c)
    }
    fn template_reads(&self, name: &str) -> Option<bool> {
        self.text.template.as_ref()?.reads(name)
    }
    fn think_tags(&self) -> Option<ThinkTags> {
        self.text.think_tags().cloned()
    }
    fn generation_prompt(&self, kwarg: Option<bool>, want: Option<bool>) -> Option<Vec<u32>> {
        self.text
            .generation_prompt(kwarg, want)
            .map(<[u32]>::to_vec)
    }
    fn template_renders(&self) -> u64 {
        self.renders()
    }
    fn text(&self) -> Option<&Text> {
        Some(&self.text)
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
        unreachable!("the front end never decodes")
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
        Ok(())
    }
    fn shutdown(&self) {}
}

fn kw(engine: &dyn Served, want: Option<bool>) -> Map<String, Value> {
    let mut kwargs = Map::new();
    thinking::set_kwarg(engine, want, &mut kwargs);
    kwargs
}

/// [`kw`], with the `preserve_thinking` the front end sets by default
/// (`true` where the template has the switch: the history keeps the
/// reasoning).
fn kw_front(engine: &dyn Served, front: &Front, want: Option<bool>) -> Map<String, Value> {
    let mut kwargs = kw(engine, want);
    if front.constants().is_some_and(|c| c.persist_switch) {
        kwargs.insert(PRESERVE_THINKING.into(), Value::Bool(true));
    }
    kwargs
}

fn build(front: &Front, engine: &dyn Served, messages: &Value, want: Option<bool>) -> Prompt {
    front
        .prompt(
            engine,
            Request {
                messages,
                kwargs: kw(engine, want),
                want,
                add_bos: true,
                expand: None,
                history: None,
            },
        )
        .unwrap()
}

/// A reply the model "sampled": `ids`, decoded as the engine decodes them.
fn decoded(engine: &Stub, ids: Vec<u32>) -> Decoded {
    let text = engine.text.tokenizer.decode(&ids, false).unwrap();
    Decoded {
        text,
        tokens: ids,
        stop: Stop::Eos,
        reuse: ReuseAction::Fresh,
        reused: 0,
        prefilled: 0,
        turn: Default::default(),
        checkpoint: Default::default(),
        checkpoint_at: None,
        readback: ReadbackArm::Blocking,
        sampling: lloom::generation::Sampling::ArgMax,
        logits_sha256: None,
        stats: Default::default(),
        thinking: None,
        logprobs: None,
    }
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

fn user(s: &str) -> Value {
    json!({ "role": "user", "content": s })
}

fn assistant(s: &str) -> Value {
    json!({ "role": "assistant", "content": s })
}

/// Conversations covering the paths the templates have: a system message,
/// several turns, tool calls and results.
fn conversations() -> Vec<Value> {
    let call = json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{ "type": "function",
            "function": { "name": "get_weather", "arguments": { "city": "Paris" } } }],
    });
    let result = json!({ "role": "tool", "content": "{\"temperature\": 21}" });
    vec![
        json!([user("The capital of France?")]),
        json!([{ "role": "system", "content": "Be brief." }, user("Hi"), assistant("Hello!"), user("And now?")]),
        json!([
            user("one"),
            assistant("two"),
            user("three"),
            assistant("four"),
            user("five")
        ]),
        json!([
            user("Weather in Paris?"),
            call,
            result,
            result,
            assistant("It is 21 degrees."),
            user("Thanks")
        ]),
    ]
}

/// Every think mode a front end can be asked for on `engine`.
fn wants(engine: &dyn Served) -> Vec<Option<bool>> {
    let mut v = vec![None, Some(false)];
    if engine.template_reads(ENABLE_THINKING) == Some(true)
        || engine.think_tags().is_some_and(|t| {
            engine
                .generation_prompt(None, None)
                .is_some_and(|g| g.contains(&t.start))
        })
    {
        v.push(Some(true));
    }
    v
}

/// With no reply of ours in the history, a prompt is exactly what one
/// render, one encode and the thinking switch gave before the front end
/// (with the `preserve_thinking` it sets by default), on every tiny
/// template, every conversation and every think mode; it took one render
/// and one tokenization pass.
#[test]
fn a_prompt_without_our_replies_is_the_plain_render_in_one_pass() {
    for engine in [Stub::lfm2(), Stub::qwen3_5(), Stub::qwen4exp()] {
        let front = Front::new(&engine, SessionOptions::default());
        for conv in conversations() {
            for want in wants(&engine) {
                let before = (engine.renders(), front.tokenizations());
                let p = build(&front, &engine, &conv, want);
                assert_eq!(engine.renders() - before.0, 1, "{conv} {want:?}");
                assert_eq!(p.renders, 1);
                assert_eq!(front.tokenizations() - before.1, 1);
                let reference = engine.reference(&conv, &kw_front(&engine, &front, want), want);
                assert_eq!(p.ids, reference, "{} {conv} {want:?}", engine.caps.family);
                assert_eq!(p.spliced.matched, 0);
            }
        }
    }
}

/// The constants the tiny templates give: the generation prompt per think
/// mode is the one `Text` reads at load, and the turn checkpoint's tail is
/// where the next turn's rendering of this one diverges — the generation
/// prompt's open block (Qwen3.5's `<think>\n`, LFM2's `<think>`) when the
/// template strips past reasoning, nothing when it keeps it.
#[test]
fn the_constants_place_the_checkpoint_where_the_next_turn_diverges() {
    let tail = |e: &Stub, kwarg: Option<bool>, want: Option<bool>, persist: Option<bool>| {
        let front = Front::new(e, SessionOptions::default());
        let m = front
            .constants()
            .unwrap()
            .mode(kwarg, want)
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(
            Some(m.gen.as_slice()),
            e.text.generation_prompt(kwarg, want)
        );
        m.keep[slot(persist)].map(|k| m.gen.len() - k)
    };
    let q = Stub::qwen3_5();
    assert_eq!(tail(&q, None, None, None), Some(2), "<think>, \\n");
    assert_eq!(tail(&q, None, None, Some(true)), Some(0));
    assert_eq!(tail(&q, Some(false), Some(false), None), Some(6));
    assert_eq!(tail(&q, Some(false), Some(false), Some(true)), Some(0));
    let l = Stub::lfm2();
    assert_eq!(tail(&l, None, None, None), Some(1), "<think>");
    assert_eq!(tail(&l, None, None, Some(true)), Some(0));
    assert_eq!(
        tail(&l, None, Some(false), None),
        Some(2),
        "<think></think>"
    );
    let x = Stub::qwen4exp();
    assert_eq!(
        tail(&x, None, None, None),
        Some(0),
        "Qwen4-exp keeps reasoning"
    );
    assert_eq!(tail(&x, None, None, Some(false)), Some(2));
    let c = |e: &Stub| {
        let f = Front::new(e, SessionOptions::default());
        let c = f.constants().unwrap();
        (c.persists, c.persist_switch, c.open.clone(), c.groups_tools)
    };
    assert_eq!(
        c(&q),
        (Some(false), true, Some("<|im_start|>".into()), true)
    );
    assert_eq!(
        c(&l),
        (Some(false), true, Some("<|im_start|>".into()), false)
    );
    assert_eq!(c(&x), (Some(true), true, Some("<|im_start|>".into()), true));
}

/// The DoD's test. The model sampled `<think>` as seven byte tokens — not
/// the text's canonical tokenization, which is the one added token. The next
/// turn's prompt carries those seven ids exactly, so the session's log
/// (turn 1's prompt and reply) is a prefix of it and `prepare` extends.
/// Re-tokenizing the reply instead puts the added token there, and the
/// prefix property — the test's assertion — fails.
#[test]
fn the_next_turn_reuses_the_sampled_ids_not_their_canonical_tokenization() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let want = Some(false);
    let first = json!([user("Spell the tag.")]);
    let p1 = build(&front, &engine, &first, want);
    assert_eq!(p1.turn_tail, 0, "this template keeps the generation prompt");

    let sampled = bytes("It is <think> in text.");
    let reply = decoded(&engine, sampled.clone());
    assert_ne!(
        engine.enc(&reply.text),
        sampled,
        "the precondition: not canonical"
    );
    assert!(front.record(&engine, p1.origin, &reply));

    let second = json!([
        user("Spell the tag."),
        assistant(&reply.text),
        user("Again?")
    ]);
    let p2 = build(&front, &engine, &second, want);
    let mut log = p1.ids.clone();
    log.extend_from_slice(&sampled);
    assert!(
        p2.ids.starts_with(&log),
        "the session log is not a prefix of the next turn's prompt:\n log  {log:?}\n next {:?}",
        &p2.ids[..log.len().min(p2.ids.len())]
    );
    assert_eq!(p2.spliced.matched, 1);
    assert_eq!(p2.spliced.spliced_tokens, sampled.len());
    assert_eq!(p2.renders, 1);
    // The text is the template's own rendering, whatever the ids.
    assert_eq!(
        engine.text.tokenizer.decode(&p2.ids, false).unwrap(),
        engine
            .text
            .tokenizer
            .decode(&engine.reference(&second, &kw(&engine, want), want), false)
            .unwrap()
    );
}

/// A reply with reasoning, on a template whose own default strips it.
/// Stripped (`--strip-thinking`, opt-in): the next turn's prompt equals turn
/// 1's up to the checkpoint (the tail the constants gave), then carries the
/// reply's content as sampled. Persisted (the default, or
/// `preserve_thinking`): the whole log is a prefix.
#[test]
fn reasoning_stripped_splices_the_content_and_persisted_extends() {
    for persist in [Some(false), None, Some(true)] {
        let engine = Stub::qwen3_5();
        let opts = SessionOptions {
            persist,
            ..SessionOptions::default()
        };
        let front = Front::new(&engine, opts);
        let first = json!([user("2+2?")]);
        let p1 = build(&front, &engine, &first, Some(true));
        let end = TinyTokenizer::qwen3_5().id("</think>").unwrap();
        let mut sampled = bytes("Two and two.\n");
        sampled.push(end);
        sampled.extend(bytes("\n\nFour."));
        let reply = decoded(&engine, sampled.clone());
        assert!(front.record(&engine, p1.origin, &reply));
        let second = json!([user("2+2?"), assistant(&reply.text), user("3+3?")]);
        let p2 = build(&front, &engine, &second, Some(true));
        assert_eq!(p2.spliced.matched, 1);
        let kept = p1.ids.len() - p1.turn_tail;
        assert_eq!(p2.ids[..kept], p1.ids[..kept], "{persist:?}");
        match persist {
            Some(false) => {
                assert_eq!(p1.turn_tail, 2);
                assert_ne!(p2.ids[kept], p1.ids[kept], "diverges at the checkpoint");
                let four = bytes("Four.");
                assert_eq!(&p2.ids[kept..kept + four.len()], &four[..]);
                assert_eq!(p2.spliced.spliced_tokens, four.len());
            }
            None | Some(true) => {
                assert_eq!(p1.turn_tail, 0);
                let mut log = p1.ids.clone();
                log.extend_from_slice(&sampled);
                assert!(p2.ids.starts_with(&log), "persisted: the log extends");
            }
        }
    }
}

/// A client that drops the reasoning on a template that keeps it: as sent,
/// the history's reasoning block is empty and the prompt diverges from the
/// log at the reply; restored from the log (`session.missing_reasoning =
/// log`), the log is a prefix.
#[test]
fn a_dropped_reasoning_is_rendered_as_sent_or_restored_from_the_log() {
    let end = TinyTokenizer::qwen4exp().id("</think>").unwrap();
    for restore in [false, true] {
        let engine = Stub::qwen4exp();
        let front = Front::new(
            &engine,
            SessionOptions {
                restore_reasoning: restore,
                ..SessionOptions::default()
            },
        );
        let first = json!([user("2+2?")]);
        let p1 = build(&front, &engine, &first, Some(true));
        let mut sampled = bytes("Add them.\n");
        sampled.push(end);
        sampled.extend(bytes("\n\nFour."));
        assert!(front.record(&engine, p1.origin, &decoded(&engine, sampled.clone())));
        let second = json!([user("2+2?"), assistant("Four."), user("3+3?")]);
        let p2 = build(&front, &engine, &second, Some(true));
        let mut log = p1.ids.clone();
        log.extend_from_slice(&sampled);
        assert_eq!(p2.spliced.matched, 1);
        assert_eq!(p2.spliced.restored, usize::from(restore));
        assert_eq!(p2.ids.starts_with(&log), restore, "restore={restore}");
    }
}

/// A reply the client edited is not ours: tokenized as sent.
#[test]
fn an_edited_reply_is_tokenized_as_sent() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let first = json!([user("Hi")]);
    let p1 = build(&front, &engine, &first, Some(false));
    let sampled = bytes("It is <think> in text.");
    assert!(front.record(&engine, p1.origin, &decoded(&engine, sampled)));
    let second = json!([user("Hi"), assistant("It is <think> in TEXT."), user("Ok")]);
    let p2 = build(&front, &engine, &second, Some(false));
    assert_eq!(p2.spliced.matched, 0);
    assert_eq!(
        p2.ids,
        engine.reference(&second, &kw(&engine, Some(false)), Some(false))
    );
}

/// A front end that restores a dropped reasoning from the log
/// (`session.missing_reasoning = log`).
fn restoring(engine: &Stub) -> Front {
    Front::new(
        engine,
        SessionOptions {
            restore_reasoning: true,
            ..SessionOptions::default()
        },
    )
}

/// A reply that reasoned `reasoning` and answered `content`, as sampled
/// (the tiny Qwen4-exp template's end marker between them).
fn reasoned(reasoning: &str, content: &str) -> Vec<u32> {
    let end = TinyTokenizer::qwen4exp().id("</think>").unwrap();
    let mut ids = bytes(reasoning);
    ids.push(end);
    ids.extend(bytes(content));
    ids
}

fn build_kw(
    front: &Front,
    engine: &dyn Served,
    messages: &Value,
    want: Option<bool>,
    extra: &[(&str, Value)],
) -> Prompt {
    let mut kwargs = kw(engine, want);
    for (k, v) in extra {
        kwargs.insert((*k).into(), v.clone());
    }
    front
        .prompt(
            engine,
            Request {
                messages,
                kwargs,
                want,
                add_bos: true,
                expand: None,
                history: None,
            },
        )
        .unwrap()
}

fn with(history: &Value, more: &[Value]) -> Value {
    let mut v = history.as_array().unwrap().clone();
    v.extend_from_slice(more);
    json!(v)
}

/// Two conversations answer with the same text after different reasoning,
/// turn after turn, interleaved, in both orders. Each conversation's next
/// turn restores its own reply's reasoning and runs its own sampled ids, so
/// its log is a prefix of the next prompt, whichever conversation logged
/// last; the other's reasoning never appears in it.
#[test]
fn two_conversations_answering_the_same_text_keep_their_own_reasoning() {
    for reversed in [false, true] {
        let engine = Stub::qwen4exp();
        let front = restoring(&engine);
        let mut convs = [
            (
                "Is it raining?",
                "Clouds",
                json!([user("Is it raining?")]),
                Vec::new(),
            ),
            (
                "Is it cold?",
                "Frost",
                json!([user("Is it cold?")]),
                Vec::new(),
            ),
        ];
        if reversed {
            convs.reverse();
        }
        for round in 0..3 {
            for c in convs.iter_mut() {
                let (_, why, history, log) = c;
                let p = build(&front, &engine, history, Some(true));
                assert!(
                    p.ids.starts_with(log),
                    "reversed={reversed} round {round} {why}: its own log is not a prefix:\n log  {log:?}\n next {:?}",
                    &p.ids[..log.len().min(p.ids.len())]
                );
                let text = engine.text.tokenizer.decode(&p.ids, false).unwrap();
                let other = if *why == "Clouds" { "Frost" } else { "Clouds" };
                assert!(
                    !text.contains(other),
                    "{why}: {other}'s reasoning restored: {text:?}"
                );
                assert_eq!(p.spliced.restored, round, "{why} round {round}");
                let sampled = reasoned(&format!("{why} {round}.\n"), "\n\nYes.");
                assert!(front.record(&engine, p.origin, &decoded(&engine, sampled.clone())));
                *log = p.ids.clone();
                log.extend_from_slice(&sampled);
                *history = with(history, &[assistant("Yes."), user("Sure?")]);
            }
        }
    }
}

/// The same text sampled as different ids in two conversations (one as
/// bytes, one canonically): each next turn runs its own ids.
#[test]
fn the_same_text_sampled_as_different_ids_splices_each_conversations_own() {
    let text = "It is <think> in text.";
    for reversed in [false, true] {
        let engine = Stub::qwen4exp();
        let front = Front::new(&engine, SessionOptions::default());
        let canonical = engine.enc(text);
        assert_ne!(
            bytes(text),
            canonical,
            "the precondition: two tokenizations"
        );
        let mut convs = [
            (json!([user("Spell it.")]), bytes(text)),
            (json!([user("Say it.")]), canonical),
        ];
        if reversed {
            convs.reverse();
        }
        let prompts: Vec<Prompt> = convs
            .iter()
            .map(|(q, _)| build(&front, &engine, q, Some(false)))
            .collect();
        for ((_, ids), p) in convs.iter().zip(&prompts) {
            assert!(front.record(&engine, p.origin, &decoded(&engine, ids.clone())));
        }
        for ((q, ids), p) in convs.iter().zip(&prompts) {
            let next = build(
                &front,
                &engine,
                &with(q, &[assistant(text), user("Again?")]),
                Some(false),
            );
            let mut log = p.ids.clone();
            log.extend_from_slice(ids);
            assert!(
                next.ids.starts_with(&log),
                "reversed={reversed} {q}: not its own ids"
            );
            assert_eq!(next.spliced.spliced_tokens, ids.len());
        }
    }
}

/// A history that never received a reply of ours, carrying the same text,
/// takes nothing from the log: no ids, no reasoning. It is the request as
/// sent, and the miss is counted.
#[test]
fn a_history_that_never_received_the_reply_cannot_restore_it() {
    let engine = Stub::qwen4exp();
    let front = restoring(&engine);
    let p = build(
        &front,
        &engine,
        &json!([user("Is it raining?")]),
        Some(true),
    );
    let sampled = reasoned("Clouds.\n", "\n\nYes.");
    assert!(front.record(&engine, p.origin, &decoded(&engine, sampled)));
    let foreign = json!([user("Is it cold?"), assistant("Yes."), user("Sure?")]);
    let q = build(&front, &engine, &foreign, Some(true));
    assert_eq!((q.spliced.matched, q.spliced.restored), (0, 0));
    assert_eq!(q.spliced.unverified, 1);
    assert_eq!(
        q.ids,
        engine.reference(&foreign, &kw(&engine, Some(true)), Some(true))
    );
    let text = engine.text.tokenizer.decode(&q.ids, false).unwrap();
    assert!(!text.contains("Clouds"), "{text:?}");
}

/// An earlier message the client changed makes a new conversation: the
/// reply after it is not the logged one, and goes as sent.
#[test]
fn an_edited_earlier_message_makes_the_reply_unverified() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &json!([user("Hi")]), Some(false));
    let sampled = bytes("It is <think> in text.");
    let reply = decoded(&engine, sampled);
    assert!(front.record(&engine, p1.origin, &reply));
    let edited = json!([user("Hello"), assistant(&reply.text), user("Ok")]);
    let p2 = build(&front, &engine, &edited, Some(false));
    assert_eq!((p2.spliced.matched, p2.spliced.unverified), (0, 1));
    assert_eq!(
        p2.ids,
        engine.reference(&edited, &kw(&engine, Some(false)), Some(false))
    );
}

/// A reasoning case: its name, the reply message, the request's extra
/// template variables, and whether the log restores.
type Case<'a> = (&'a str, Value, Vec<(&'a str, Value)>, bool);

/// What a history message says about its reasoning decides, under `log`:
/// absent is restored; `null`, `""` and an explicit strip are kept as sent
/// and counted; a reply whose reasoning the template keeps after the last
/// user turn (a tool loop) is not restored under an explicit strip either.
#[test]
fn absent_null_empty_and_stripped_reasoning_are_distinct() {
    let engine = Stub::qwen4exp();
    let q = json!([user("2+2?")]);
    let sampled = reasoned("Add them.\n", "\n\nFour.");
    let cases: [Case; 4] = [
        ("absent", assistant("Four."), vec![], true),
        (
            "null",
            json!({ "role": "assistant", "content": "Four.", "reasoning_content": null }),
            vec![],
            false,
        ),
        (
            "empty",
            json!({ "role": "assistant", "content": "Four.", "reasoning_content": "" }),
            vec![],
            false,
        ),
        (
            "stripped",
            assistant("Four."),
            vec![(PRESERVE_THINKING, json!(false))],
            false,
        ),
    ];
    for (name, reply, extra, restored) in cases {
        let front = restoring(&engine);
        let p1 = build_kw(&front, &engine, &q, Some(true), &extra);
        assert!(front.record(&engine, p1.origin, &decoded(&engine, sampled.clone())));
        let next = with(&q, &[reply, user("3+3?")]);
        let p2 = build_kw(&front, &engine, &next, Some(true), &extra);
        assert_eq!(p2.spliced.matched, 1, "{name}");
        assert_eq!(p2.spliced.restored, usize::from(restored), "{name}");
        // `null` and `""` decline what the log would restore; the strip
        // drops the reply's reasoning from the render, so nothing is wanted.
        let declined = matches!(name, "null" | "empty");
        assert_eq!(p2.spliced.declined, usize::from(declined), "{name}");
        let text = engine.text.tokenizer.decode(&p2.ids, false).unwrap();
        assert_eq!(text.contains("Add them"), restored, "{name}: {text:?}");
    }
    // A tool loop: the reply is after the last user turn, where the
    // template keeps reasoning whatever `preserve_thinking` says. Absent, it
    // is restored; under an explicit strip, it is declined.
    for strip in [false, true] {
        let front = restoring(&engine);
        let extra: Vec<(&str, Value)> = if strip {
            vec![(PRESERVE_THINKING, json!(false))]
        } else {
            vec![]
        };
        let q = json!([user("Weather?")]);
        let p1 = build_kw(&front, &engine, &q, Some(true), &extra);
        let sampled = reasoned("Need the tool.\n", "\n\nChecking.");
        assert!(front.record(&engine, p1.origin, &decoded(&engine, sampled)));
        let next = with(
            &q,
            &[
                assistant("Checking."),
                json!({ "role": "tool", "content": "21" }),
            ],
        );
        let p2 = build_kw(&front, &engine, &next, Some(true), &extra);
        assert_eq!(p2.spliced.matched, 1, "strip={strip}");
        assert_eq!(p2.spliced.restored, usize::from(!strip), "strip={strip}");
        assert_eq!(p2.spliced.declined, usize::from(strip), "strip={strip}");
        let text = engine.text.tokenizer.decode(&p2.ids, false).unwrap();
        assert_eq!(
            text.contains("Need the tool"),
            !strip,
            "strip={strip}: {text:?}"
        );
    }
}

/// Whitespace the client trimmed off a reply leaves it ours: matched, its
/// ids spliced where its text lies, its reasoning restored under `log`.
#[test]
fn a_trimmed_reply_is_still_ours() {
    let engine = Stub::qwen4exp();
    let front = restoring(&engine);
    let q = json!([user("2+2?")]);
    let p1 = build(&front, &engine, &q, Some(true));
    assert!(front.record(
        &engine,
        p1.origin,
        &decoded(&engine, reasoned("Add.\n", "\n\nFour.\n"))
    ));
    let p2 = build(
        &front,
        &engine,
        &with(&q, &[assistant("Four."), user("3+3?")]),
        Some(true),
    );
    assert_eq!((p2.spliced.matched, p2.spliced.restored), (1, 1));
    assert!(p2.spliced.spliced_tokens >= bytes("Four.").len());
}

/// The tools a conversation was rendered under are part of its scope: two
/// conversations alike but for their tools each run their own reply.
#[test]
fn the_tools_are_part_of_the_conversation() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let text = "It is <think> in text.";
    let tool = |name: &str| json!([{ "type": "function", "function": { "name": name } }]);
    let q = json!([user("Spell it.")]);
    let arms = [(tool("a"), bytes(text)), (tool("b"), engine.enc(text))];
    let prompts: Vec<Prompt> = arms
        .iter()
        .map(|(t, _)| build_kw(&front, &engine, &q, Some(false), &[(TOOLS, t.clone())]))
        .collect();
    for ((_, ids), p) in arms.iter().zip(&prompts) {
        assert!(front.record(&engine, p.origin, &decoded(&engine, ids.clone())));
    }
    for ((t, ids), p) in arms.iter().zip(&prompts) {
        let next = build_kw(
            &front,
            &engine,
            &with(&q, &[assistant(text), user("Again?")]),
            Some(false),
            &[(TOOLS, t.clone())],
        );
        let mut log = p.ids.clone();
        log.extend_from_slice(ids);
        assert!(next.ids.starts_with(&log), "{t}: not its own ids");
    }
}

/// A reply's scope is the history as the client sent it: two
/// conversations whose rendered messages are alike but whose images differ
/// (the render sees placeholders) each run their own reply.
#[test]
fn the_images_a_conversation_sent_are_part_of_it() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let text = "It is <think> in text.";
    let sent = |url: &str, more: &[Value]| {
        let mut v = vec![json!({ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": url } },
            { "type": "text", "text": "Look." },
        ] })];
        v.extend_from_slice(more);
        json!(v)
    };
    let prompt = |messages: &Value, history: &Value| {
        front
            .prompt(
                &engine,
                Request {
                    messages,
                    kwargs: kw(&engine, Some(false)),
                    want: Some(false),
                    add_bos: true,
                    expand: None,
                    history: Some(history),
                },
            )
            .unwrap()
    };
    let shown = json!([user("Look.")]);
    let arms = [("data:a", bytes(text)), ("data:b", engine.enc(text))];
    let firsts: Vec<Prompt> = arms
        .iter()
        .map(|(u, _)| prompt(&shown, &sent(u, &[])))
        .collect();
    for ((_, ids), p) in arms.iter().zip(&firsts) {
        assert!(front.record(&engine, p.origin, &decoded(&engine, ids.clone())));
    }
    let more = [assistant(text), user("Again?")];
    for ((u, ids), p) in arms.iter().zip(&firsts) {
        let next = prompt(&with(&shown, &more), &sent(u, &more));
        let mut log = p.ids.clone();
        log.extend_from_slice(ids);
        assert!(next.ids.starts_with(&log), "{u}: not its own ids");
    }
}

/// Two distinct replies of one conversation with the same text (a
/// regeneration that reasoned otherwise) leave the text ambiguous: it goes
/// as sent, nothing restored. The same reply logged twice is one.
#[test]
fn a_regenerated_reply_with_the_same_text_is_ambiguous() {
    let engine = Stub::qwen4exp();
    let q = json!([user("2+2?")]);
    let next = with(&q, &[assistant("Four."), user("3+3?")]);
    for distinct in [true, false] {
        let front = restoring(&engine);
        let p1 = build(&front, &engine, &q, Some(true));
        let a = reasoned("Add them.\n", "\n\nFour.");
        let b = if distinct {
            reasoned("Count up.\n", "\n\nFour.")
        } else {
            a.clone()
        };
        assert!(front.record(&engine, p1.origin, &decoded(&engine, a)));
        assert!(front.record(&engine, p1.origin, &decoded(&engine, b)));
        let p2 = build(&front, &engine, &next, Some(true));
        assert_eq!(
            p2.spliced.ambiguous,
            usize::from(distinct),
            "distinct={distinct}"
        );
        assert_eq!(
            p2.spliced.restored,
            usize::from(!distinct),
            "distinct={distinct}"
        );
        if distinct {
            assert_eq!(
                p2.ids,
                engine.reference(&next, &kw(&engine, Some(true)), Some(true))
            );
        }
    }
}

/// A reply the log no longer holds — evicted, or logged by a process that
/// has since restarted — goes as sent.
#[test]
fn an_evicted_or_restarted_reply_goes_as_sent() {
    let engine = Stub::qwen4exp();
    let q = json!([user("Spell it.")]);
    let text = "It is <think> in text.";
    let next = with(&q, &[assistant(text), user("Again?")]);
    let reference = engine.reference(&next, &kw(&engine, Some(false)), Some(false));
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &q, Some(false));
    assert!(front.record(&engine, p1.origin, &decoded(&engine, bytes(text))));
    assert_ne!(
        build(&front, &engine, &next, Some(false)).ids,
        reference,
        "spliced while logged"
    );
    let restarted = Front::new(&engine, SessionOptions::default());
    let p = build(&restarted, &engine, &next, Some(false));
    assert_eq!(
        (p.spliced.matched, p.ids == reference),
        (0, true),
        "restarted"
    );
    for i in 0..LOG_REPLIES {
        assert!(front.record(
            &engine,
            p1.origin,
            &decoded(&engine, bytes(&format!("other {i}")))
        ));
    }
    let p = build(&front, &engine, &next, Some(false));
    assert_eq!(
        (p.spliced.matched, p.ids == reference),
        (0, true),
        "evicted"
    );
}

/// Conversations served concurrently through one front end each extend
/// their own log, every turn, though every reply has the same text.
#[test]
fn concurrent_conversations_keep_their_own_replies() {
    let engine = Stub::qwen4exp();
    let front = restoring(&engine);
    std::thread::scope(|s| {
        for c in 0..4 {
            let (front, engine) = (&front, &engine);
            s.spawn(move || {
                let mut history = json!([user(&format!("Question {c}?"))]);
                let mut log: Vec<u32> = Vec::new();
                for round in 0..6 {
                    let p = build(front, engine, &history, Some(true));
                    assert!(p.ids.starts_with(&log), "conversation {c} round {round}");
                    let sampled =
                        reasoned(&format!("Conversation {c} round {round}.\n"), "\n\nYes.");
                    assert!(front.record(engine, p.origin, &decoded(engine, sampled.clone())));
                    log = p.ids.clone();
                    log.extend_from_slice(&sampled);
                    history = with(&history, &[assistant("Yes."), user("Again?")]);
                }
            });
        }
    });
}

/// The prompt budget's cut, as spans of the one render's ids, is the
/// re-rendered shorter conversation's prompt (`budgets::fit_messages`) on
/// every tiny template and every budget that cuts.
#[test]
fn the_budget_cuts_spans_that_equal_the_rerendered_conversation() {
    let long = json!([
        { "role": "system", "content": "Be brief." },
        user("one"), assistant("two"), user("three"),
        { "role": "assistant", "content": "", "tool_calls": [{ "type": "function",
            "function": { "name": "f", "arguments": { "x": "1" } } }] },
        { "role": "tool", "content": "r1" }, { "role": "tool", "content": "r2" },
        assistant("four"), user("five"), assistant("six"), user("seven"),
    ]);
    for engine in [Stub::lfm2(), Stub::qwen3_5(), Stub::qwen4exp()] {
        let front = Front::new(&engine, SessionOptions::default());
        for want in wants(&engine) {
            let p = build(&front, &engine, &long, want);
            let msgs = long.as_array().unwrap();
            let kwargs = kw_front(&engine, &front, want);
            let mut cuts = 0;
            for budget in (1..p.ids.len()).rev() {
                let spans = front.fit(&p, msgs, budget).expect("the spans are known");
                let reference = budgets::fit_messages(msgs, budget, p.ids.len(), |ms| {
                    Ok(engine.reference(&json!(ms), &kwargs, want))
                })
                .unwrap();
                match (spans, reference) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(
                            a.ids, b.ids,
                            "{} {want:?} budget {budget}",
                            engine.caps.family
                        );
                        assert_eq!(a.messages, b.messages);
                        assert_eq!(a.dropped, b.dropped);
                        cuts += 1;
                    }
                    (a, b) => panic!(
                        "{} budget {budget}: spans {} reference {}",
                        engine.caps.family,
                        a.is_some(),
                        b.is_some()
                    ),
                }
            }
            assert!(cuts > 0, "no budget cut anything");
        }
    }
}

/// The logged logprobs ride with the ids they belong to: the next turn's
/// prompt carries each spliced id's, and nothing elsewhere.
#[test]
fn kept_logprobs_line_up_with_the_spliced_ids() {
    let engine = Stub::qwen4exp();
    let front = Front::new(
        &engine,
        SessionOptions {
            keep: KeepLogprobs::Sampled,
            ..SessionOptions::default()
        },
    );
    let p1 = build(&front, &engine, &json!([user("Hi")]), Some(false));
    let sampled = bytes("Hello there.");
    let lp: Vec<f32> = (0..sampled.len()).map(|i| -(i as f32) - 0.5).collect();
    let mut d = decoded(&engine, sampled.clone());
    d.logprobs = Some(crate::Logprobs {
        sampled: lp.clone(),
        top: vec![Vec::new(); sampled.len()],
        rows: None,
    });
    assert!(front.record(&engine, p1.origin, &d));
    assert_eq!(
        front.reply("Hello there.").unwrap().logprobs.as_deref(),
        Some(&lp[..])
    );
    let second = json!([user("Hi"), assistant("Hello there."), user("Bye")]);
    let p2 = build(&front, &engine, &second, Some(false));
    let got = p2.logprobs.expect("kept");
    assert_eq!(got.len(), p2.ids.len());
    let at = p1.ids.len();
    assert_eq!(&p2.ids[at..at + sampled.len()], &sampled[..]);
    let spliced: Vec<f32> = got[at..at + sampled.len()]
        .iter()
        .map(|v| v.unwrap())
        .collect();
    assert_eq!(spliced, lp);
    assert!(got[..at].iter().all(Option::is_none));
    assert!(got[at + sampled.len()..].iter().all(Option::is_none));
}

/// A multi-byte character sampled byte by byte stays one unit, so its
/// reply still aligns; a reply the client got part of is logged as that
/// part's ids.
#[test]
fn a_split_character_aligns_and_a_reply_cut_short_logs_its_part() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &json!([user("Café?")]), Some(false));
    let sampled = bytes("Un café, s'il vous plaît.");
    let d = decoded(&engine, sampled.clone());
    assert!(front.record(&engine, p1.origin, &d));
    let second = json!([user("Café?"), assistant(&d.text), user("Merci")]);
    let p2 = build(&front, &engine, &second, Some(false));
    assert_eq!(p2.spliced.spliced_tokens, sampled.len());
    // A reply the client got only part of — a stop string cut it, or the
    // run ended inside a character — is logged as the ids of that part.
    let mut cut = decoded(&engine, bytes("abc STOP def"));
    cut.text = "abc ".into();
    assert!(front.record(&engine, p1.origin, &cut));
    assert_eq!(front.reply("abc ").unwrap().ids, bytes("abc "));
    let mut held = decoded(&engine, [bytes("ok then"), vec![0xE2]].concat());
    held.text = "ok then".into();
    assert!(front.record(&engine, p1.origin, &held));
    assert_eq!(front.reply("ok then").unwrap().ids, bytes("ok then"));
    let mut neither = decoded(&engine, bytes("abc"));
    neither.text = "xyz".into();
    assert!(!front.record(&engine, p1.origin, &neither), "not ours");
    // A delivered text past the whole ids that does not begin the next
    // unit's text is not the run's either.
    let tr = TinyTokenizer::qwen4exp().id("<tool_response>").unwrap();
    let mut astray = decoded(&engine, [bytes("ab"), vec![tr]].concat());
    astray.text = "ab<x".into();
    assert!(
        !front.record(&engine, p1.origin, &astray),
        "not the run's text"
    );
}

/// The log is bounded: the oldest reply goes first, and its text no longer
/// matches.
#[test]
fn the_log_forgets_its_oldest_reply() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let p = build(&front, &engine, &json!([user("Hi")]), Some(false));
    for i in 0..=LOG_REPLIES {
        assert!(front.record(
            &engine,
            p.origin,
            &decoded(&engine, bytes(&format!("reply {i}")))
        ));
    }
    assert_eq!(front.logged(), LOG_REPLIES);
    assert!(front.reply("reply 0").is_none());
    assert!(front.reply(&format!("reply {LOG_REPLIES}")).is_some());
}

/// The checkpoints' own templates and tokenizers, when cached: the plain
/// equivalence and the budget's spans, as on the tiny templates.
#[test]
fn the_checkpoints_templates_build_the_plain_prompt_and_cut_the_same_spans() {
    let hub = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/huggingface/hub")
        });
    let mut ran = 0;
    for (repo, bos, caps) in [
        (
            "models--LiquidAI--LFM2.5-VL-3B",
            "<|startoftext|>",
            LFM2_CAPABILITIES,
        ),
        (
            "models--LiquidAI--LFM2.5-2.6B",
            "<|startoftext|>",
            LFM2_CAPABILITIES,
        ),
        ("models--Qwen--Qwen3.6-27B", "", QWEN3_5_CAPABILITIES),
        (
            "models--Qwen--Qwen3.8-Flash-Next",
            "",
            QWEN4EXP_CAPABILITIES,
        ),
    ] {
        let Some(dir) = std::fs::read_dir(hub.join(repo).join("snapshots"))
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
        else {
            eprintln!("skipped: {repo} is not cached");
            continue;
        };
        let (Ok(tok), Ok(src)) = (
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")),
            std::fs::read_to_string(dir.join("chat_template.jinja")),
        ) else {
            eprintln!("skipped: {repo} has no tokenizer.json + chat_template.jinja");
            continue;
        };
        let engine = Stub::new(tok, &src, bos, caps);
        let front = Front::new(&engine, SessionOptions::default());
        eprintln!("{repo}: {}", front.line());
        for conv in conversations() {
            for want in wants(&engine) {
                let p = build(&front, &engine, &conv, want);
                let kwargs = kw_front(&engine, &front, want);
                assert_eq!(
                    p.ids,
                    engine.reference(&conv, &kwargs, want),
                    "{repo} {conv} {want:?}"
                );
                let msgs = conv.as_array().unwrap();
                if let Some(Some(a)) = front.fit(&p, msgs, p.ids.len() - 1) {
                    let b = budgets::fit_messages(msgs, p.ids.len() - 1, p.ids.len(), |ms| {
                        Ok(engine.reference(&json!(ms), &kwargs, want))
                    })
                    .unwrap()
                    .unwrap();
                    assert_eq!(a.ids, b.ids, "{repo} {conv} {want:?}");
                }
            }
        }
        // A reply of ours, sampled as its text's canonical tokenization:
        // every id of it is spliced, and the prompt's text is the template's
        // rendering of the history as sent. Under `text`: with no end marker
        // this reply is open wherever the prompt opened the block (LFM2.5),
        // and the default gives it to the template as the reasoning it is
        // (`an_open_reply_sent_back_on_the_checkpoints_templates`).
        for want in wants(&engine) {
            let fresh = Front::new(
                &engine,
                SessionOptions {
                    incomplete: Incomplete::Text,
                    ..SessionOptions::default()
                },
            );
            let first = json!([user("What is the capital of France?")]);
            let p1 = build(&fresh, &engine, &first, want);
            let sampled = engine.enc("The capital of France is Paris.");
            let reply = decoded(&engine, sampled.clone());
            assert!(fresh.record(&engine, p1.origin, &reply), "{repo}");
            let second = json!([
                user("What is the capital of France?"),
                assistant(&reply.text),
                user("And Germany?")
            ]);
            let p2 = build(&fresh, &engine, &second, want);
            let kwargs = kw_front(&engine, &fresh, want);
            let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
            assert_eq!(
                decode(&p2.ids),
                decode(&engine.reference(&second, &kwargs, want)),
                "{repo} {want:?}"
            );
            assert_eq!(
                p2.spliced.spliced_tokens,
                sampled.len(),
                "{repo} {want:?}: {:?}",
                p2.spliced
            );
            eprintln!(
                "{repo} {want:?}: turn_tail {} -> reused if extended {}",
                p1.turn_tail,
                p2.ids
                    .iter()
                    .zip(&p1.ids)
                    .take_while(|(a, b)| a == b)
                    .count()
            );
        }
        ran += 1;
    }
    eprintln!("{ran} checkpoint templates checked");
}

/// Given back as the model wrote it, a reply's history is the template's own
/// rendering of the whole text the client got: stripped (opt-in), LFM2's
/// template splits the content at the end marker and trims it, and so does
/// the front end's rewrite; kept (the default, or `preserve_thinking`), the
/// rewrite is the model's sampled text and the log extends.
#[test]
fn a_reply_given_back_renders_as_the_templates_own_split() {
    for persist in [Some(false), None, Some(true)] {
        let engine = Stub::lfm2();
        let front = Front::new(
            &engine,
            SessionOptions {
                persist,
                ..SessionOptions::default()
            },
        );
        let p1 = build(&front, &engine, &json!([user("2+2?")]), None);
        assert!(
            p1.thinking.unwrap().on,
            "the tiny LFM2 template opens the block"
        );
        let end = TinyTokenizer::lfm2().id("</think>").unwrap();
        let mut sampled = bytes("Add them.");
        sampled.push(end);
        sampled.extend(bytes("\n\nFour."));
        let reply = decoded(&engine, sampled.clone());
        assert!(front.record(&engine, p1.origin, &reply));
        let second = json!([user("2+2?"), assistant(&reply.text), user("3+3?")]);
        let p2 = build(&front, &engine, &second, None);
        let mut kw = kw(&engine, None);
        kw.insert(
            PRESERVE_THINKING.into(),
            Value::Bool(persist.unwrap_or(true)),
        );
        let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
        match persist {
            Some(false) => assert_eq!(
                decode(&p2.ids),
                decode(&engine.reference(&second, &kw, None)),
                "stripped: the template's own split of the whole text"
            ),
            None | Some(true) => {
                let mut log = p1.ids.clone();
                log.extend_from_slice(&sampled);
                assert!(p2.ids.starts_with(&log), "kept: the log extends");
            }
        }
    }
}

/// A reply that closes its block twice (a thinking budget closed it, the
/// model went on reasoning and wrote the marker itself): stripped (opt-in),
/// a template keeps only what follows the last marker, and the prompt is that
/// render with the final content's sampled ids spliced. Kept (the default),
/// the reply goes
/// to the template as reasoning and content apart: a template that writes
/// the content whole beside a reasoning field (tiny LFM2, and Qwen3.6's rule
/// on the tiny Qwen3.5 template) extends the log; the tiny Qwen3.5 template
/// splits the content at the marker even then, and the prompt is that
/// render. Qwen3.6's rule is the one a stripped reply rewritten with its
/// reasoning apart would leak the second block through.
#[test]
fn a_reply_that_closes_its_block_twice_is_stripped_past_the_last_marker() {
    for (engine, end, want, whole) in [
        (
            Stub::qwen3_5(),
            TinyTokenizer::qwen3_5().id("</think>").unwrap(),
            Some(true),
            false,
        ),
        (
            Stub::qwen3_6_like(),
            TinyTokenizer::qwen3_5().id("</think>").unwrap(),
            Some(true),
            true,
        ),
        (
            Stub::lfm2(),
            TinyTokenizer::lfm2().id("</think>").unwrap(),
            None,
            true,
        ),
    ] {
        for persist in [Some(false), None, Some(true)] {
            let front = Front::new(
                &engine,
                SessionOptions {
                    persist,
                    ..SessionOptions::default()
                },
            );
            let p1 = build(&front, &engine, &json!([user("2+2?")]), want);
            let mut sampled = bytes("Add them.\n");
            sampled.push(end);
            sampled.extend(bytes("\n\nFour, I think\n"));
            sampled.push(end);
            sampled.extend(bytes("\n\nFour."));
            let reply = decoded(&engine, sampled.clone());
            assert!(front.record(&engine, p1.origin, &reply));
            let second = json!([user("2+2?"), assistant(&reply.text), user("3+3?")]);
            let p2 = build(&front, &engine, &second, want);
            assert_eq!(p2.spliced.matched, 1);
            let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
            let mut kw = kw(&engine, want);
            kw.insert(
                PRESERVE_THINKING.into(),
                Value::Bool(persist.unwrap_or(true)),
            );
            match persist {
                Some(false) => {
                    let text = decode(&p2.ids);
                    assert_eq!(
                        text,
                        decode(&engine.reference(&second, &kw, want)),
                        "stripped: the template's own split of the whole text"
                    );
                    assert!(!text.contains("I think"), "{text:?}");
                    assert_eq!(
                        p2.spliced.spliced_tokens,
                        bytes("Four.").len(),
                        "the final content's ids, spliced: {text:?}"
                    );
                }
                None | Some(true) if whole => {
                    let mut log = p1.ids.clone();
                    log.extend_from_slice(&sampled);
                    assert!(p2.ids.starts_with(&log), "kept: the log extends");
                }
                None | Some(true) => {
                    let logged = front.reply(&reply.text).unwrap();
                    let mut apart = assistant(&logged.content_text);
                    for f in REASONING_FIELDS {
                        apart[f] = json!(logged.reasoning_text);
                    }
                    let given = json!([user("2+2?"), apart, user("3+3?")]);
                    assert_eq!(
                        decode(&p2.ids),
                        decode(&engine.reference(&given, &kw, want)),
                        "kept: the template's render of the reply apart"
                    );
                }
            }
        }
    }
}

/// `session.keep_logprobs` is refused at load by the body the process
/// decodes through, from the one refusal table: the K=1 body serves it on
/// either readback arm (a logprobs run takes the blocking one); the
/// speculative body and the batch scheduler refuse it, each by name.
#[test]
fn the_keep_logprobs_refusal_follows_the_decode_body() {
    use lloom::nn::Body;
    assert_eq!(super::logprobs_refused_on(Body::Step { ring: true }), None);
    assert_eq!(super::logprobs_refused_on(Body::Step { ring: false }), None);
    for (body, says) in [
        (Body::Window, "this process decodes through the speculative body (--dspark, --mtp): the window reads back"),
        (Body::BatchedStep, "this process decodes through the batch scheduler (--batch N): a batched body does not keep per-token logprobs"),
        (Body::BatchedWindow, "this process decodes through the batch scheduler (--batch N): a batched body does not keep per-token logprobs"),
    ] {
        let why = super::logprobs_refused_on(body).unwrap_or_default();
        assert!(why.starts_with(says), "{body:?}: {why}");
    }
}

/// The tiny Qwen4-exp tokenizer's end marker.
fn q4_end() -> u32 {
    TinyTokenizer::qwen4exp().id("</think>").unwrap()
}

/// A reply whose prompt left the block open, stopped by `max_tokens` before,
/// at and after its closing marker: the log keeps what of it is reasoning as
/// reasoning whether or not the block closed, every sampled id, and no
/// marker the model did not sample.
#[test]
fn a_reply_stopped_inside_its_reasoning_keeps_it_as_reasoning() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &json!([user("2+2?")]), Some(true));
    assert!(p1.thinking.unwrap().on, "the prompt leaves the block open");
    let end = q4_end();
    for (case, ids, reasoning, content) in [
        ("before", bytes("Add them"), "Add them", ""),
        (
            "at",
            [bytes("Add them"), vec![end]].concat(),
            "Add them",
            "",
        ),
        (
            "after",
            [bytes("Add them"), vec![end], bytes("\n\nFour")].concat(),
            "Add them",
            "\n\nFour",
        ),
    ] {
        let mut d = decoded(&engine, ids.clone());
        d.stop = Stop::Length;
        assert!(front.record(&engine, p1.origin, &d), "{case}");
        let r = front.reply(&d.text).unwrap();
        assert_eq!(r.ids, ids, "{case}: every sampled id");
        assert_eq!(
            (r.reasoning_text.as_str(), r.content_text.as_str()),
            (reasoning, content),
            "{case}: the reasoning, and the content past the marker"
        );
    }
}

/// A stop string that begins inside a token (`<tool_response>` is one id;
/// the stop is `resp`, withheld from its first byte): the delivered text
/// ends inside that token's text. The log keeps the whole ids before it as
/// sampled and the text past them as text — the accepted frontier — and the
/// next turn splices those ids and encodes the rest, the render's own.
/// Stops that end on a character's units, or between ids, keep every unit
/// before them.
#[test]
fn a_stop_inside_a_token_keeps_the_whole_ids_before_it() {
    let engine = Stub::qwen4exp();
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &json!([user("Tools?")]), Some(false));
    let tr = TinyTokenizer::qwen4exp().id("<tool_response>").unwrap();
    let mut d = decoded(&engine, [bytes("ab"), vec![tr], bytes("cd")].concat());
    d.stop = Stop::StopString;
    d.text = "ab<tool_".into();
    assert_eq!(
        front.record_frontier(&engine, p1.origin, &d),
        Frontier {
            generated: 5,
            accepted: 2,
            partial: "<tool_".len(),
            delivered: "ab<tool_".len(),
            logged: true,
        },
        "a reply whose delivered text ends inside a token is logged"
    );
    let r = front.reply("ab<tool_").unwrap();
    assert_eq!((r.partial, r.generated), ("<tool_".len(), 5));
    assert_eq!(
        r.ids,
        bytes("ab"),
        "the whole ids before the cut, as sampled"
    );
    let second = json!([user("Tools?"), assistant("ab<tool_"), user("More?")]);
    let p2 = build(&front, &engine, &second, Some(false));
    assert_eq!(p2.spliced.matched, 1, "{:?}", p2.spliced);
    assert_eq!(p2.spliced.spliced_tokens, 2, "{:?}", p2.spliced);
    let kwargs = kw(&engine, Some(false));
    let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
    assert_eq!(
        decode(&p2.ids),
        decode(&engine.reference(&second, &kwargs, Some(false))),
        "the template's render of the delivered text"
    );

    // A character's units stay whole: a stop at `é!` keeps `caf`, one
    // across ids (`fé`) keeps `ca`.
    for (stop_at, kept) in [("caf", "caf"), ("ca", "ca")] {
        let mut d = decoded(&engine, bytes("café!"));
        d.stop = Stop::StopString;
        d.text = stop_at.into();
        assert!(front.record(&engine, p1.origin, &d), "{stop_at}");
        assert_eq!(front.reply(stop_at).unwrap().ids, bytes(kept), "{stop_at}");
    }
}

/// The checkpoints' own templates and tokenizers (when cached): a reply cut
/// inside the reasoning block its prompt opened, sent back whole, then the
/// next turn's prompt per `session.incomplete_reasoning` arm, persisted and
/// stripped. Printed: where the next prompt leaves the log (the previous
/// prompt and the reply) against the previous prompt's end, and where the
/// turn checkpoint stood. Asserted:
/// - `text` is the plain render of the history as sent (the default's
///   prompt is unchanged);
/// - `reasoning` leaves the log at or past the checkpoint, kept never before
///   the previous prompt's end, and none of the end markers the template
///   adds after the reasoning is a sampled id;
/// - `recover`'s checkpoint is at or before the `text` prompt's divergence.
#[test]
fn an_open_reply_on_the_checkpoints_templates_per_arm() {
    let hub = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/huggingface/hub")
        });
    let mut ran = 0;
    for (repo, bos, caps) in [
        (
            "models--LiquidAI--LFM2.5-2.6B",
            "<|startoftext|>",
            LFM2_CAPABILITIES,
        ),
        ("models--Qwen--Qwen3.6-27B", "", QWEN3_5_CAPABILITIES),
        ("models--Qwen--Qwen3.6-35B-A3B", "", QWEN3_5_CAPABILITIES),
        (
            "models--Qwen--Qwen3.8-Flash-Next",
            "",
            QWEN4EXP_CAPABILITIES,
        ),
    ] {
        let Some(dir) = std::fs::read_dir(hub.join(repo).join("snapshots"))
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
        else {
            eprintln!("skipped: {repo} is not cached");
            continue;
        };
        let (Ok(tok), Ok(src)) = (
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")),
            std::fs::read_to_string(dir.join("chat_template.jinja")),
        ) else {
            eprintln!("skipped: {repo} has no tokenizer.json + chat_template.jinja");
            continue;
        };
        let engine = Stub::new(tok, &src, bos, caps);
        let want = Some(true);
        for persist in [Some(true), Some(false)] {
            let mut text_div = None;
            for arm in [Incomplete::Text, Incomplete::Reasoning, Incomplete::Recover] {
                let front = Front::new(
                    &engine,
                    SessionOptions {
                        persist,
                        incomplete: arm,
                        ..SessionOptions::default()
                    },
                );
                if front.persist_unsupported().is_some() {
                    continue;
                }
                let first = json!([user("What is the capital of France?")]);
                let p1 = build(&front, &engine, &first, want);
                if !p1.thinking.is_some_and(|t| t.on) {
                    eprintln!("{repo}: thinking does not open a block; skipped");
                    break;
                }
                let sampled = engine.enc("The user asks for the capital of France. That is");
                let mut d = decoded(&engine, sampled.clone());
                d.stop = Stop::Length;
                assert!(front.record(&engine, p1.origin, &d), "{repo}");
                let second = json!([
                    user("What is the capital of France?"),
                    assistant(&d.text),
                    user("And Germany?")
                ]);
                let p2 = build(&front, &engine, &second, want);
                let log: Vec<u32> = p1.ids.iter().chain(&sampled).copied().collect();
                let kept = p2.ids.iter().zip(&log).take_while(|(a, b)| a == b).count();
                let ckpt = p1.ids.len() - p1.turn_tail;
                eprintln!(
                    "OPEN-REPLY {repo} persist={persist:?} arm={}: prompt {} reply {} -> next prompt \
                     leaves the log at {kept} ({:+} from the prompt's end); checkpoint at {ckpt} \
                     (tail {}); open {} apart {} spliced {}",
                    arm.name(),
                    p1.ids.len(),
                    sampled.len(),
                    kept as i64 - p1.ids.len() as i64,
                    p1.turn_tail,
                    p2.spliced.open,
                    p2.spliced.open_apart,
                    p2.spliced.spliced_tokens,
                );
                assert_eq!(p2.spliced.open, 1, "{repo} {}", arm.name());
                match arm {
                    Incomplete::Text => {
                        let mut kw = kw(&engine, want);
                        kw.insert(PRESERVE_THINKING.into(), json!(persist.unwrap()));
                        let decode =
                            |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
                        assert_eq!(
                            decode(&p2.ids),
                            decode(&engine.reference(&second, &kw, want)),
                            "{repo}: the default's prompt is the history as sent"
                        );
                        text_div = Some(kept);
                    }
                    Incomplete::Reasoning => {
                        assert_eq!(p2.spliced.open_apart, 1, "{repo}");
                        assert!(
                            ckpt <= kept,
                            "{repo} persist={persist:?}: the next prompt leaves the log at {kept}, \
                             before the checkpoint at {ckpt}"
                        );
                        if persist == Some(true) {
                            assert!(
                                kept >= p1.ids.len(),
                                "{repo}: kept, the open reply leaves the log before the previous \
                                 prompt's end ({kept} < {})",
                                p1.ids.len()
                            );
                        }
                        let in_sampled = |i: usize| p2.sampled.iter().any(|r| r.contains(&i));
                        assert!(
                            (kept..p2.ids.len())
                                .filter(|&i| p2.ids[i] == engine.text.think_tags().unwrap().end)
                                .all(|i| !in_sampled(i)),
                            "{repo}: an end marker after the reasoning is the template's"
                        );
                    }
                    Incomplete::Recover => {
                        let div = text_div.expect("text ran first");
                        assert!(
                            ckpt <= div,
                            "{repo} persist={persist:?}: the checkpoint {ckpt} is past the \
                             divergence {div}"
                        );
                    }
                }
            }
        }
        ran += 1;
    }
    eprintln!("{ran} checkpoint templates checked");
}

/// An open reply under `Incomplete::Reasoning` is given apart only when the
/// client sent it back whole with no reasoning field of its own: a field
/// set to `null`, `""` or text, an edited reply, and a text two distinct
/// replies of the conversation carry each go as sent; so does every case
/// under the default `text`.
#[test]
fn an_open_reply_is_given_apart_only_as_it_came_back() {
    let engine = Stub::qwen4exp();
    let q = || json!([user("2+2?")]);
    let next = |a: Value| json!([user("2+2?"), a, user("3+3?")]);
    let want = Some(true);
    let as_sent = |front: &Front, history: &Value| {
        let p = build(front, &engine, history, want);
        let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
        let reference = engine.reference(history, &kw(&engine, want), want);
        (p.spliced, decode(&p.ids) == decode(&reference))
    };
    for arm in [Incomplete::Text, Incomplete::Reasoning] {
        let front = Front::new(
            &engine,
            SessionOptions {
                incomplete: arm,
                ..SessionOptions::default()
            },
        );
        let p1 = build(&front, &engine, &q(), want);
        let mut d = decoded(&engine, bytes("Add them"));
        d.stop = Stop::Length;
        assert!(front.record(&engine, p1.origin, &d));

        // Whole, no field: apart under `reasoning`, as sent under `text`.
        let (sp, plain) = as_sent(&front, &next(assistant("Add them")));
        assert_eq!((sp.matched, sp.open), (1, 1), "{}", arm.name());
        assert_eq!(sp.open_apart, usize::from(arm == Incomplete::Reasoning));
        assert_eq!(plain, arm == Incomplete::Text, "{}", arm.name());

        for (case, field) in [
            ("null", Value::Null),
            ("empty", json!("")),
            ("text", json!("Add them")),
        ] {
            let mut a = assistant("Add them");
            a["reasoning_content"] = field;
            let (sp, plain) = as_sent(&front, &next(a));
            assert_eq!(sp.open_apart, 0, "{} {case}", arm.name());
            assert!(plain, "{} {case}: as sent", arm.name());
        }
        let (sp, plain) = as_sent(&front, &next(assistant("Add them, edited")));
        assert_eq!(sp.matched, 0, "{}: edited", arm.name());
        assert!(plain, "{}: edited", arm.name());

        // A regeneration of one text from other ids (an added token's text
        // sampled as its bytes): ambiguous while both are logged.
        let tr = TinyTokenizer::qwen4exp().id("<tool_response>").unwrap();
        for ids in [
            [bytes("Add "), vec![tr]].concat(),
            bytes("Add <tool_response>"),
        ] {
            let mut d = decoded(&engine, ids);
            d.stop = Stop::Length;
            assert!(front.record(&engine, p1.origin, &d));
        }
        let (sp, plain) = as_sent(&front, &next(assistant("Add <tool_response>")));
        assert_eq!(sp.ambiguous, 1, "{}: two replies, one text", arm.name());
        assert!(plain, "{}: ambiguous", arm.name());
    }
}

/// A reply cut inside its reasoning, given back by a client that sends the
/// reasoning apart — the field set to the text it got, the content empty,
/// null or absent: the shape the template reads, and the one
/// `Incomplete::Reasoning` rewrites a whole reply into. It is the logged
/// reply, under every arm: its span takes the sampled ids (here not the
/// text's canonical tokenization: `<tool_response>` is one added token), so
/// the log is a prefix of the next prompt, and the rendering is the history
/// as sent. A field the client edited, content of its own beside the field,
/// another conversation's reply and two distinct replies of one text each
/// go as sent, with nothing taken.
#[test]
fn an_open_reply_sent_back_apart_takes_its_sampled_ids() {
    let engine = Stub::qwen4exp();
    let want = Some(true);
    let q = json!([user("2+2?")]);
    let next = |a: Value| json!([user("2+2?"), a, user("3+3?")]);
    let apart = |reasoning: &str, content: Option<Value>| {
        let mut a = json!({ "role": "assistant", "reasoning_content": reasoning });
        if let Some(c) = content {
            a["content"] = c;
        }
        a
    };
    let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
    let sampled = bytes("Add <tool_response> them");
    for arm in [Incomplete::Text, Incomplete::Reasoning, Incomplete::Recover] {
        let front = Front::new(
            &engine,
            SessionOptions {
                incomplete: arm,
                ..SessionOptions::default()
            },
        );
        let p1 = build(&front, &engine, &q, want);
        let mut d = decoded(&engine, sampled.clone());
        d.stop = Stop::Length;
        assert_ne!(
            engine.enc(&d.text),
            sampled,
            "the precondition: not canonical"
        );
        assert!(front.record(&engine, p1.origin, &d));
        let log: Vec<u32> = p1.ids.iter().chain(&sampled).copied().collect();

        for (case, content) in [
            ("empty", Some(json!(""))),
            ("null", Some(Value::Null)),
            ("absent", None),
        ] {
            let history = next(apart(&d.text, content));
            let p2 = build(&front, &engine, &history, want);
            assert_eq!(
                (p2.spliced.matched, p2.spliced.open),
                (1, 1),
                "{} {case}: the reply is ours: {:?}",
                arm.name(),
                p2.spliced
            );
            assert!(
                p2.ids.starts_with(&log),
                "{} {case}: the log is not a prefix of the next prompt:\n log  {log:?}\n next {:?}",
                arm.name(),
                &p2.ids[..log.len().min(p2.ids.len())]
            );
            assert_eq!(
                p2.spliced.spliced_tokens,
                sampled.len(),
                "{} {case}",
                arm.name()
            );
            assert_eq!(
                decode(&p2.ids),
                decode(&engine.reference(&history, &kw(&engine, want), want)),
                "{} {case}: the history as sent",
                arm.name()
            );
        }

        // Not the reply as we gave it: tokenized as sent.
        let mut two_fields = apart(&d.text, Some(json!("")));
        two_fields["thinking"] = json!("Add them otherwise");
        for (case, a) in [
            ("edited", apart("Add <tool_response> THEM", Some(json!("")))),
            ("own content", apart(&d.text, Some(json!("Four.")))),
            ("two fields that differ", two_fields),
        ] {
            let history = next(a);
            let p2 = build(&front, &engine, &history, want);
            assert_eq!(p2.spliced.matched, 0, "{} {case}", arm.name());
            assert_eq!(
                p2.ids,
                engine.reference(&history, &kw(&engine, want), want),
                "{} {case}",
                arm.name()
            );
        }
        let elsewhere = json!([user("4+4?"), apart(&d.text, Some(json!(""))), user("3+3?")]);
        let p = build(&front, &engine, &elsewhere, want);
        assert_eq!(
            (p.spliced.matched, p.spliced.unverified),
            (0, 1),
            "{}: another conversation",
            arm.name()
        );
    }

    // A reply with no reasoning block (its prompt closed the block) is not
    // reasoning, whatever field a client puts its text in.
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &q, Some(false));
    assert!(front.record(&engine, p1.origin, &decoded(&engine, sampled.clone())));
    let history = next(apart("Add <tool_response> them", Some(json!(""))));
    let p = build(&front, &engine, &history, want);
    assert_eq!(
        p.spliced.matched, 0,
        "a reply with no block, given as reasoning"
    );
    assert_eq!(p.ids, engine.reference(&history, &kw(&engine, want), want));

    // Two distinct replies of one conversation with that text: ambiguous.
    let front = Front::new(&engine, SessionOptions::default());
    let p1 = build(&front, &engine, &q, want);
    let tr = TinyTokenizer::qwen4exp().id("<tool_response>").unwrap();
    for ids in [
        sampled.clone(),
        [bytes("Add "), vec![tr], bytes(" them")].concat(),
    ] {
        let mut d = decoded(&engine, ids);
        d.stop = Stop::Length;
        assert!(front.record(&engine, p1.origin, &d));
    }
    let p = build(
        &front,
        &engine,
        &next(apart("Add <tool_response> them", Some(json!("")))),
        want,
    );
    assert_eq!((p.spliced.matched, p.spliced.ambiguous), (0, 1));
}

/// Where a reply cut inside its reasoning lands in the next turn's history,
/// on the checkpoints' own templates and tokenizers (when cached), per
/// `session.incomplete_reasoning` arm, persisted and stripped, for the two
/// shapes a client sends it back in: `echo` (the content it got: every
/// OpenAI client that keeps `content`) and `apart` (`reasoning_content` set
/// to that text, the content empty). Printed: the assistant message's span
/// as the next prompt renders it, where the unfinished thought stands in it
/// (`answer`: where the template puts a final answer; `reasoning`: inside
/// the reasoning block; `absent`), and where the next prompt leaves the log.
/// Asserted:
/// - `echo` under `text` and `recover`: the thought stands as the answer
///   (the default's history reads the unfinished reasoning as what the
///   model told the user);
/// - `echo` under `reasoning`, and `apart` under every arm: inside the
///   block where reasoning is kept, absent where it is stripped (the
///   message is then empty);
/// - `apart` is the logged reply (matched), and kept, the next prompt
///   leaves the log no earlier than the previous prompt's end.
#[test]
fn an_open_reply_sent_back_on_the_checkpoints_templates() {
    let hub = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/huggingface/hub")
        });
    let mut ran = 0;
    for (repo, bos, caps) in [
        (
            "models--LiquidAI--LFM2.5-2.6B",
            "<|startoftext|>",
            LFM2_CAPABILITIES,
        ),
        ("models--Qwen--Qwen3.6-27B", "", QWEN3_5_CAPABILITIES),
        ("models--Qwen--Qwen3.6-35B-A3B", "", QWEN3_5_CAPABILITIES),
        (
            "models--Qwen--Qwen3.8-Flash-Next",
            "",
            QWEN4EXP_CAPABILITIES,
        ),
    ] {
        let Some(dir) = std::fs::read_dir(hub.join(repo).join("snapshots"))
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
        else {
            eprintln!("skipped: {repo} is not cached");
            continue;
        };
        let (Ok(tok), Ok(src)) = (
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")),
            std::fs::read_to_string(dir.join("chat_template.jinja")),
        ) else {
            eprintln!("skipped: {repo} has no tokenizer.json + chat_template.jinja");
            continue;
        };
        let engine = Stub::new(tok, &src, bos, caps);
        let tags = engine.text.think_tags().cloned().unwrap();
        let end = engine.token_str(tags.end);
        let want = Some(true);
        let thought = "The user asks for the capital of France. That is";
        for persist in [Some(true), Some(false)] {
            for arm in [Incomplete::Text, Incomplete::Reasoning, Incomplete::Recover] {
                let front = Front::new(
                    &engine,
                    SessionOptions {
                        persist,
                        incomplete: arm,
                        ..SessionOptions::default()
                    },
                );
                if front.persist_unsupported().is_some() {
                    continue;
                }
                let first = json!([user("What is the capital of France?")]);
                let p1 = build(&front, &engine, &first, want);
                assert!(p1.thinking.is_some_and(|t| t.on), "{repo}");
                let sampled = engine.enc(thought);
                let mut d = decoded(&engine, sampled.clone());
                d.stop = Stop::Length;
                assert!(front.record(&engine, p1.origin, &d), "{repo}");
                let log: Vec<u32> = p1.ids.iter().chain(&sampled).copied().collect();
                for shape in ["echo", "apart"] {
                    let reply = match shape {
                        "echo" => assistant(&d.text),
                        _ => json!({ "role": "assistant", "content": "",
                                     "reasoning_content": d.text }),
                    };
                    let second = json!([
                        user("What is the capital of France?"),
                        reply,
                        user("And Germany?")
                    ]);
                    let p2 = build(&front, &engine, &second, want);
                    let b = p2.blocks.as_ref().expect("the spans");
                    let span = engine
                        .text
                        .tokenizer
                        .decode(&p2.ids[b.starts[1].unwrap()..b.starts[2].unwrap()], false)
                        .unwrap();
                    let stands = match span.find(thought) {
                        None => "absent",
                        Some(at) if span[at..].contains(end.as_str()) => "reasoning",
                        Some(_) => "answer",
                    };
                    let kept = p2.ids.iter().zip(&log).take_while(|(a, b)| a == b).count();
                    eprintln!(
                        "OPEN-SENT-BACK {repo} persist={persist:?} arm={} shape={shape}: the \
                         thought stands as {stands}; matched {} spliced {}; next prompt leaves \
                         the log at {:+} from the prompt's end; span {span:?}",
                        arm.name(),
                        p2.spliced.matched,
                        p2.spliced.spliced_tokens,
                        kept as i64 - p1.ids.len() as i64,
                    );
                    let kept_reasoning = persist == Some(true);
                    let want_stands = match (shape, arm) {
                        ("echo", Incomplete::Text | Incomplete::Recover) => "answer",
                        _ if kept_reasoning => "reasoning",
                        _ => "absent",
                    };
                    assert_eq!(
                        stands,
                        want_stands,
                        "{repo} persist={persist:?} {} {shape}: {span:?}",
                        arm.name()
                    );
                    if shape == "apart" {
                        assert_eq!(
                            p2.spliced.matched,
                            1,
                            "{repo} persist={persist:?} {}: sent back apart, the reply is ours",
                            arm.name()
                        );
                        if kept_reasoning {
                            assert!(
                                kept >= p1.ids.len(),
                                "{repo} {}: kept, the next prompt leaves the log at {kept}, \
                                 before the previous prompt's end {}",
                                arm.name(),
                                p1.ids.len()
                            );
                        }
                    }
                }
            }
        }
        ran += 1;
    }
    eprintln!("{ran} checkpoint templates checked");
}

/// An open reply the log no longer holds (a restarted process, an evicted
/// entry) goes as sent, under `Incomplete::Reasoning` too. Echoed back as
/// the content the client got, the unfinished thought is then the message's
/// content where, logged, it was its reasoning. Sent back apart, it is
/// reasoning either way: what a client carries does not depend on what the
/// server still holds.
#[test]
fn an_open_reply_the_log_no_longer_holds_reads_as_the_client_sent_it() {
    let engine = Stub::qwen4exp();
    let want = Some(true);
    let opts = SessionOptions {
        incomplete: Incomplete::Reasoning,
        ..SessionOptions::default()
    };
    let q = json!([user("2+2?")]);
    let next = |a: Value| json!([user("2+2?"), a, user("3+3?")]);
    let echo = next(assistant("Add them"));
    let apart =
        next(json!({ "role": "assistant", "content": "", "reasoning_content": "Add them" }));
    let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
    let reference = |h: &Value| decode(&engine.reference(h, &kw(&engine, want), want));
    let rendered = |f: &Front, h: &Value| decode(&build(f, &engine, h, want).ids);

    let front = Front::new(&engine, opts);
    let p1 = build(&front, &engine, &q, want);
    let mut d = decoded(&engine, bytes("Add them"));
    d.stop = Stop::Length;
    assert!(front.record(&engine, p1.origin, &d));
    let as_reasoning = reference(&apart);
    assert_ne!(as_reasoning, reference(&echo), "the two renderings differ");
    assert_eq!(rendered(&front, &echo), as_reasoning, "logged: reasoning");
    assert_eq!(rendered(&front, &apart), as_reasoning, "logged, apart");

    let restarted = Front::new(&engine, opts);
    for i in 0..LOG_REPLIES {
        assert!(front.record(
            &engine,
            p1.origin,
            &decoded(&engine, bytes(&format!("other {i}")))
        ));
    }
    for (case, f) in [("restarted", &restarted), ("evicted", &front)] {
        assert_eq!(
            build(f, &engine, &echo, want).spliced.matched,
            0,
            "{case}: not held"
        );
        assert_eq!(
            rendered(f, &echo),
            reference(&echo),
            "{case}: echoed, the thought is content"
        );
        assert_eq!(
            rendered(f, &apart),
            as_reasoning,
            "{case}: apart, still reasoning"
        );
    }
}

/// A stop string that begins inside a token inside the reasoning
/// (`<tool_response>` is one id; the stop is `resp`): the block stays open,
/// the whole ids before the cut are the accepted frontier, and the delivered
/// bytes past them are reasoning, never content. Under
/// `Incomplete::Reasoning` the next turn splices those ids inside the block
/// and encodes the rest; under `text` the delivered text is the content.
#[test]
fn a_stop_inside_a_token_inside_the_reasoning_keeps_the_block_open() {
    let engine = Stub::qwen4exp();
    let want = Some(true);
    let q = json!([user("Tools?")]);
    let tr = TinyTokenizer::qwen4exp().id("<tool_response>").unwrap();
    for arm in [Incomplete::Text, Incomplete::Reasoning] {
        let front = Front::new(
            &engine,
            SessionOptions {
                incomplete: arm,
                ..SessionOptions::default()
            },
        );
        let p1 = build(&front, &engine, &q, want);
        let mut d = decoded(&engine, [bytes("ab"), vec![tr], bytes("cd")].concat());
        d.stop = Stop::StopString;
        d.text = "ab<tool_".into();
        let f = front.record_frontier(&engine, p1.origin, &d);
        assert_eq!(
            (f.generated, f.accepted, f.partial, f.logged),
            (5, 2, "<tool_".len(), true)
        );
        let r = front.reply("ab<tool_").unwrap();
        assert_eq!(r.block, Block::Open { interior: 0..2 }, "{}", arm.name());
        assert_eq!(
            (r.reasoning_text.as_str(), r.content_text.as_str()),
            ("ab<tool_", "")
        );
        let echo = json!([user("Tools?"), assistant("ab<tool_"), user("More?")]);
        let p2 = build(&front, &engine, &echo, want);
        let mut log = p1.ids.clone();
        log.extend(bytes("ab"));
        assert_eq!(p2.spliced.spliced_tokens, 2, "{}", arm.name());
        let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
        let shown = match arm {
            Incomplete::Reasoning => json!([user("Tools?"),
                { "role": "assistant", "content": "", "reasoning_content": "ab<tool_" },
                user("More?")]),
            _ => echo.clone(),
        };
        assert_eq!(
            decode(&p2.ids),
            decode(&engine.reference(&shown, &kw(&engine, want), want)),
            "{}",
            arm.name()
        );
        if arm == Incomplete::Reasoning {
            assert!(p2.ids.starts_with(&log), "the accepted ids, in the block");
        }
    }
}

/// Decision 46(a): by default (`session.incomplete_reasoning` unset) an open
/// reply sent back as the content the client got goes to the template as the
/// open reasoning it is, and the loaded line says so.
#[test]
fn by_default_an_open_reply_goes_to_the_template_as_reasoning() {
    let engine = Stub::qwen4exp();
    let want = Some(true);
    let front = Front::new(
        &engine,
        SessionOptions::of(&crate::options::SessionConfig::default()),
    );
    assert!(
        front.line().contains("incomplete_reasoning=reasoning"),
        "{}",
        front.line()
    );
    let p1 = build(&front, &engine, &json!([user("2+2?")]), want);
    let mut d = decoded(&engine, bytes("Add them"));
    d.stop = Stop::Length;
    assert!(front.record(&engine, p1.origin, &d));
    let echo = json!([user("2+2?"), assistant("Add them"), user("3+3?")]);
    let apart = json!([user("2+2?"),
        { "role": "assistant", "content": "", "reasoning_content": "Add them" },
        user("3+3?")]);
    let p2 = build(&front, &engine, &echo, want);
    assert_eq!((p2.spliced.open, p2.spliced.open_apart), (1, 1));
    let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
    assert_eq!(
        decode(&p2.ids),
        decode(&engine.reference(&apart, &kw(&engine, want), want))
    );
}

/// The supported checkpoints' own templates (when cached), as `lloom chat`
/// and `lloom serve` reach them. The GGUF checkpoints render one of these:
/// a Qwen GGUF its base model's snapshot's template (`qwen3_5.rs`
/// `resolve_text`, `qwen4exp.rs`), an LFM2 GGUF its embedded one, which is
/// the snapshot's byte for byte (`measurements/issue-946.md` §25).
const AUDITED: [(&str, &str, Capabilities); 7] = [
    (
        "models--LiquidAI--LFM2.5-2.6B",
        "<|startoftext|>",
        LFM2_CAPABILITIES,
    ),
    (
        "models--LiquidAI--LFM2.5-VL-3B",
        "<|startoftext|>",
        LFM2_CAPABILITIES,
    ),
    (
        "models--LiquidAI--d1-3B",
        "<|startoftext|>",
        LFM2_CAPABILITIES,
    ),
    ("models--Qwen--Qwen3.6-27B", "", QWEN3_5_CAPABILITIES),
    ("models--Qwen--Qwen3.6-35B-A3B", "", QWEN3_5_CAPABILITIES),
    ("models--Qwen--Qwen3.8-27B", "", QWEN3_5_CAPABILITIES),
    (
        "models--Qwen--Qwen3.8-Flash-Next",
        "",
        QWEN4EXP_CAPABILITIES,
    ),
];

/// A checkpoint's own template and tokenizer from the Hugging Face cache.
fn cached(repo: &str, bos: &str, caps: Capabilities) -> Option<Stub> {
    let hub = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/huggingface/hub")
        });
    let dir = std::fs::read_dir(hub.join(repo).join("snapshots"))
        .ok()?
        .next()?
        .ok()?
        .path();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).ok()?;
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).ok()?;
    Some(Stub::new(tok, &src, bos, caps))
}

/// The audit of the supported checkpoints' templates against what lloom
/// sets (#946 run 5, the maintainer's direction of 2026-10-15: past
/// reasoning persisted by default, stripping opt-in). One `AUDIT` line per
/// template for what the template reads and does, one per lloom option for
/// what reaches it, and one per (thinking, persistence, reply) for where
/// the next turn's prompt leaves the session's log. Asserted:
/// - unset or `true`, the history keeps what the model generated: a reply in
///   the template's own format, an answer with thinking off, and a reply cut
///   inside its reasoning (decision 46(a)) each extend the log;
/// - `false` (stripping, opt-in) is still served where the template has the
///   switch;
/// - under every option the next prompt leaves the log no earlier than the
///   turn checkpoint.
#[test]
fn the_checkpoints_templates_persist_what_the_model_generated() {
    let user1 = user("What is the capital of France?");
    let user2 = user("And Germany?");
    let mut ran = 0;
    let mut failures: Vec<String> = Vec::new();
    for (repo, bos, caps) in AUDITED {
        let Some(engine) = cached(repo, bos, caps) else {
            eprintln!("skipped: {repo} has no tokenizer.json + chat_template.jinja");
            continue;
        };
        let name = repo.rsplit("--").next().unwrap();
        let t = engine.text.template.as_ref().unwrap();
        let reads: Vec<&str> = t
            .variables()
            .map(|v| v.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let tags = engine.text.think_tags().cloned();
        let render = |m: &Value, kw: &Map<String, Value>, gen: bool| {
            t.render_with(m, bos, kw, gen).map_err(|e| format!("{e:#}"))
        };
        let with = |k: &str, v: Value| {
            let mut kw = Map::new();
            kw.insert(k.into(), v);
            kw
        };
        let one = json!([user1]);
        let bare = render(&one, &Map::new(), false).unwrap();
        let gens: Vec<String> = [None, Some(true), Some(false)]
            .iter()
            .map(|e| {
                let kw = e.map_or(Map::new(), |b| with(ENABLE_THINKING, json!(b)));
                render(&one, &kw, true).map_or_else(
                    |e| format!("refused ({e})"),
                    |g| format!("{:?}", g.strip_prefix(&bare).unwrap_or(&g)),
                )
            })
            .collect();
        let mut reply = json!({ "role": "assistant", "content": "ANSWER" });
        for f in REASONING_FIELDS {
            reply[f] = json!("REASONING");
        }
        let kept_where = |kw: &Map<String, Value>| {
            let before = render(&json!([user1, reply, user2]), kw, false).unwrap();
            let after = render(&json!([user1, reply]), kw, false).unwrap();
            match (before.contains("REASONING"), after.contains("REASONING")) {
                (true, _) => "kept",
                (false, true) => "dropped before the last user turn",
                (false, false) => "dropped",
            }
        };
        let split = render(
            &json!([user1, assistant("REASONING</think>\n\nANSWER"), user2]),
            &with(PRESERVE_THINKING, json!(true)),
            false,
        )
        .map(|s| {
            let at = s.find("REASONING").unwrap_or(0);
            if s[..at].ends_with("<think>") || s[..at].ends_with("<think>\n") {
                "yes: a past reply's content is split at the end marker"
            } else {
                "no: the content is rendered as sent"
            }
        })
        .unwrap_or("-");
        eprintln!(
            "AUDIT {name} template sha {} reads {reads:?}",
            t.digest()[..6]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        eprintln!(
            "AUDIT {name} generation prompt: enable_thinking unset {} | true {} | false {}",
            gens[0], gens[1], gens[2]
        );
        eprintln!(
            "AUDIT {name} past reasoning: unset {} | preserve_thinking=true {} | false {}; \
             re-splits content: {split}; tags {:?}",
            kept_where(&Map::new()),
            kept_where(&with(PRESERVE_THINKING, json!(true))),
            kept_where(&with(PRESERVE_THINKING, json!(false))),
            tags.as_ref()
                .map(|t| (engine.token_str(t.start), engine.token_str(t.end))),
        );

        // What lloom sets, per option.
        let base = Front::new(&engine, SessionOptions::default());
        let c = base.constants().unwrap();
        for want in [None, Some(true), Some(false)] {
            let r = base.prompt(
                &engine,
                Request {
                    messages: &one,
                    kwargs: kw(&engine, want),
                    want,
                    add_bos: true,
                    expand: None,
                    history: None,
                },
            );
            let effort = c.effort(
                kw(&engine, want)
                    .get(ENABLE_THINKING)
                    .and_then(Value::as_bool),
            );
            eprintln!(
                "AUDIT {name} reasoning.enabled={want:?}: {}; reasoning_effort {effort:?}",
                match r {
                    Ok(p) => format!(
                        "thinking {} by {}, enable_thinking passed {:?}",
                        p.thinking.unwrap().name(),
                        p.thinking.unwrap().by.name(),
                        p.kwargs.get(ENABLE_THINKING)
                    ),
                    Err(e) => format!("refused: {e}"),
                }
            );
        }
        for persist in [None, Some(true), Some(false)] {
            let f = Front::new(
                &engine,
                SessionOptions {
                    persist,
                    ..SessionOptions::default()
                },
            );
            let p = build(&f, &engine, &one, None);
            eprintln!(
                "AUDIT {name} session.persist_thinking={persist:?}: {}; preserve_thinking passed {:?}",
                f.persist_unsupported()
                    .map_or("served".to_string(), |w| format!("refused at load: {w}")),
                p.kwargs.get(PRESERVE_THINKING)
            );
        }

        // Where the next turn leaves the log, per thinking × persistence ×
        // reply.
        for want in wants(&engine) {
            for persist in [None, Some(true), Some(false)] {
                let front = Front::new(
                    &engine,
                    SessionOptions {
                        persist,
                        ..SessionOptions::default()
                    },
                );
                if front.persist_unsupported().is_some() {
                    continue;
                }
                let p1 = build(&front, &engine, &one, want);
                let on = p1.thinking.unwrap().on;
                // The reply in the template's own format: its rendering of
                // the reply as the conversation's last message (which every
                // template keeps whole), past the prompt that ran.
                let mut own = json!({ "role": "assistant", "content": "Paris." });
                if on {
                    for f in REASONING_FIELDS {
                        own[f] = json!("The capital of France is Paris.");
                    }
                }
                let full = render(&json!([user1, own]), &p1.kwargs, false).unwrap();
                // Where the engine closed the block the template opened (no
                // switch), the reply follows that marker, which no rendering
                // of the history writes: it is the answer alone.
                let by_engine = p1.thinking.unwrap().by == By::Engine;
                let formatted = match full.strip_prefix(p1.text.as_str()) {
                    Some(rest) => {
                        rest[..rest.rfind("<|im_end|>").unwrap_or(rest.len())].to_string()
                    }
                    None => {
                        assert!(
                            by_engine,
                            "{name} want={want:?}: the prompt that ran is not a prefix of the \
                             template's rendering of its reply"
                        );
                        "Paris.".to_string()
                    }
                };
                let mut replies = vec![("own format", formatted.clone(), Stop::Eos)];
                if on {
                    let end = engine.token_str(tags.as_ref().unwrap().end);
                    if let Some(at) = formatted.find(end.as_str()) {
                        let mut extra = formatted.clone();
                        extra.insert(at, '\n');
                        replies.push(("one more newline before the end marker", extra, Stop::Eos));
                    }
                    replies.push((
                        "cut inside its reasoning",
                        "The capital of France is".into(),
                        Stop::Length,
                    ));
                }
                for (kind, text, stop) in replies {
                    let front = Front::new(
                        &engine,
                        SessionOptions {
                            persist,
                            ..SessionOptions::default()
                        },
                    );
                    let p1 = build(&front, &engine, &one, want);
                    let sampled = engine.enc(&text);
                    let mut d = decoded(&engine, sampled.clone());
                    d.stop = stop;
                    assert!(front.record(&engine, p1.origin, &d), "{name} {kind}");
                    let history = json!([user1, assistant(&d.text), user2]);
                    let p2 = build(&front, &engine, &history, want);
                    let log: Vec<u32> = p1.ids.iter().chain(&sampled).copied().collect();
                    let kept = p2.ids.iter().zip(&log).take_while(|(a, b)| a == b).count();
                    let ckpt = p1.ids.len() - p1.turn_tail;
                    let extends = kept == log.len();
                    eprintln!(
                        "AUDIT {name} want={want:?} thinking={} persist={persist:?} {kind}: preserve_thinking \
                         {:?}; the next prompt leaves the log at {:+} from the prompt's end, {} \
                         (checkpoint at {:+})",
                        p1.thinking.unwrap().name(),
                        p2.kwargs.get(PRESERVE_THINKING),
                        kept as i64 - p1.ids.len() as i64,
                        if extends { "extends" } else { "diverges" },
                        ckpt as i64 - p1.ids.len() as i64,
                    );
                    if kept < ckpt {
                        failures.push(format!(
                            "{name} want={want:?} persist={persist:?} {kind}: the next prompt leaves \
                             the log at {kept}, before the checkpoint at {ckpt}"
                        ));
                    }
                    if persist != Some(false)
                        && !by_engine
                        && kind != "one more newline before the end marker"
                        && !extends
                    {
                        failures.push(format!(
                            "{name} want={want:?} persist={persist:?} {kind}: the history keeps \
                             what the model generated, so the next prompt extends the log; it \
                             leaves it at {:+} from the prompt's end",
                            kept as i64 - p1.ids.len() as i64
                        ));
                    }
                }
            }
        }
        ran += 1;
    }
    eprintln!("{ran} checkpoint templates audited");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A reply with text before a reasoning block the model opened itself (it
/// sampled the start marker mid-reply: thinking off, an answer naming the
/// tag) cannot be given to the template as reasoning and content apart: a
/// template writes the reasoning first, and the text before the block is in
/// neither part. It goes as sent, open or closed, under every arm and
/// either `missing_reasoning`, so the next prompt carries all of it.
#[test]
fn a_reply_with_text_before_the_block_it_opened_goes_as_sent() {
    let engine = Stub::qwen4exp();
    let want = Some(false);
    let start = TinyTokenizer::qwen4exp().id("<think>").unwrap();
    let end = q4_end();
    let q = json!([user("Spell it.")]);
    let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
    for (case, ids) in [
        (
            "open",
            [bytes("It is "), vec![start], bytes(" in text.")].concat(),
        ),
        (
            "closed",
            [bytes("Hi "), vec![start], bytes("R"), vec![end], bytes("A")].concat(),
        ),
    ] {
        for arm in [Incomplete::Text, Incomplete::Reasoning, Incomplete::Recover] {
            for restore_reasoning in [false, true] {
                let front = Front::new(
                    &engine,
                    SessionOptions {
                        incomplete: arm,
                        restore_reasoning,
                        ..SessionOptions::default()
                    },
                );
                let p1 = build(&front, &engine, &q, want);
                let d = decoded(&engine, ids.clone());
                assert!(front.record(&engine, p1.origin, &d), "{case}");
                let r = front.reply(&d.text).unwrap();
                assert!(
                    r.block.interior().is_some_and(|i| i.start > 1),
                    "{case}: the precondition, text before the block: {:?}",
                    r.block
                );
                let history = with(&q, &[assistant(&d.text), user("Again?")]);
                let p2 = build(&front, &engine, &history, want);
                let at = format!("{case} {} restore={restore_reasoning}", arm.name());
                assert_eq!(p2.spliced.matched, 1, "{at}");
                assert_eq!(
                    decode(&p2.ids),
                    decode(&engine.reference(&history, &kw(&engine, want), want)),
                    "{at}: the history as sent"
                );
            }
        }
    }
}

/// The load line's `gen/tail` is the checkpoint tail a request that sets no
/// persistence runs with. Thinking off can still need an interior checkpoint:
/// the engine closes the block, but the template omits that empty block
/// when rendering the reply again.
#[test]
fn the_load_line_reports_the_tail_the_default_runs_with() {
    let engine = Stub::qwen3_5();
    let tails = |persist: Option<bool>| -> Vec<String> {
        let front = Front::new(
            &engine,
            SessionOptions {
                persist,
                ..SessionOptions::default()
            },
        );
        let line = front.line();
        let list = line.rsplit_once("=[").unwrap().1.trim_end_matches(']');
        let reported: Vec<String> = list
            .split(", ")
            .map(|g| g.split('/').nth(1).unwrap_or("-").to_string())
            .collect();
        let messages = json!([user("Hello")]);
        let actual: Vec<String> = MODES
            .into_iter()
            .map(|want| {
                build(&front, &engine, &messages, want)
                    .turn_tail
                    .to_string()
            })
            .collect();
        assert_eq!(reported, actual, "persist={persist:?}: {line}");
        reported
    };
    assert_eq!(tails(None), ["0", "0", "0"], "persisted by default");
    assert_eq!(tails(Some(true)), ["0", "0", "0"]);
    assert!(
        tails(Some(false)).iter().all(|t| t != "0"),
        "stripped: {:?}",
        tails(Some(false))
    );
}

#[test]
fn neutral_gguf_effort_is_accepted_without_inventing_a_template_control() {
    let mut engine = Stub::lfm2();
    engine.caps = crate::app::causal_loaded::served::lfm2_gguf_capabilities();
    let front = Front::new(&engine, SessionOptions::default());
    let messages = json!([user("Hi")]);
    let base = front
        .prompt(
            &engine,
            Request {
                messages: &messages,
                kwargs: Map::new(),
                want: None,
                add_bos: true,
                expand: None,
                history: None,
            },
        )
        .unwrap();
    for effort in ["none", "default"] {
        let prompt = front
            .prompt(
                &engine,
                Request {
                    messages: &messages,
                    kwargs: Map::from_iter([(REASONING_EFFORT.into(), json!(effort))]),
                    want: None,
                    add_bos: true,
                    expand: None,
                    history: None,
                },
            )
            .unwrap();
        // The GGUF owner exposes rendering through Served without Text. Its
        // plain path performs a second ignored-variable check of its own.
        let plain = front
            .plain(
                &engine,
                Request {
                    messages: &messages,
                    kwargs: Map::from_iter([(REASONING_EFFORT.into(), json!(effort))]),
                    want: None,
                    add_bos: true,
                    expand: None,
                    history: None,
                },
            )
            .unwrap();
        assert_eq!(plain.ids, base.ids);
        assert_eq!(prompt.ids, base.ids);
        assert_eq!(prompt.thinking, base.thinking);
    }
    // Meaningful unsupported controls still refuse; supported templates keep
    // their own validation (covered by the existing template tests).
    assert!(front
        .prompt(
            &engine,
            Request {
                messages: &messages,
                kwargs: Map::from_iter([(REASONING_EFFORT.into(), json!("high"))]),
                want: None,
                add_bos: true,
                expand: None,
                history: None
            }
        )
        .is_err());
}

#[test]
fn other_families_keep_their_existing_neutral_effort_validation() {
    let engine = Stub::lfm2(); // The safetensors/Metal family, not lfm2-gguf.
    let front = Front::new(&engine, SessionOptions::default());
    let messages = json!([user("Hi")]);
    for effort in ["none", "default"] {
        let request = || Request {
            messages: &messages,
            kwargs: Map::from_iter([(REASONING_EFFORT.into(), json!(effort))]),
            want: None,
            add_bos: true,
            expand: None,
            history: None,
        };
        let error = front.prompt(&engine, request()).unwrap_err().to_string();
        assert!(
            error.contains("lfm2's chat template does not read reasoning_effort"),
            "{error}"
        );
        let error = front.plain(&engine, request()).unwrap_err().to_string();
        assert!(
            error.contains("lfm2's chat template does not read reasoning_effort"),
            "{error}"
        );
    }
}
