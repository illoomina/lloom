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
        readback: ReadbackArm::Blocking,
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
/// render, one encode and the thinking switch gave before the front end, on
/// every tiny template, every conversation and every think mode; it took one
/// render and one tokenization pass.
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
                let reference = engine.reference(&conv, &kw(&engine, want), want);
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

/// A reply with reasoning, on a template that strips it by default: the
/// next turn's prompt equals turn 1's up to the checkpoint (the tail the
/// constants gave), then carries the reply's content as sampled. Persisted
/// (`preserve_thinking`), the whole log is a prefix.
#[test]
fn reasoning_stripped_splices_the_content_and_persisted_extends() {
    for persist in [None, Some(true)] {
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
            None => {
                assert_eq!(p1.turn_tail, 2);
                assert_ne!(p2.ids[kept], p1.ids[kept], "diverges at the checkpoint");
                let four = bytes("Four.");
                assert_eq!(&p2.ids[kept..kept + four.len()], &four[..]);
                assert_eq!(p2.spliced.spliced_tokens, four.len());
            }
            Some(_) => {
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
            let kwargs = kw(&engine, want);
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
                let kwargs = kw(&engine, want);
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
        // every id of it is spliced, and the prompt's text is the template's.
        for want in wants(&engine) {
            let fresh = Front::new(&engine, SessionOptions::default());
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
            let kwargs = kw(&engine, want);
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
/// rendering of the whole text the client got: stripped, LFM2's template
/// splits the content at the end marker and trims it, and so does the front
/// end's rewrite; kept (`preserve_thinking`), the rewrite is the model's
/// sampled text and the log extends.
#[test]
fn a_reply_given_back_renders_as_the_templates_own_split() {
    for persist in [None, Some(true)] {
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
        if let Some(p) = persist {
            kw.insert(PRESERVE_THINKING.into(), Value::Bool(p));
        }
        let decode = |ids: &[u32]| engine.text.tokenizer.decode(ids, false).unwrap();
        match persist {
            None => assert_eq!(
                decode(&p2.ids),
                decode(&engine.reference(&second, &kw, None)),
                "stripped: the template's own split of the whole text"
            ),
            Some(_) => {
                let mut log = p1.ids.clone();
                log.extend_from_slice(&sampled);
                assert!(p2.ids.starts_with(&log), "kept: the log extends");
            }
        }
    }
}

/// A reply that closes its block twice (a thinking budget closed it, the
/// model went on reasoning and wrote the marker itself): a stripping
/// template keeps only what follows the last marker, and the prompt is that
/// render with the final content's sampled ids spliced. Kept, the reply goes
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
        for persist in [None, Some(true)] {
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
            if let Some(p) = persist {
                kw.insert(PRESERVE_THINKING.into(), Value::Bool(p));
            }
            match persist {
                None => {
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
                Some(_) if whole => {
                    let mut log = p1.ids.clone();
                    log.extend_from_slice(&sampled);
                    assert!(p2.ids.starts_with(&log), "kept: the log extends");
                }
                Some(_) => {
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
