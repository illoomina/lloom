//! The per-response machine against fixed references and against its
//! event-list oracle (decision 85). Every case finishes the machine, which
//! runs the oracle's settlement check; `checked` asserts it agreed.
use super::*;

const START: u32 = 1000;
const END: u32 = 1001;
const TOOL: u32 = 1002;

fn markers() -> Vec<Marker> {
    [
        (START, "<think>", Kind::Start),
        (END, "</think>", Kind::End),
        (TOOL, "<tool_call>", Kind::Tool),
    ]
    .into_iter()
    .map(|(id, text, kind)| Marker {
        id,
        text: text.into(),
        kind,
    })
    .collect()
}

fn parser(inside: bool) -> Reasoning {
    Reasoning::new(markers(), inside, false)
}

fn by_text(inside: bool) -> Reasoning {
    Reasoning::new(markers(), inside, true)
}

fn decode(ids: &[u32]) -> lloom_tensor::Result<String> {
    Ok(ids
        .iter()
        .map(|id| match *id {
            START => "<think>".into(),
            END => "</think>".into(),
            TOOL => "<tool_call>".into(),
            id => char::from_u32(id).unwrap().to_string(),
        })
        .collect())
}

fn ids_of(text: &str) -> Vec<u32> {
    text.chars().map(u32::from).collect()
}

fn collect(parts: impl IntoIterator<Item = Map<String, Value>>) -> (String, String) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    for part in parts {
        content.push_str(part.get("content").and_then(Value::as_str).unwrap_or(""));
        reasoning.push_str(
            part.get("reasoning_content")
                .and_then(Value::as_str)
                .unwrap_or(""),
        );
    }
    (content, reasoning)
}

/// The buffers after finish, with the oracle's verdict asserted.
fn checked(p: &Reasoning) -> (String, String) {
    assert_eq!(p.oracle_mismatch(), None, "the event-list oracle disagrees");
    (p.content.clone(), p.reasoning.clone())
}

#[test]
fn all_fragment_boundaries_preserve_channels_and_partial_tags() {
    for (open, ids, cut, content, reason, transitions) in [
        (true, vec![97, 960, END, 65], 0, "A", "aπ", 1),
        (false, vec![START, 97, END, 65], 0, "A", "a", 2),
        (true, vec![97, END], 3, "", "a</thi", 0),
        (true, vec![97, TOOL, 123, 125], 0, "<tool_call>{}", "a", 1),
        (false, vec![65], 0, "A", "", 0),
        (true, vec![END], 0, "", "", 1),
    ] {
        let raw = decode(&ids).unwrap();
        let text = &raw[..raw.len() - cut];
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .collect();
        for &i in &boundaries {
            for &j in boundaries.iter().filter(|&&j| j >= i) {
                let mut p = parser(open);
                p.observe(&ids, &raw, 0, &decode).unwrap();
                let parts = [
                    p.push(&text[..i]),
                    p.push(&text[i..j]),
                    p.push(&text[j..]),
                    p.finish_fields(),
                ];
                let want = (content.to_owned(), reason.to_owned());
                assert_eq!(collect(parts), want, "{open} {text:?} {i} {j}");
                assert_eq!(checked(&p), want, "buffers {open} {text:?} {i} {j}");
                assert_eq!(p.transitions(), transitions, "{open} {text:?}");
                assert_eq!(p.published(), text.len());
            }
        }
    }
}

#[test]
fn literal_tag_spelling_is_not_a_control_token() {
    for inside in [true, false] {
        let text = "literal <think> and </think> and <tool_call> π";
        let mut p = parser(inside);
        let mut parts = Vec::new();
        let mut from = 0;
        for c in text.chars() {
            let fragment = c.to_string();
            p.observe(&[c as u32], &fragment, from, &decode).unwrap();
            from += fragment.len();
            parts.push(p.push(&fragment));
        }
        parts.push(p.finish_fields());
        let want = if inside {
            (String::new(), text.to_owned())
        } else {
            (text.to_owned(), String::new())
        };
        assert_eq!(collect(parts), want);
        assert_eq!(checked(&p), want);
        assert_eq!(
            p.transitions(),
            0,
            "typed-as-text markers are no transition"
        );
    }
}

#[test]
fn a_re_observed_suffix_replaces_only_spans_past_the_frontier() {
    // The decoder re-observes an incomplete run from the same origin; the
    // second attempt no longer carries the marker the first one placed.
    let mut p = parser(true);
    p.observe(&[97], "a", 0, &decode).unwrap();
    assert_eq!(collect([p.push("a")]), (String::new(), "a".into()));
    p.observe(&[END, 0xfffd], "</think>\u{fffd}", 1, &decode)
        .unwrap();
    assert_eq!(p.retained(), (1, 0));
    p.observe(&ids_of("</think>"), "</think>", 1, &decode)
        .unwrap();
    assert_eq!(p.retained(), (0, 0), "the retracted marker span is gone");
    let parts = [p.push("</think>"), p.finish_fields()];
    let want = (String::new(), "</think>".to_owned());
    assert_eq!(collect(parts), want, "a literal re-spelling stays text");
    assert_eq!(checked(&p), ("".into(), "a</think>".into()));
    assert_eq!(p.transitions(), 0);

    // And the other way: a literal attempt replaced by the marker id.
    let mut p = parser(true);
    p.observe(&ids_of("a</think>"), "a</think>", 0, &decode)
        .unwrap();
    p.observe(&[97, END], "a</think>", 0, &decode).unwrap();
    p.observe(&[66], "B", 9, &decode).unwrap();
    let parts = [p.push("a</th"), p.push("ink>B"), p.finish_fields()];
    assert_eq!(collect(parts), ("B".into(), "a".into()));
    assert_eq!(checked(&p), ("B".into(), "a".into()));
    assert_eq!(p.transitions(), 1);
}

#[test]
fn a_marker_split_across_publications_is_held_and_credited_when_complete() {
    let ids = [97, END, 88];
    let raw = decode(&ids).unwrap();
    let mut p = parser(true);
    p.observe(&ids, &raw, 0, &decode).unwrap();
    let (a, resolved_a) = p.push_counted("a<");
    assert_eq!(collect([a]), ("".into(), "a".into()));
    assert_eq!(resolved_a, 1, "the held '<' is not credited yet");
    assert_eq!(p.retained(), (1, 1));
    let (b, resolved_b) = p.push_counted("/think>X");
    assert_eq!(collect([b]), ("X".into(), "".into()));
    assert_eq!(resolved_b, raw.len() - 1);
    assert_eq!(p.retained(), (0, 0));
    let (c, resolved_c) = p.finish_counted();
    assert!(c.is_empty());
    assert_eq!(resolved_c, 0);
    assert_eq!(checked(&p), ("X".into(), "a".into()));
}

#[test]
fn a_tool_call_leaves_the_block_and_keeps_its_literal_in_content() {
    for inside in [true, false] {
        let ids = [97, TOOL, 123, END, 125];
        let raw = decode(&ids).unwrap();
        let mut p = parser(inside);
        p.observe(&ids, &raw, 0, &decode).unwrap();
        for c in raw.chars() {
            p.push(&c.to_string());
        }
        p.finish();
        let want = if inside {
            ("<tool_call>{</think>}".to_owned(), "a".to_owned(), 1)
        } else {
            ("a<tool_call>{</think>}".to_owned(), String::new(), 0)
        };
        assert_eq!((checked(&p), p.transitions()), ((want.0, want.1), want.2));
    }
}

#[test]
fn a_runaway_reply_keeps_constant_marker_state_and_counts_its_transitions() {
    // More markers than the retired descriptor bound (65,536) admitted.
    const PAIRS: usize = 40_000;
    let mut p = parser(false);
    let mut from = 0;
    let mut max_retained = (0, 0);
    for _ in 0..PAIRS {
        for (ids, text) in [(&[START][..], "<think>"), (&[97, END][..], "a</think>")] {
            p.observe(ids, text, from, &decode).unwrap();
            from += text.len();
            p.publish(text);
            let r = p.retained();
            max_retained = (max_retained.0.max(r.0), max_retained.1.max(r.1));
        }
    }
    p.finish();
    assert_eq!(
        max_retained,
        (0, 0),
        "no marker state outlives its publication"
    );
    assert_eq!(p.transitions(), 2 * PAIRS as u64);
    assert_eq!(checked(&p), (String::new(), "a".repeat(PAIRS)));
}

#[test]
fn the_unpublished_window_alone_is_retained() {
    // Markers observed ahead of a held stop-string prefix wait in `ahead`;
    // nothing published stays behind.
    let mut p = parser(false);
    let ids = [START, 97, END, START, 98];
    let raw = decode(&ids).unwrap();
    p.observe(&ids, &raw, 0, &decode).unwrap();
    assert_eq!(p.retained(), (3, 0));
    p.push("<think>a");
    assert_eq!(p.retained(), (2, 0));
    p.push("</think><th");
    assert_eq!(p.retained(), (1, 3));
    p.push("ink>b");
    assert_eq!(p.retained(), (0, 0));
    p.finish();
    assert_eq!(checked(&p), (String::new(), "ab".into()));
    assert_eq!(p.transitions(), 3);
}

#[test]
fn repeated_pending_decode_and_lagging_publication_do_not_duplicate_markers() {
    let mut p = parser(true);
    let ids = [97, END];
    let text = decode(&ids).unwrap();
    p.observe(&ids, &text, 0, &decode).unwrap();
    p.observe(&ids, &text, 0, &decode).unwrap();
    p.observe(&[65], "A", text.len(), &decode).unwrap();
    let parts = [p.push(&(text + "A")), p.finish_fields()];
    assert_eq!(collect(parts), ("A".into(), "a".into()));
    assert_eq!(checked(&p), ("A".into(), "a".into()));
}

#[test]
fn text_mode_places_literals_across_publications_and_holds_a_prefix() {
    let text = "The user asks.</think>Paris<think>more";
    let boundaries: Vec<_> = (0..=text.len()).collect();
    for &i in &boundaries {
        for &j in boundaries.iter().filter(|&&j| j >= i) {
            let mut p = by_text(true);
            let parts = [
                p.push(&text[..i]),
                p.push(&text[i..j]),
                p.push(&text[j..]),
                p.finish_fields(),
            ];
            let want = ("Paris".to_owned(), "The user asks.more".to_owned());
            assert_eq!(collect(parts), want, "{i} {j}");
            assert_eq!(checked(&p), want);
            assert_eq!(p.transitions(), 2);
        }
    }
    // Cut short by a stop string: the held prefix is text in its block.
    let mut p = by_text(true);
    assert_eq!(
        collect([p.push("thought</th")]),
        ("".into(), "thought".into())
    );
    assert_eq!(collect([p.finish_fields()]), ("".into(), "</th".into()));
    assert_eq!(checked(&p), ("".into(), "thought</th".into()));
    // Observations are not the text mode's input.
    let mut p = by_text(false);
    p.observe(&[START], "<think>", 0, &decode).unwrap();
    assert_eq!(p.retained(), (0, 0));
}

#[test]
fn the_byte_tokenizer_stream_utf8_stops_and_publication_order_agree_with_the_oracle() {
    use lloom::testing::tiny::{tokenizer::tokenizer_json, TinyTokenizer};
    let tiny = TinyTokenizer::lfm2();
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_json(&tiny.added).as_bytes()).unwrap();
    let start = tokenizer.token_to_id("<think>").unwrap();
    let end = tokenizer.token_to_id("</think>").unwrap();
    let mut ids: Vec<u32> = "literal <think> π".bytes().map(u32::from).collect();
    ids.push(end);
    ids.extend("Answer".bytes().map(u32::from));
    ids.push(start);
    ids.extend("π".bytes().map(u32::from));
    for delayed in [false, true] {
        for stops in [vec![], vec!["/think>".to_owned()]] {
            let mut p = Reasoning::new(
                vec![
                    Marker {
                        id: start,
                        text: "<think>".into(),
                        kind: Kind::Start,
                    },
                    Marker {
                        id: end,
                        text: "</think>".into(),
                        kind: Kind::End,
                    },
                ],
                true,
                false,
            );
            let p_ref = parking_lot::Mutex::new(&mut p);
            let mut stream = crate::stream::TextStream::default();
            let mut from = 0;
            let mut observer =
                |ids: &[u32],
                 text: &str,
                 from,
                 decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>| {
                    p_ref.lock().observe(ids, text, from, decode)
                };
            let mut decoder = |ids: &[u32]| {
                crate::app::causal_loaded::observed_decode(
                    &tokenizer,
                    ids,
                    &mut from,
                    &mut observer,
                )
            };
            let mut published = Vec::new();
            for id in &ids {
                let (text, stopped) = stream.publish(&[*id], &mut decoder, &stops).unwrap();
                if let Some(text) = text {
                    published.push(text);
                }
                if stopped {
                    break;
                }
            }
            let (text, _) = stream.flush(&mut decoder, &stops).unwrap();
            published.extend(text);
            let (raw, _) = stream.finish(&stops);
            let p = p_ref.into_inner();
            if delayed {
                let joined: String = published.concat();
                p.publish(&joined);
            } else {
                for text in &published {
                    p.publish(text);
                }
            }
            p.finish();
            assert_eq!(
                p.published(),
                raw.len(),
                "the published text is the accepted text"
            );
            let want = if stops.is_empty() {
                ("Answer".to_owned(), "literal <think> ππ".to_owned())
            } else {
                (String::new(), "literal <think> π<".to_owned())
            };
            assert_eq!(checked(p), want, "delayed={delayed} stops={stops:?}");
        }
    }
}

// The production byte decoder's from/pending rule: a marker plus the first
// byte of pi stays pending through repeated observations until the second
// UTF-8 byte arrives, and only then is it published.
#[test]
fn an_incomplete_utf8_run_after_a_marker_is_replaced_at_one_origin() {
    use lloom::testing::tiny::{tokenizer::tokenizer_json, TinyTokenizer};
    let tiny = TinyTokenizer::lfm2();
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_json(&tiny.added).as_bytes()).unwrap();
    let start = tokenizer.token_to_id("<think>").unwrap();
    let end = tokenizer.token_to_id("</think>").unwrap();
    let mut p = Reasoning::new(
        vec![
            Marker {
                id: start,
                text: "<think>".into(),
                kind: Kind::Start,
            },
            Marker {
                id: end,
                text: "</think>".into(),
                kind: Kind::End,
            },
        ],
        false,
        false,
    );
    let mut stream = crate::stream::TextStream::default();
    let mut from = 0;
    let publish = |p: &mut Reasoning,
                   ids: &[u32],
                   stream: &mut crate::stream::TextStream,
                   from: &mut usize| {
        let p = parking_lot::Mutex::new(p);
        let mut observe =
            |ids: &[u32],
             text: &str,
             from,
             decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>| {
                p.lock().observe(ids, text, from, decode)
            };
        let mut decode = |ids: &[u32]| {
            crate::app::causal_loaded::observed_decode(&tokenizer, ids, from, &mut observe)
        };
        let (text, stopped) = stream.publish(ids, &mut decode, &[]).unwrap();
        assert!(!stopped);
        if let Some(text) = &text {
            p.lock().publish(text);
        }
        text
    };
    const GROUPS: usize = 32;
    for _ in 0..GROUPS {
        publish(&mut p, &[start, 97, end, 65], &mut stream, &mut from).unwrap();
        assert_eq!(p.retained(), (0, 0), "a published group retains nothing");
    }
    for byte in " literal <think> </think> ".bytes() {
        publish(&mut p, &[u32::from(byte)], &mut stream, &mut from);
    }
    publish(&mut p, &[start, 98], &mut stream, &mut from).unwrap();
    let origin = from;
    for ids in [&[end, 0xcf][..], &[][..], &[][..]] {
        assert_eq!(publish(&mut p, ids, &mut stream, &mut from), None);
        assert_eq!(from, origin, "an incomplete run is replaced at one origin");
        assert_eq!(p.retained(), (1, 0), "one marker span, never duplicated");
    }
    assert_eq!(
        publish(&mut p, &[0x80], &mut stream, &mut from).as_deref(),
        Some("</think>π")
    );
    assert_eq!(p.retained(), (0, 0));
    let (raw, _) = stream.finish(&[]);
    p.finish();
    assert_eq!(p.published(), raw.len());
    assert_eq!(
        checked(&p),
        (
            format!("{} literal <think> </think> π", "A".repeat(GROUPS)),
            format!("{}b", "a".repeat(GROUPS))
        )
    );
}

#[test]
fn the_oracle_catches_a_machine_that_disagrees() {
    // The oracle's positive control: drive a machine whose transition is
    // corrupted and check the oracle names the disagreement, without the
    // settlement panic.
    let mut p = parser(true);
    p.observe(&[97, END, 65], "a</think>A", 0, &decode).unwrap();
    p.inside = false;
    p.publish("a</think>A");
    assert!(
        p.oracle_mismatch()
            .is_some_and(|m| m.contains("publication")),
        "{:?}",
        p.oracle_mismatch()
    );
}

#[test]
#[should_panic(expected = "disagrees with its event-list oracle")]
fn a_disagreement_panics_at_settlement_in_a_debug_build() {
    let mut p = parser(true);
    p.observe(&[97, END, 65], "a</think>A", 0, &decode).unwrap();
    p.inside = false;
    p.publish("a</think>A");
    p.finish();
}
