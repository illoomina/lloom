//! Chat wire fields: reasoning in `reasoning_content`, the answer in `content`.
//! `reasoning_in_content` opts out. Typed observations identify marker tokens,
//! not ordinary tokens spelling the same literal; legacy adapters use the text
//! fallback. Raw engine history and non-chat completions remain unchanged.
use crate::app::served::Served;
use anyhow::Result;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Start,
    End,
    Tool,
}

#[derive(Clone)]
struct Marker {
    id: u32,
    text: String,
    kind: Kind,
}
#[derive(Clone, Copy)]
struct Event {
    start: usize,
    end: usize,
    kind: Kind,
}

pub(super) struct Reasoning {
    markers: Vec<Marker>,
    events: Arc<Mutex<Events>>,
    initial_inside: bool,
    inside: bool,
    pending: String,
    offset: usize,
    /// Markers placed from the text itself (each marker's literal, with a
    /// partial suffix held back across pushes) because the family supplies no
    /// decode observations (`Served::observes_decode`). Observation-placed
    /// markers are exact by id; text-placed ones match the marker's literal.
    by_text: bool,
}

const MAX_MARKERS: usize = 1 << 16;

/// The producer replaces only its unpublished UTF-8 suffix. The consumer
/// removes only markers completed by already accepted text; retained markers
/// replay the accepted fields before a slow socket finishes.
#[derive(Default)]
struct Events {
    pending: VecDeque<Event>,
    retained: Vec<Event>,
}

pub(super) struct Observer {
    markers: Vec<Marker>,
    events: Arc<Mutex<Events>>,
    scratch: Vec<Event>,
}

impl Observer {
    pub(super) fn new(parser: &Arc<Mutex<Reasoning>>) -> Self {
        parser.lock().observer()
    }

    pub(super) fn observe(
        &mut self,
        ids: &[u32],
        text: &str,
        from: usize,
        decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>,
    ) -> lloom_tensor::Result<()> {
        self.scratch.clear();
        for (i, id) in ids.iter().enumerate() {
            let Some(marker) = self.markers.iter().find(|marker| marker.id == *id) else {
                continue;
            };
            let before = decode(&ids[..i])?;
            let through = decode(&ids[..=i])?;
            if !text.starts_with(&through)
                || through.strip_prefix(&before) != Some(marker.text.as_str())
            {
                lloom_tensor::bail!("reasoning marker decoding is not an additive literal")
            }
            if self.scratch.len() == MAX_MARKERS {
                lloom_tensor::bail!("response marker descriptor limit exceeded")
            }
            self.scratch.push(Event {
                start: from + before.len(),
                end: from + through.len(),
                kind: marker.kind,
            });
        }
        let mut events = self.events.lock();
        while events
            .pending
            .back()
            .is_some_and(|event| event.start >= from)
        {
            events.pending.pop_back();
        }
        while events
            .retained
            .last()
            .is_some_and(|event| event.start >= from)
        {
            events.retained.pop();
        }
        if events.retained.len() + self.scratch.len() > MAX_MARKERS {
            lloom_tensor::bail!("response marker descriptor limit exceeded")
        }
        events.pending.extend(self.scratch.iter().copied());
        events.retained.extend(self.scratch.iter().copied());
        Ok(())
    }
}

impl Reasoning {
    /// The splitter a chat request runs with: `None` where the request asked
    /// for today's rendering (`reasoning_in_content`) or the template has no
    /// reasoning markers to split on.
    pub(super) fn for_chat(
        engine: &dyn Served,
        prompt: &[u32],
        in_content: bool,
    ) -> Result<Option<Self>> {
        if in_content {
            return Ok(None);
        }
        let Some(tags) = engine.think_tags() else {
            return Ok(None);
        };
        let mut markers = Vec::new();
        for (id, kind) in [(tags.start, Kind::Start), (tags.end, Kind::End)]
            .into_iter()
            .chain(tags.tool_call.map(|id| (id, Kind::Tool)))
        {
            let text = engine.token_str_checked(id)?;
            anyhow::ensure!(!text.is_empty(), "empty reasoning marker");
            markers.push(Marker { id, text, kind });
        }
        Ok(Some(Self {
            markers,
            events: Arc::new(Mutex::new(Events::default())),
            initial_inside: tags.open_at_end(prompt).is_some(),
            inside: tags.open_at_end(prompt).is_some(),
            pending: String::new(),
            offset: 0,
            // Without decode observations the markers cannot be placed by id,
            // and a prompt that opened the block would never close it (the
            // answer came back in `reasoning_content`): place them from the
            // text instead.
            by_text: !engine.observes_decode(),
        }))
    }

    /// Text mode: events for every complete marker literal in `pending` past
    /// the last placed one, and the length of a trailing partial literal to
    /// hold back until the next push (a stop string may cut it short; finish
    /// emits it as it is). Idempotent across pushes: `push` consumes every
    /// event it places, so a region is scanned once.
    fn place_from_text(&mut self) -> usize {
        let base = self.offset;
        let mut pos = 0;
        loop {
            let mut best: Option<(usize, usize, Kind)> = None;
            for marker in &self.markers {
                if let Some(at) = self.pending[pos..].find(marker.text.as_str()) {
                    let at = pos + at;
                    if best.map_or(true, |b: (usize, usize, Kind)| at < b.0) {
                        best = Some((at, marker.text.len(), marker.kind));
                    }
                }
            }
            let Some((at, len, kind)) = best else { break };
            self.events.lock().pending.push_back(Event {
                start: base + at,
                end: base + at + len,
                kind,
            });
            pos = at + len;
        }
        let tail = &self.pending[pos..];
        let mut hold = 0;
        for marker in &self.markers {
            for k in 1..marker.text.len().min(tail.len() + 1) {
                if marker.text.is_char_boundary(k) && tail.ends_with(&marker.text[..k]) {
                    hold = hold.max(k);
                }
            }
        }
        hold
    }

    fn observer(&self) -> Observer {
        Observer {
            markers: self.markers.clone(),
            events: Arc::clone(&self.events),
            scratch: Vec::with_capacity(16),
        }
    }

    #[cfg(test)]
    pub(super) fn observe(
        &mut self,
        ids: &[u32],
        text: &str,
        from: usize,
        decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>,
    ) -> lloom_tensor::Result<()> {
        self.observer().observe(ids, text, from, decode)
    }

    /// Route the settled accepted text without waiting for network delivery.
    /// Retained descriptors distinguish genuine marker ids from literal bytes.
    pub(super) fn accepted_fields(&self, text: &str) -> Map<String, Value> {
        let retained = self.events.lock().retained.clone();
        let mut parser = Self {
            markers: self.markers.clone(),
            events: Arc::new(Mutex::new(Events {
                pending: retained.into(),
                retained: Vec::new(),
            })),
            initial_inside: self.initial_inside,
            inside: self.initial_inside,
            pending: String::new(),
            offset: 0,
            by_text: self.by_text,
        };
        let mut fields = parser.push(text);
        for (key, value) in parser.finish() {
            let next = value.as_str().unwrap_or("");
            match fields
                .entry(key)
                .or_insert_with(|| Value::String(String::new()))
            {
                Value::String(previous) => previous.push_str(next),
                _ => unreachable!("reasoning fields are strings"),
            }
        }
        fields
    }

    /// Only a genuine marker crossing a writer batch is held. A stop string
    /// may cut a marker short; finish then emits those bytes without a transition.
    pub(super) fn push(&mut self, text: &str) -> Map<String, Value> {
        self.pending.push_str(text);
        let hold = if self.by_text {
            self.place_from_text()
        } else {
            0
        };
        let (mut content, mut reasoning) = (String::new(), String::new());
        let end = self.offset + self.pending.len() - hold;
        let mut consumed = 0;
        loop {
            let event = self.events.lock().pending.front().copied();
            let until = event.map_or(end, |event| event.start.min(end));
            let count = until - self.offset;
            let out = if self.inside {
                &mut reasoning
            } else {
                &mut content
            };
            out.push_str(&self.pending[consumed..consumed + count]);
            consumed += count;
            self.offset = until;
            let Some(event) = event else {
                break;
            };
            if event.end > end {
                break;
            }
            self.events.lock().pending.pop_front();
            let count = event.end - self.offset;
            match event.kind {
                Kind::Start if !self.inside => self.inside = true,
                Kind::End if self.inside => self.inside = false,
                Kind::Tool if self.inside => {
                    self.inside = false;
                    content.push_str(&self.pending[consumed..consumed + count]);
                }
                _ => {
                    let out = if self.inside {
                        &mut reasoning
                    } else {
                        &mut content
                    };
                    out.push_str(&self.pending[consumed..consumed + count]);
                }
            }
            consumed += count;
            self.offset = event.end;
        }
        self.pending.drain(..consumed);
        fields(content, reasoning)
    }

    /// Raw bytes resolved by this event, excluding a held partial marker.
    /// The writer acknowledges this count only after the event succeeds.
    pub(super) fn push_counted(&mut self, text: &str) -> (Map<String, Value>, usize) {
        let before = self.offset;
        let fields = self.push(text);
        (fields, self.offset - before)
    }

    pub(super) fn finish_counted(&mut self) -> (Map<String, Value>, usize) {
        let before = self.offset;
        let fields = self.finish();
        (fields, self.offset - before)
    }

    pub(super) fn finish(&mut self) -> Map<String, Value> {
        let text = std::mem::take(&mut self.pending);
        self.offset += text.len();
        if self.inside {
            fields(String::new(), text)
        } else {
            fields(text, String::new())
        }
    }
}

fn fields(content: String, reasoning: String) -> Map<String, Value> {
    let mut out = Map::new();
    if !content.is_empty() {
        out.insert("content".into(), content.into());
    }
    if !reasoning.is_empty() {
        out.insert("reasoning_content".into(), reasoning.into());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parser(inside: bool) -> Reasoning {
        Reasoning {
            markers: [
                (1000, "<think>", Kind::Start),
                (1001, "</think>", Kind::End),
                (1002, "<tool_call>", Kind::Tool),
            ]
            .into_iter()
            .map(|(id, text, kind)| Marker {
                id,
                text: text.into(),
                kind,
            })
            .collect(),
            events: Arc::new(Mutex::new(Events::default())),
            initial_inside: inside,
            inside,
            pending: String::new(),
            offset: 0,
            by_text: false,
        }
    }
    fn decode(ids: &[u32]) -> lloom_tensor::Result<String> {
        Ok(ids
            .iter()
            .map(|id| match id {
                1000 => "<think>".into(),
                1001 => "</think>".into(),
                1002 => "<tool_call>".into(),
                id => char::from_u32(*id).unwrap().to_string(),
            })
            .collect())
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
    #[test]
    fn a_slow_observer_decoder_does_not_hold_the_consumer_parser() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;
        let parser = Arc::new(parking_lot::Mutex::new(parser(true)));
        parser
            .lock()
            .observe(&[97, 1001, 65], "a</think>A", 0, &decode)
            .unwrap();
        let mut observer = Observer::new(&parser);
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let classifier = std::thread::spawn(move || {
            let released = parking_lot::Mutex::new(released);
            observer
                .observe(&[1000], "<think>", 10, &|ids| {
                    if ids.is_empty() {
                        entered.send(()).unwrap();
                        released
                            .lock()
                            .recv_timeout(Duration::from_secs(5))
                            .unwrap();
                    }
                    decode(ids)
                })
                .unwrap();
        });
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        let (sent, received) = mpsc::channel();
        let consumer = std::thread::spawn(move || {
            sent.send(parser.lock().push_counted("a</think>A")).unwrap();
        });
        let progressed = received.recv_timeout(Duration::from_millis(250));
        release.send(()).unwrap();
        classifier.join().unwrap();
        consumer.join().unwrap();
        let (fields, bytes) = progressed.expect("observer decoding held the consumer parser");
        assert_eq!(collect([fields]), ("A".into(), "a".into()));
        assert_eq!(bytes, 10);
    }

    #[test]
    fn accepted_fields_do_not_depend_on_the_writer_consumption_frontier() {
        for consumed in ["", "a<", "a</think>A"] {
            let mut parser = parser(true);
            parser
                .observe(&[97, 1001, 65], "a</think>A", 0, &decode)
                .unwrap();
            parser.push(consumed);
            assert_eq!(
                collect([parser.accepted_fields("a</think>A")]),
                ("A".into(), "a".into())
            );
            assert_eq!(
                collect([parser.accepted_fields("a</thi")]),
                ("".into(), "a</thi".into())
            );
        }
    }

    #[test]
    fn marker_descriptors_have_a_finite_error_boundary() {
        let parser = Arc::new(Mutex::new(parser(false)));
        let mut observer = Observer::new(&parser);
        for index in 0..MAX_MARKERS {
            observer
                .observe(&[1000], "<think>", index * 7, &decode)
                .unwrap();
        }
        let error = observer
            .observe(&[1000], "<think>", MAX_MARKERS * 7, &decode)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("response marker descriptor limit exceeded"),
            "{error}"
        );
        assert_eq!(parser.lock().events.lock().retained.len(), MAX_MARKERS);
        assert_eq!(parser.lock().events.lock().pending.len(), MAX_MARKERS);
    }

    #[test]
    fn all_fragment_boundaries_preserve_channels_and_partial_tags() {
        for (open, ids, cut, content, reason) in [
            (true, vec![97, 960, 1001, 65], 0, "A", "aπ"),
            (false, vec![1000, 97, 1001, 65], 0, "A", "a"),
            (true, vec![97, 1001], 3, "", "a</thi"),
            (true, vec![97, 1002, 123, 125], 0, "<tool_call>{}", "a"),
            (false, vec![65], 0, "A", ""),
            (true, vec![1001], 0, "", ""),
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
                        p.finish(),
                    ];
                    assert_eq!(
                        collect(parts),
                        (content.into(), reason.into()),
                        "{open} {text:?} {i} {j}"
                    );
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
            parts.push(p.finish());
            assert_eq!(
                collect(parts),
                if inside {
                    ("".into(), text.into())
                } else {
                    (text.into(), "".into())
                }
            );
        }
    }
    #[test]
    fn repeated_pending_decode_and_lagging_writer_do_not_duplicate_markers() {
        let mut p = parser(true);
        let ids = [97, 1001];
        let text = decode(&ids).unwrap();
        p.observe(&ids, &text, 0, &decode).unwrap();
        p.observe(&ids, &text, 0, &decode).unwrap();
        p.observe(&[65], "A", text.len(), &decode).unwrap();
        assert_eq!(
            collect([p.push(&(text + "A")), p.finish()]),
            ("A".into(), "a".into())
        );
    }
    #[test]
    fn real_byte_tokenizer_text_stream_utf8_stops_and_delayed_writer() {
        use lloom::testing::tiny::{tokenizer::tokenizer_json, TinyTokenizer};
        let tiny = TinyTokenizer::lfm2();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json(&tiny.added).as_bytes()).unwrap();
        let end = tokenizer.token_to_id("</think>").unwrap();
        let mut ids: Vec<u32> = "literal <think> π".bytes().map(u32::from).collect();
        ids.push(end);
        ids.extend("Answer".bytes().map(u32::from));
        for delayed in [false, true] {
            for stops in [vec![], vec!["/think>".to_owned()]] {
                let parser = std::sync::Arc::new(parking_lot::Mutex::new(parser(true)));
                // Match the real tokenizer's ids/strings, while preserving initial state.
                parser.lock().markers = vec![Marker {
                    id: end,
                    text: "</think>".into(),
                    kind: Kind::End,
                }];
                let mut stream = crate::stream::TextStream::default();
                let mut from = 0;
                let mut observer =
                    |ids: &[u32],
                     text: &str,
                     from,
                     decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>| {
                        parser.lock().observe(ids, text, from, decode)
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
                let mut parts = Vec::new();
                for id in &ids {
                    let (text, stopped) = stream.publish(&[*id], &mut decoder, &stops).unwrap();
                    if let Some(text) = text {
                        if delayed {
                            published.push(text);
                        } else {
                            parts.push(parser.lock().push(&text));
                        }
                    }
                    if stopped {
                        break;
                    }
                }
                let (text, _) = stream.flush(&mut decoder, &stops).unwrap();
                if let Some(text) = text {
                    published.push(text);
                }
                for text in published {
                    parts.push(parser.lock().push(&text));
                }
                parts.push(parser.lock().finish());
                let expected = if stops.is_empty() {
                    ("Answer", "literal <think> π")
                } else {
                    ("", "literal <think> π<")
                };
                assert_eq!(
                    collect(parts),
                    (expected.0.into(), expected.1.into()),
                    "delayed={delayed} stops={stops:?}"
                );
            }
        }
    }

    // Exercise the production byte decoder's from/pending rule, not a fake
    // monotone counter: a marker plus the first byte of pi must stay pending
    // through repeated observations until the second UTF-8 byte arrives.
    fn pending_suffix_with_lagging_history(drained_groups: usize) {
        use lloom::testing::tiny::{tokenizer::tokenizer_json, TinyTokenizer};
        use parking_lot::Mutex;
        let tiny = TinyTokenizer::lfm2();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json(&tiny.added).as_bytes()).unwrap();
        let start = tokenizer.token_to_id("<think>").unwrap();
        let end = tokenizer.token_to_id("</think>").unwrap();
        let mut p = parser(false);
        p.markers = vec![
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
        ];
        let p = Mutex::new(p);
        let mut stream = crate::stream::TextStream::default();
        let mut from = 0;
        let mut published = VecDeque::new();
        let mut parts = Vec::new();
        let mut raw_credited = 0;
        let publish = |ids: &[u32], stream: &mut crate::stream::TextStream, from: &mut usize| {
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
            stream.publish(ids, &mut decode, &[]).unwrap()
        };
        const GROUPS: usize = 32;
        for _ in 0..GROUPS {
            let (text, stopped) = publish(&[start, 97, end, 65], &mut stream, &mut from);
            assert!(!stopped);
            published.push_back(text.unwrap());
        }
        assert_eq!(
            p.lock().events.lock().pending.len(),
            GROUPS * 2,
            "historical events must really be retained"
        );
        for _ in 0..drained_groups {
            let (fields, resolved) = p.lock().push_counted(&published.pop_front().unwrap());
            raw_credited += resolved;
            parts.push(fields);
        }
        assert_eq!(
            p.lock().events.lock().pending.len(),
            (GROUPS - drained_groups) * 2
        );
        // Spell the apparent markers with ordinary byte ids, not added ids.
        // They must remain literal content despite identical strings.
        let literal = " literal <think> </think> ";
        for byte in literal.bytes() {
            let (text, stopped) = publish(&[u32::from(byte)], &mut stream, &mut from);
            assert!(!stopped);
            if let Some(text) = text {
                published.push_back(text);
            }
        }
        let (text, stopped) = publish(&[start, 98], &mut stream, &mut from);
        assert!(!stopped);
        published.push_back(text.unwrap());
        let pending_from = from;
        let prefix: Vec<_> = p
            .lock()
            .events
            .lock()
            .pending
            .iter()
            .map(|e| (e.start, e.end))
            .collect();
        assert!(prefix.iter().all(|&(start, _)| start < pending_from));
        // Real first UTF-8 byte: observed text contains an additive genuine
        // marker but ends in U+FFFD, so TextStream publishes nothing.
        for ids in [&[end, 0xcf][..], &[][..], &[][..]] {
            let (text, stopped) = publish(ids, &mut stream, &mut from);
            assert_eq!(text, None);
            assert!(!stopped);
            assert_eq!(
                from, pending_from,
                "incomplete run must be replaced at the same origin"
            );
            let parser = p.lock();
            let guard = parser.events.lock();
            let actual_prefix: Vec<_> = guard
                .pending
                .iter()
                .take(prefix.len())
                .map(|e| (e.start, e.end))
                .collect();
            assert_eq!(actual_prefix, prefix, "historical prefix changed");
            assert_eq!(
                guard.pending.len(),
                prefix.len() + 1,
                "pending marker duplicated or history lost"
            );
            assert_eq!(guard.pending.back().unwrap().start, pending_from);
        }
        let (text, stopped) = publish(&[0x80], &mut stream, &mut from);
        assert!(!stopped);
        assert_eq!(text.as_deref(), Some("</think>π"));
        assert_eq!(from, pending_from + "</think>π".len());
        published.push_back(text.unwrap());
        assert_eq!(p.lock().events.lock().pending.len(), prefix.len() + 1);
        let tail = " tail <think>";
        for byte in tail.bytes() {
            let (text, stopped) = publish(&[u32::from(byte)], &mut stream, &mut from);
            assert!(!stopped);
            if let Some(text) = text {
                published.push_back(text);
            }
        }
        // No incomplete bytes remain; finish preserves the same raw text.
        let (raw, _) = stream.finish(&[]);
        for text in published {
            let (fields, resolved) = p.lock().push_counted(&text);
            raw_credited += resolved;
            parts.push(fields);
        }
        let (fields, resolved) = p.lock().finish_counted();
        raw_credited += resolved;
        parts.push(fields);
        assert_eq!(raw_credited, raw.len());
        assert!(p.lock().events.lock().pending.is_empty());
        assert_eq!(
            collect(parts),
            (
                format!("{}{literal}π{tail}", "A".repeat(GROUPS)),
                format!("{}b", "a".repeat(GROUPS))
            )
        );
    }

    #[test]
    fn lagging_history_keeps_only_one_repeated_incomplete_utf8_marker_suffix() {
        pending_suffix_with_lagging_history(0);
    }

    #[test]
    fn front_consumption_preserves_pending_suffix_and_literal_marker_distinction() {
        pending_suffix_with_lagging_history(8);
    }
}
