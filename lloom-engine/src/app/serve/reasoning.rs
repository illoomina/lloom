//! Chat wire fields: reasoning in `reasoning_content`, the answer in `content`.
//! `reasoning_in_content` opts out. Raw engine history and non-chat
//! completions remain unchanged.
//!
//! One state machine per response routes the text as it is published into
//! two buffers, `content` and `reasoning` (decision 85, `docs/sessions.md`
//! §6). It transitions on the markers' token ids, from the decoder's typed
//! observations, so an ordinary token spelling a marker's literal stays text;
//! a family without observations places the markers from their literals and
//! holds back a literal's prefix across publications. The machine sits at the
//! published frontier: a re-observed suffix replaces only marker spans past
//! it, and the only text it holds is a marker cut at that frontier. The
//! stored session reply and the socket's deltas are both taken from the two
//! buffers.
use crate::app::served::Served;
use anyhow::Result;
use serde_json::{Map, Value};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A marker's raw byte range in the reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    kind: Kind,
}

/// One publication's output: where its text begins in each buffer, and the
/// raw bytes it resolved (routed text plus consumed markers; a marker held at
/// the frontier resolves when it completes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Routed {
    pub content: usize,
    pub reasoning: usize,
    pub resolved: usize,
}

pub(super) struct Reasoning {
    markers: Vec<Marker>,
    inside: bool,
    content: String,
    reasoning: String,
    /// Typed marker spans past the published frontier, in order. Bounded by
    /// the unpublished window: a stop string's held prefix and one
    /// incomplete UTF-8 run.
    ahead: VecDeque<Span>,
    /// Published bytes not yet routed: a marker, or in text mode a marker
    /// literal's prefix, cut at the published frontier.
    held: String,
    /// Raw bytes routed; `held` starts here.
    routed: usize,
    /// Markers placed from their literals, because the family supplies no
    /// decode observations (`Served::observes_decode`).
    by_text: bool,
    transitions: u64,
    #[cfg(any(test, debug_assertions, feature = "profile"))]
    check: oracle::Check,
}

impl Reasoning {
    /// The splitter a chat request runs with: `None` where the request asked
    /// for the reply in `content` (`reasoning_in_content`) or the template
    /// has no reasoning markers to split on.
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
        Ok(Some(Self::new(
            markers,
            tags.open_at_end(prompt).is_some(),
            !engine.observes_decode(),
        )))
    }

    fn new(markers: Vec<Marker>, inside: bool, by_text: bool) -> Self {
        Self {
            #[cfg(any(test, debug_assertions, feature = "profile"))]
            check: oracle::Check::new(&markers, inside, by_text),
            markers,
            inside,
            content: String::new(),
            reasoning: String::new(),
            ahead: VecDeque::new(),
            held: String::new(),
            routed: 0,
            by_text,
            transitions: 0,
        }
    }

    /// A detokenizer attempt over a pending run starting at raw offset
    /// `from`: classify its marker ids, replacing any spans an earlier
    /// attempt placed from `from` on. Text mode places nothing here.
    pub(super) fn observe(
        &mut self,
        ids: &[u32],
        text: &str,
        from: usize,
        decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>,
    ) -> lloom_tensor::Result<()> {
        if self.by_text {
            return Ok(());
        }
        debug_assert!(
            from >= self.published(),
            "an observed run starts behind the published frontier"
        );
        while self.ahead.back().is_some_and(|span| span.start >= from) {
            self.ahead.pop_back();
        }
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
            self.ahead.push_back(Span {
                start: from + before.len(),
                end: from + through.len(),
                kind: marker.kind,
            });
        }
        #[cfg(any(test, debug_assertions, feature = "profile"))]
        self.check.observe(ids, text, from, decode);
        Ok(())
    }

    /// Route a published fragment into the two buffers.
    pub(super) fn publish(&mut self, text: &str) -> Routed {
        let out = Routed {
            content: self.content.len(),
            reasoning: self.reasoning.len(),
            resolved: self.routed,
        };
        self.held.push_str(text);
        let done = if self.by_text {
            self.route_by_text()
        } else {
            self.route_typed()
        };
        self.held.drain(..done);
        self.routed += done;
        let out = Routed {
            resolved: self.routed - out.resolved,
            ..out
        };
        #[cfg(any(test, debug_assertions, feature = "profile"))]
        self.check.publish(
            text,
            &self.content[out.content..],
            &self.reasoning[out.reasoning..],
            out.resolved,
        );
        out
    }

    /// The run ended: a marker still held was cut short (a stop string, a
    /// refusal) and its bytes are text in the block they were in.
    pub(super) fn finish(&mut self) -> Routed {
        let out = Routed {
            content: self.content.len(),
            reasoning: self.reasoning.len(),
            resolved: self.held.len(),
        };
        self.route(0, self.held.len());
        self.routed += self.held.len();
        self.held.clear();
        self.ahead.clear();
        #[cfg(any(test, debug_assertions, feature = "profile"))]
        {
            self.check.finish(
                &self.content[out.content..],
                &self.reasoning[out.reasoning..],
                out.resolved,
            );
            self.check
                .settle(&self.content, &self.reasoning, self.transitions);
        }
        out
    }

    /// The typed spans decide: route up to each marker, transition through
    /// it, and stop at one the frontier cuts. Returns the bytes of `held`
    /// routed.
    fn route_typed(&mut self) -> usize {
        let base = self.routed;
        let end = base + self.held.len();
        let mut at = base;
        while let Some(&span) = self.ahead.front() {
            if span.start >= end {
                break;
            }
            if span.start < at {
                self.ahead.pop_front();
                continue;
            }
            self.route(at - base, span.start - base);
            at = span.start;
            if span.end > end {
                return at - base;
            }
            self.ahead.pop_front();
            self.transition(span.kind, span.start - base, span.end - base);
            at = span.end;
        }
        self.route(at - base, end - base);
        end - base
    }

    /// The literals decide: every complete marker literal in `held`, then
    /// hold back the longest suffix that begins one.
    fn route_by_text(&mut self) -> usize {
        let mut at = 0;
        loop {
            let mut best: Option<(usize, usize, Kind)> = None;
            for marker in &self.markers {
                if let Some(found) = self.held[at..].find(marker.text.as_str()) {
                    let found = at + found;
                    if best.is_none_or(|b| found < b.0) {
                        best = Some((found, marker.text.len(), marker.kind));
                    }
                }
            }
            let Some((found, len, kind)) = best else {
                break;
            };
            self.route(at, found);
            self.transition(kind, found, found + len);
            at = found + len;
        }
        let tail = &self.held[at..];
        let mut hold = 0;
        for marker in &self.markers {
            for k in 1..marker.text.len().min(tail.len() + 1) {
                if marker.text.is_char_boundary(k) && tail.ends_with(&marker.text[..k]) {
                    hold = hold.max(k);
                }
            }
        }
        let until = self.held.len() - hold;
        self.route(at, until);
        until
    }

    /// `held[from..to]` as text, in the block the machine is in.
    fn route(&mut self, from: usize, to: usize) {
        let text = &self.held[from..to];
        if self.inside {
            self.reasoning.push_str(text);
        } else {
            self.content.push_str(text);
        }
    }

    /// A marker at `held[from..to]`. A start inside the block, or an end or
    /// a tool call outside it, is not a transition and stays text.
    fn transition(&mut self, kind: Kind, from: usize, to: usize) {
        match kind {
            Kind::Start if !self.inside => self.inside = true,
            Kind::End if self.inside => self.inside = false,
            Kind::Tool if self.inside => {
                self.inside = false;
                self.content.push_str(&self.held[from..to]);
            }
            _ => return self.route(from, to),
        }
        self.transitions += 1;
    }

    /// The text one publication added to each buffer.
    pub(super) fn piece(&self, r: Routed) -> (&str, &str) {
        (&self.content[r.content..], &self.reasoning[r.reasoning..])
    }

    /// Raw bytes published so far, a held marker included.
    pub(super) fn published(&self) -> usize {
        self.routed + self.held.len()
    }

    /// Marker transitions taken: a reply that keeps opening and closing its
    /// block shows here.
    pub(super) fn transitions(&self) -> u64 {
        self.transitions
    }

    /// The accepted reply as the session keeps it.
    pub(super) fn fields(&self) -> Map<String, Value> {
        fields(&self.content, &self.reasoning)
    }

    /// [`Self::publish`], as wire fields.
    #[cfg(test)]
    pub(super) fn push(&mut self, text: &str) -> Map<String, Value> {
        self.push_counted(text).0
    }

    #[cfg(test)]
    pub(super) fn push_counted(&mut self, text: &str) -> (Map<String, Value>, usize) {
        let r = self.publish(text);
        let (content, reasoning) = self.piece(r);
        (fields(content, reasoning), r.resolved)
    }

    /// [`Self::finish`], as wire fields.
    #[cfg(test)]
    pub(super) fn finish_fields(&mut self) -> Map<String, Value> {
        self.finish_counted().0
    }

    #[cfg(test)]
    pub(super) fn finish_counted(&mut self) -> (Map<String, Value>, usize) {
        let r = self.finish();
        let (content, reasoning) = self.piece(r);
        (fields(content, reasoning), r.resolved)
    }

    /// A content-only machine with `capacity` reserved in its buffers and
    /// its oracle off, for a test of its own allocations; and a reader of its
    /// content buffer.
    #[cfg(test)]
    pub(super) fn sized_for_test(capacity: usize) -> (Self, fn(&Self) -> &str) {
        let mut machine = Self::new(Vec::new(), false, false);
        machine.check.enabled = false;
        machine.content.reserve(capacity);
        machine.held.reserve(64);
        (machine, |m| m.content.as_str())
    }

    /// Typed spans waiting past the frontier and bytes held at it: the
    /// machine's whole retained marker state.
    #[cfg(test)]
    fn retained(&self) -> (usize, usize) {
        (self.ahead.len(), self.held.len())
    }

    /// The differential oracle's verdict so far: `None` while the event
    /// list agrees with the machine.
    #[cfg(test)]
    pub(super) fn oracle_mismatch(&self) -> Option<&str> {
        self.check.mismatch()
    }
}

/// The wire fields a piece of text fills: each only when nonempty.
pub(super) fn fields(content: &str, reasoning: &str) -> Map<String, Value> {
    let mut out = Map::new();
    if !content.is_empty() {
        out.insert("content".into(), content.into());
    }
    if !reasoning.is_empty() {
        out.insert("reasoning_content".into(), reasoning.into());
    }
    out
}

/// The retired event-list splitter, kept as a differential oracle in debug
/// and profile builds (decisions 50, 85): every marker it places is retained
/// for the response's life, and the accepted fields are rebuilt from those
/// events. The machine is checked against it on every publication and at
/// settlement; shipping builds compile it out.
#[cfg(any(test, debug_assertions, feature = "profile"))]
mod oracle {
    use super::{Kind, Marker, Span};
    use std::collections::VecDeque;

    #[derive(Clone)]
    pub(super) struct Events {
        markers: Vec<Marker>,
        pending: VecDeque<Span>,
        retained: Vec<Span>,
        initial_inside: bool,
        inside: bool,
        text: String,
        offset: usize,
        by_text: bool,
        transitions: u64,
    }

    impl Events {
        pub(super) fn new(markers: &[Marker], inside: bool, by_text: bool) -> Self {
            Self {
                markers: markers.to_vec(),
                pending: VecDeque::new(),
                retained: Vec::new(),
                initial_inside: inside,
                inside,
                text: String::new(),
                offset: 0,
                by_text,
                transitions: 0,
            }
        }

        pub(super) fn observe(
            &mut self,
            ids: &[u32],
            text: &str,
            from: usize,
            decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>,
        ) -> lloom_tensor::Result<()> {
            let mut placed = Vec::new();
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
                placed.push(Span {
                    start: from + before.len(),
                    end: from + through.len(),
                    kind: marker.kind,
                });
            }
            while self.pending.back().is_some_and(|e| e.start >= from) {
                self.pending.pop_back();
            }
            while self.retained.last().is_some_and(|e| e.start >= from) {
                self.retained.pop();
            }
            self.pending.extend(placed.iter().copied());
            self.retained.extend(placed);
            Ok(())
        }

        fn place_from_text(&mut self) -> usize {
            let base = self.offset;
            let mut pos = 0;
            loop {
                let mut best: Option<(usize, usize, Kind)> = None;
                for marker in &self.markers {
                    if let Some(at) = self.text[pos..].find(marker.text.as_str()) {
                        let at = pos + at;
                        if best.is_none_or(|b: (usize, usize, Kind)| at < b.0) {
                            best = Some((at, marker.text.len(), marker.kind));
                        }
                    }
                }
                let Some((at, len, kind)) = best else { break };
                self.pending.push_back(Span {
                    start: base + at,
                    end: base + at + len,
                    kind,
                });
                pos = at + len;
            }
            let tail = &self.text[pos..];
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

        pub(super) fn push(&mut self, text: &str) -> (String, String, usize) {
            let before = self.offset;
            self.text.push_str(text);
            let hold = if self.by_text {
                self.place_from_text()
            } else {
                0
            };
            let (mut content, mut reasoning) = (String::new(), String::new());
            let end = self.offset + self.text.len() - hold;
            let mut consumed = 0;
            loop {
                let event = self.pending.front().copied();
                let until = event.map_or(end, |event| event.start.min(end));
                let count = until - self.offset;
                let out = if self.inside {
                    &mut reasoning
                } else {
                    &mut content
                };
                out.push_str(&self.text[consumed..consumed + count]);
                consumed += count;
                self.offset = until;
                let Some(event) = event else {
                    break;
                };
                if event.end > end {
                    break;
                }
                self.pending.pop_front();
                let count = event.end - self.offset;
                match event.kind {
                    Kind::Start if !self.inside => {
                        self.inside = true;
                        self.transitions += 1;
                    }
                    Kind::End if self.inside => {
                        self.inside = false;
                        self.transitions += 1;
                    }
                    Kind::Tool if self.inside => {
                        self.inside = false;
                        self.transitions += 1;
                        content.push_str(&self.text[consumed..consumed + count]);
                    }
                    _ => {
                        let out = if self.inside {
                            &mut reasoning
                        } else {
                            &mut content
                        };
                        out.push_str(&self.text[consumed..consumed + count]);
                    }
                }
                consumed += count;
                self.offset = event.end;
            }
            self.text.drain(..consumed);
            (content, reasoning, self.offset - before)
        }

        pub(super) fn finish(&mut self) -> (String, String, usize) {
            let text = std::mem::take(&mut self.text);
            self.offset += text.len();
            let resolved = text.len();
            if self.inside {
                (String::new(), text, resolved)
            } else {
                (text, String::new(), resolved)
            }
        }

        /// The accepted fields rebuilt from the retained events over the
        /// whole published text, as the retired settlement path did.
        pub(super) fn accepted(&self, text: &str) -> (String, String, u64) {
            let mut replay = Self {
                pending: self.retained.iter().copied().collect(),
                retained: Vec::new(),
                inside: self.initial_inside,
                text: String::new(),
                offset: 0,
                transitions: 0,
                ..self.clone()
            };
            let (mut content, mut reasoning, _) = replay.push(text);
            let (c, r, _) = replay.finish();
            content.push_str(&c);
            reasoning.push_str(&r);
            (content, reasoning, replay.transitions)
        }

        pub(super) fn transitions(&self) -> u64 {
            self.transitions
        }
    }

    /// The machine's inputs replayed through [`Events`], and the first
    /// disagreement.
    pub(super) struct Check {
        events: Events,
        published: String,
        mismatch: Option<String>,
        /// Off only in a test that measures the machine's own allocations.
        pub(super) enabled: bool,
    }

    impl Check {
        pub(super) fn new(markers: &[Marker], inside: bool, by_text: bool) -> Self {
            Self {
                events: Events::new(markers, inside, by_text),
                published: String::new(),
                mismatch: None,
                enabled: true,
            }
        }

        #[cfg(test)]
        pub(super) fn mismatch(&self) -> Option<&str> {
            self.mismatch.as_deref()
        }

        fn differ(&mut self, what: String) {
            if self.mismatch.is_none() {
                self.mismatch = Some(what);
            }
        }

        pub(super) fn observe(
            &mut self,
            ids: &[u32],
            text: &str,
            from: usize,
            decode: &dyn Fn(&[u32]) -> lloom_tensor::Result<String>,
        ) {
            if !self.enabled {
                return;
            }
            if let Err(e) = self.events.observe(ids, text, from, decode) {
                self.differ(format!(
                    "the event list refused an observation the machine took: {e}"
                ));
            }
        }

        pub(super) fn publish(
            &mut self,
            text: &str,
            content: &str,
            reasoning: &str,
            resolved: usize,
        ) {
            if !self.enabled {
                return;
            }
            self.published.push_str(text);
            let want = self.events.push(text);
            if (want.0.as_str(), want.1.as_str(), want.2) != (content, reasoning, resolved) {
                self.differ(format!(
                    "publication {text:?}: machine ({content:?}, {reasoning:?}, {resolved}) \
                     against events {want:?}"
                ));
            }
        }

        pub(super) fn finish(&mut self, content: &str, reasoning: &str, resolved: usize) {
            if !self.enabled {
                return;
            }
            let want = self.events.finish();
            if (want.0.as_str(), want.1.as_str(), want.2) != (content, reasoning, resolved) {
                self.differ(format!(
                    "finish: machine ({content:?}, {reasoning:?}, {resolved}) against events {want:?}"
                ));
            }
        }

        /// The settled buffers against the fields the event list rebuilds,
        /// and the transition counts. A disagreement panics in a debug build
        /// and is reported in a profile build.
        pub(super) fn settle(&mut self, content: &str, reasoning: &str, transitions: u64) {
            if !self.enabled {
                return;
            }
            let (c, r, replayed) = self.events.accepted(&self.published);
            if (c.as_str(), r.as_str()) != (content, reasoning) {
                self.differ(format!(
                    "settled: machine ({content:?}, {reasoning:?}) against rebuilt ({c:?}, {r:?})"
                ));
            }
            if (self.events.transitions(), replayed) != (transitions, transitions) {
                self.differ(format!(
                    "transitions: machine {transitions}, events {} streamed / {replayed} rebuilt",
                    self.events.transitions()
                ));
            }
            if let Some(what) = &self.mismatch {
                if cfg!(debug_assertions) {
                    panic!("reasoning machine disagrees with its event-list oracle: {what}");
                }
                eprintln!("[reasoning: oracle mismatch: {what}]");
            }
        }
    }
}

#[cfg(test)]
#[path = "reasoning_tests.rs"]
mod tests;
