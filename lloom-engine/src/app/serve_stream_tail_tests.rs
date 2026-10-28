//! CPU-only transport/finalization controls. Scripted sampled ids and a byte
//! tokenizer exercise the real observer, routing, writer, and reply-log paths;
//! this is not a model/numerical/native-device acceptance test.
use super::*;
use crate::app::common::{DecodeTiming, PrefillTiming, Text};
use crate::app::family::{Capabilities, LFM2_CAPABILITIES};
use crate::app::served::DecodeObserver;
use crate::app::template::{Rendered, Template, ThinkTags};
use crate::ReadbackArm;
use lloom::load_warning::LoadWarning;
use lloom::testing::tiny::{tokenizer::tokenizer_json, TinyTokenizer};
use lloom_tensor::Device;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Case {
    PartialTail,
    CompleteMarker,
    SplitMarker,
}
#[derive(Default)]
struct Wire {
    committed: Vec<u8>,
    events: Vec<Value>,
    first_text: bool,
    failures: usize,
}
#[derive(Clone, Default)]
struct Progress(Arc<(ParkingMutex<Wire>, Condvar)>);
impl Progress {
    fn await_first_text(&self, sink: &mut (dyn FnMut(&str) -> bool + Send)) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            // DeltaBuffer's try-lock can leave the first push in its producer
            // spill. An empty hand-over carries it over without changing text.
            assert!(sink(""));
            let (lock, changed) = &*self.0;
            let mut wire = lock.lock();
            if wire.first_text {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "first routed event never committed"
            );
            changed.wait_for(&mut wire, Duration::from_millis(2));
        }
    }
}

// Fail exactly the chosen completed event, not an arbitrary OS-sized write.
// A successful earlier event is independently witnessed before SplitMarker's
// next fragment is offered; no sleep or writer-thread scheduling assumption.
struct Writer {
    progress: Progress,
    pending: Vec<u8>,
    fail_delta: Option<Value>,
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        let text = std::str::from_utf8(&self.pending).unwrap();
        let event = text.find("data: ").and_then(|start| {
            let data = &text[start + "data: ".len()..];
            let end = data.find("\n\n").unwrap();
            let data = &data[..end];
            (data != "[DONE]").then(|| serde_json::from_str::<Value>(data).unwrap())
        });
        let (lock, changed) = &*self.progress.0;
        let mut wire = lock.lock();
        if let (Some(event), Some(fail)) = (&event, &self.fail_delta) {
            if &event["choices"][0]["delta"] == fail {
                wire.failures += 1;
                self.pending.clear();
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "scripted tail write",
                ));
            }
        }
        if let Some(event) = event {
            if event["choices"][0]["delta"]["reasoning_content"] == "a" {
                wire.first_text = true;
                changed.notify_all();
            }
            wire.events.push(event);
        }
        wire.committed.append(&mut self.pending);
        Ok(())
    }
}

struct Scripted {
    device: Device,
    text: Text,
    case: Case,
    progress: Progress,
    /// Whether this stub reports typed decode observations (it supplies
    /// them; a family on the default seam reports `false`).
    observes: bool,
}
impl Scripted {
    fn new(case: Case, progress: Progress) -> Self {
        let tiny = TinyTokenizer::lfm2();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json(&tiny.added).as_bytes()).unwrap();
        Self {
            device: Device::Cpu,
            text: Text {
                tokenizer,
                template: Some(Template::compile(tiny.chat_template).unwrap()),
                bos_token: "<|startoftext|>".into(),
                bos_id: None,
                eos_ids: vec![],
                seed_tokens: vec![97],
                think: Default::default(),
                gen_prompts: Default::default(),
            },
            case,
            progress,
            observes: true,
        }
    }
}
impl Served for Scripted {
    fn observes_decode(&self) -> bool {
        self.observes
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn model_name(&self) -> &str {
        "scripted-tail"
    }
    fn kv_capacity(&self) -> usize {
        64
    }
    fn readback(&self) -> ReadbackArm {
        ReadbackArm::Blocking
    }
    fn prefill_tile(&self) -> usize {
        1
    }
    fn batch(&self) -> usize {
        1
    }
    fn eos_ids(&self) -> &[u32] {
        &[]
    }
    fn bos_id(&self) -> Option<u32> {
        None
    }
    fn health(&self) -> Value {
        Value::Null
    }
    fn describe(&self) -> String {
        String::new()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            family: "lfm2-gguf",
            ..LFM2_CAPABILITIES
        }
    }
    fn load_warnings(&self) -> Vec<LoadWarning> {
        vec![]
    }
    fn note_load_warnings(&mut self, _: Vec<LoadWarning>) {}
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        Ok(self
            .text
            .tokenizer
            .encode(text, special)
            .unwrap()
            .get_ids()
            .to_vec())
    }
    fn token_str(&self, id: u32) -> String {
        self.text.tokenizer.decode(&[id], false).unwrap()
    }
    fn render_checked(
        &self,
        messages: &Value,
        kwargs: &serde_json::Map<String, Value>,
        checked: &[&str],
    ) -> Result<Rendered> {
        self.text.template.as_ref().unwrap().render_checked(
            messages,
            &self.text.bos_token,
            kwargs,
            checked,
        )
    }
    fn template_reads(&self, name: &str) -> Option<bool> {
        self.text.template.as_ref()?.reads(name)
    }
    fn think_tags(&self) -> Option<ThinkTags> {
        self.text.think_tags().cloned()
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
        panic!("typed observer must be used")
    }
    fn complete_observed(
        &self,
        _: &[u32],
        _: &[DecodedImage],
        _: &DecodeParams,
        sink: &mut (dyn FnMut(&str) -> bool + Send),
        observe: &mut DecodeObserver<'_>,
    ) -> Result<Decoded> {
        let end = self.text.tokenizer.token_to_id("</think>").unwrap();
        let mut stream = crate::stream::TextStream::default();
        let mut from = 0;
        let mut decoder = |ids: &[u32]| {
            crate::app::causal_loaded::observed_decode(
                &self.text.tokenizer,
                ids,
                &mut from,
                observe,
            )
        };
        let stops = match self.case {
            Case::PartialTail => vec!["nk>".into()],
            _ => vec![],
        };
        let ids = match self.case {
            Case::PartialTail => vec![97, end],
            _ => vec![97, end, 88],
        };
        let mut stopped = false;
        let mut batched = String::new();
        for id in ids {
            let (fragment, hit) = stream.publish(&[id], &mut decoder, &stops)?;
            if let Some(fragment) = fragment {
                if matches!(self.case, Case::SplitMarker) {
                    batched.push_str(&fragment);
                } else {
                    assert!(sink(&fragment));
                }
            }
            if hit {
                stopped = true;
                break;
            }
        }
        if !stopped {
            let (fragment, _) = stream.flush(&mut decoder, &stops)?;
            if let Some(fragment) = fragment {
                if matches!(self.case, Case::SplitMarker) {
                    batched.push_str(&fragment);
                } else {
                    assert!(sink(&fragment));
                }
            }
        }
        if matches!(self.case, Case::SplitMarker) {
            // Adversarial transport batching at the Served seam, after the
            // genuine tokenizer/observer produced this exact raw text. It
            // forces an earlier held prefix, not a different sampled token.
            assert_eq!(batched, "a</think>X");
            assert!(sink("a<"));
            self.progress.await_first_text(sink);
            assert!(sink("/think>X"));
        }
        let (text, tokens) = stream.finish(&stops);
        Ok(Decoded {
            text,
            tokens,
            stop: if stopped {
                Stop::StopString
            } else {
                Stop::Length
            },
            reuse: lloom::nn::ReuseAction::Fresh,
            reused: 0,
            prefilled: 0,
            turn: Default::default(),
            checkpoint: Default::default(),
            checkpoint_at: None,
            readback: ReadbackArm::Blocking,
            sampling: lloom::generation::Sampling::ArgMax,
            readback_note: None,
            logits_sha256: None,
            stats: Default::default(),
            thinking: None,
            logprobs: None,
            pin: Default::default(),
        })
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

fn run(
    case: Case,
    fail: Option<Value>,
    raw: &str,
    expected_written: usize,
) -> (String, Vec<Value>) {
    let progress = Progress::default();
    let engine = Box::new(Scripted::new(case, progress.clone()));
    let tags = engine
        .think_tags()
        .expect("actual template has reasoning tags");
    let end = tags.end;
    let front = Front::new(engine.as_ref(), Default::default());
    let origin = front.origin(
        &json!([]),
        &Default::default(),
        Some(Thinking {
            on: true,
            by: super::super::thinking::By::Template,
        }),
    );
    let shared = Shared {
        engine,
        front,
        decisions: super::super::decisions::Decisions::Refused("test".into()),
        turns: turns::Turns::default(),
        decision_pack: false,
        aborts: Mutex::new(HashSet::new()),
        abort_epoch: AtomicU64::new(0),
        server: Server::http("127.0.0.1:0").unwrap(),
        model_name: "scripted-tail".into(),
        sampling: Default::default(),
        reuse: ParkingMutex::new(Default::default()),
        sessions: super::test_sessions(),
        activity: Arc::new(keep_warm::Activity::new(0)),
        keep_warm: keep_warm::Policy::Off { why: "test".into() },
        keep_warm_pressure_file: None,
    };
    let run = StreamingRun {
        kind: StreamKind::Chat,
        prompt_ids: vec![tags.start],
        facts: Default::default(),
        images: vec![],
        params: DecodeParams::default(),
        id: "tail".into(),
        model: "scripted-tail".into(),
        record: Some(origin),
        return_ids: false,
        reasoning_in_content: false,
        session: None,
        first: FirstToken::default(),
    };
    let failed = fail.is_some();
    let result = stream_response_with_writer(&shared, run, || {
        Ok(ChunkedWriter {
            w: Box::new(Writer {
                progress: progress.clone(),
                pending: vec![],
                fail_delta: fail,
            }),
        })
    });
    let (lock, _) = &*progress.0;
    let wire = lock.lock();
    let bytes = String::from_utf8(wire.committed.clone()).unwrap();
    let frontier = if failed {
        assert_eq!(wire.failures, 1, "selected failure was not reached");
        assert!(!bytes.contains("[DONE]"));
        let error = result.unwrap_err().to_string();
        assert!(error.contains("scripted tail write"), "{error}");
        serde_json::from_str::<Value>(
            error
                .split_once("; frontier ")
                .expect("frontier must be reported after I/O failure")
                .1,
        )
        .unwrap()
    } else {
        result.unwrap();
        assert!(bytes.contains("[DONE]"));
        wire.events
            .iter()
            .find_map(|event| event.pointer("/timings/frontier"))
            .expect("success trailer frontier")
            .clone()
    };
    assert_eq!(frontier["written_bytes"], expected_written);
    assert_eq!(frontier["write_failed"], failed);
    assert_eq!(frontier["logged"], true);
    assert_eq!(frontier["delivered_bytes"], raw.len());
    let reply = shared
        .front
        .reply(raw)
        .expect("actual production record_frontier must retain accepted history");
    assert_eq!(reply.text, raw);
    if matches!(case, Case::PartialTail) {
        assert_eq!(reply.ids, vec![97]);
        assert_eq!(reply.partial, 5);
        assert_eq!(frontier["accepted"], 1);
    } else {
        assert_eq!(reply.ids, vec![97, end, 88]);
        assert_eq!(reply.partial, 0);
        assert_eq!(frontier["accepted"], 3);
    }
    (bytes, wire.events.clone())
}

#[test]
fn partial_tail_success_credits_only_after_emission_and_records_history() {
    let (_, events) = run(Case::PartialTail, None, "a</thi", 6);
    let reasoning: String = events
        .iter()
        .filter_map(|e| {
            e.pointer("/choices/0/delta/reasoning_content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(reasoning, "a</thi");
}
#[test]
fn partial_tail_failure_keeps_history_without_credit_or_done() {
    let (_, events) = run(
        Case::PartialTail,
        Some(json!({"reasoning_content":"</thi"})),
        "a</thi",
        1,
    );
    assert!(events.iter().all(|e| e
        .pointer("/choices/0/finish_reason")
        .is_none_or(Value::is_null)));
}
#[test]
fn earlier_held_prefix_is_not_credited_when_its_later_event_fails() {
    run(
        Case::SplitMarker,
        Some(json!({"content":"X"})),
        "a</think>X",
        1,
    );
}
#[test]
fn complete_marker_counts_resolved_raw_bytes_not_serialized_field_bytes() {
    let (_, events) = run(Case::CompleteMarker, None, "a</think>X", 10);
    let reasoning: String = events
        .iter()
        .filter_map(|e| {
            e.pointer("/choices/0/delta/reasoning_content")
                .and_then(Value::as_str)
        })
        .collect();
    let content: String = events
        .iter()
        .filter_map(|e| {
            e.pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(reasoning, "a");
    assert_eq!(content, "X");
}

#[test]
fn ordered_control_flush_preserves_held_prefix_credit_on_later_failure() {
    let progress = Progress::default();
    let engine = Scripted::new(Case::CompleteMarker, progress.clone());
    let tags = engine.think_tags().unwrap();
    let mut parser = reasoning::Reasoning::for_chat(&engine, &[tags.start], false)
        .unwrap()
        .unwrap();
    let ids = [97, tags.end, 88];
    let decode = |ids: &[u32]| {
        engine
            .text
            .tokenizer
            .decode(ids, false)
            .map_err(|error| lloom_tensor::Error::msg(error.to_string()))
    };
    parser
        .observe(&ids, &decode(&ids).unwrap(), 0, &decode)
        .unwrap();
    let parser = Arc::new(ParkingMutex::new(parser));
    // Park the writer at its first flush, so it cannot win a racing delta
    // drain. queue_pending below must execute on the producer. A timeout
    // releases the worker on test failure instead of leaving Drop blocked.
    struct PausedFirstFlush {
        inner: Writer,
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
        first: bool,
    }
    impl Write for PausedFirstFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.inner.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.first {
                self.first = false;
                self.entered
                    .send(())
                    .map_err(|e| io::Error::other(e.to_string()))?;
                self.release
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|e| io::Error::new(io::ErrorKind::TimedOut, e.to_string()))?;
            }
            self.inner.flush()
        }
    }
    let (entered, wait_entered) = std::sync::mpsc::channel();
    let (release, wait_release) = std::sync::mpsc::channel();
    let writer = ChunkedWriter {
        w: Box::new(PausedFirstFlush {
            inner: Writer {
                progress: progress.clone(),
                pending: vec![],
                fail_delta: Some(json!({"content":"X"})),
            },
            entered,
            release: wait_release,
            first: true,
        }),
    };
    let sw = StreamWriter::start_with_counted_deltas(writer, move |text| {
        let (delta, resolved) = parser.lock().push_counted(text);
        (
            format!("data: {}\n\n", json!({"choices":[{"delta":delta}]})),
            resolved,
        )
    });
    sw.event(&json!({"barrier": true}));
    wait_entered.recv_timeout(Duration::from_secs(5)).unwrap();
    sw.delta("a<");
    sw.event(&json!({"control": true}));
    sw.delta("/think>X");
    let written = sw.written();
    release.send(()).unwrap();
    let (_, error) = sw.finish().unwrap();
    assert_eq!(error.unwrap().kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(written.load(Ordering::Relaxed), 1);
    let wire = progress.0 .0.lock();
    assert_eq!(wire.failures, 1);
    assert!(wire.events.iter().any(|event| event["control"] == true));
}

#[derive(Clone, Default)]
struct OrderingWire(Arc<ParkingMutex<Vec<u8>>>);
impl Write for OrderingWire {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn ordered_payloads(wire: &OrderingWire) -> String {
    String::from_utf8(wire.0.lock().clone())
        .unwrap()
        .split("\r\n")
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, text)| text)
        .collect()
}
fn await_worker_boundary(sw: &StreamWriter, entered: &std::sync::mpsc::Receiver<()>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        sw.delta(""); // carry any try-lock producer spill, without new text
        match entered.recv_timeout(Duration::from_millis(2)) {
            Ok(()) => return,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => assert!(Instant::now() < deadline),
            Err(error) => panic!("worker boundary disappeared: {error}"),
        }
    }
}

#[test]
fn extracted_worker_batch_wraps_before_later_producer_batch_and_control() {
    let wire = OrderingWire::default();
    let wrapped = Arc::new(ParkingMutex::new(Vec::new()));
    let (entered, wait_entered) = std::sync::mpsc::channel();
    let (release, wait_release) = std::sync::mpsc::channel();
    let wait_release = ParkingMutex::new(wait_release);
    let seen = Arc::clone(&wrapped);
    let sw = StreamWriter::start_with_counted_deltas(
        ChunkedWriter {
            w: Box::new(wire.clone()),
        },
        move |text| {
            if text == "A" {
                // The worker has extracted A but has not mutated wrapper
                // state yet. Producer B/control must not wrap ahead of it.
                entered.send(()).unwrap();
                wait_release
                    .lock()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
            }
            seen.lock().push(text.to_string());
            (format!("<{text}>"), text.len())
        },
    );
    sw.delta("A");
    await_worker_boundary(&sw, &wait_entered);
    let order = Arc::clone(&sw.order);
    let (offered, wait_offered) = std::sync::mpsc::channel();
    let (queued, wait_queued) = std::sync::mpsc::channel();
    let producer = std::thread::spawn(move || {
        sw.delta("B");
        offered.send(()).unwrap(); // delta must not wait for the order guard
        sw.send("C".into());
        queued.send(()).unwrap();
        let (_, error) = sw.finish().unwrap();
        assert!(error.is_none());
    });
    wait_offered.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        order.try_lock().is_none(),
        "extracted A lost its wrap/enqueue order"
    );
    assert!(matches!(
        wait_queued.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    release.send(()).unwrap();
    wait_queued.recv_timeout(Duration::from_secs(5)).unwrap();
    producer.join().unwrap();
    assert_eq!(*wrapped.lock(), vec!["A", "B"]);
    assert_eq!(ordered_payloads(&wire), "<A><B>C");
}

#[test]
fn timeout_before_role_selects_queued_role_before_later_text() {
    let (tx, rx) = std::sync::mpsc::channel();
    // Force the exact branch: timeout has happened; the producer then queues
    // the role and publishes text before the consumer resumes selection.
    assert!(matches!(
        rx.recv_timeout(Duration::ZERO),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    let order = ParkingMutex::new(());
    {
        let _guard = order.lock();
        tx.send(("role".into(), 0)).unwrap();
    }
    let buf = DeltaBuffer::new();
    buf.push(&mut String::new(), "A");
    let wrap: Wrap = Arc::new(|text| (format!("<{text}>"), text.len()));
    let mut local = String::new();
    let (first, closed) =
        StreamWriter::pending_after_timeout(&rx, Some(&buf), Some(&wrap), &order, &mut local);
    assert!(!closed);
    assert_eq!(first, Some(("role".into(), 0)));
    let (second, closed) =
        StreamWriter::pending_after_timeout(&rx, Some(&buf), Some(&wrap), &order, &mut local);
    assert!(!closed);
    assert_eq!(second, Some(("<A>".into(), 1)));
    // Disconnect still drains the last bound batch before reporting closure.
    buf.push(&mut String::new(), "B");
    drop(tx);
    let (last, closed) =
        StreamWriter::pending_after_timeout(&rx, Some(&buf), Some(&wrap), &order, &mut local);
    assert!(closed);
    assert_eq!(last, Some(("<B>".into(), 1)));
}

#[test]
fn blocked_network_flush_holds_no_order_guard_or_delta_control_path() {
    struct PausedIo {
        inner: OrderingWire,
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
        first: bool,
    }
    impl Write for PausedIo {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.inner.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.first {
                self.first = false;
                self.entered
                    .send(())
                    .map_err(|e| io::Error::other(e.to_string()))?;
                self.release
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|e| io::Error::new(io::ErrorKind::TimedOut, e.to_string()))?;
            }
            self.inner.flush()
        }
    }
    let wire = OrderingWire::default();
    let (entered, wait_entered) = std::sync::mpsc::channel();
    let (release, wait_release) = std::sync::mpsc::channel();
    let sw = StreamWriter::start_with_deltas(
        ChunkedWriter {
            w: Box::new(PausedIo {
                inner: wire.clone(),
                entered,
                release: wait_release,
                first: true,
            }),
        },
        |text| format!("<{text}>"),
    );
    sw.delta("A");
    await_worker_boundary(&sw, &wait_entered);
    assert!(
        sw.order.try_lock().is_some(),
        "network I/O held the ordering guard"
    );
    // Both return while the sole writer remains parked inside flush.
    sw.delta("B");
    sw.send("C".into());
    release.send(()).unwrap();
    let (_, error) = sw.finish().unwrap();
    assert!(error.is_none());
    assert_eq!(ordered_payloads(&wire), "<A><B>C");
}

/// A family whose `Served` seam keeps the default `complete_observed` (no
/// typed observations) splits by the markers' text instead, with a partial
/// marker held across pushes: the answer reaches `content` rather than
/// stranding in `reasoning_content` (#946, 2026-10-21).
#[test]
fn a_family_without_decode_observations_splits_by_marker_text() {
    let progress = Progress::default();
    let mut engine = Scripted::new(Case::CompleteMarker, progress.clone());
    engine.observes = false;
    let tags = engine.think_tags().unwrap();
    // The prompt opened the block: reasoning until the end marker's text.
    let mut p = reasoning::Reasoning::for_chat(&engine, &[tags.start], false)
        .unwrap()
        .unwrap();
    let a = p.push("The user asks a simple thing.</thi");
    assert_eq!(a.get("reasoning_content").and_then(|v| v.as_str()), Some("The user asks a simple thing."));
    assert!(a.get("content").is_none(), "{a:?}");
    let b = p.push("nk>Paris");
    assert_eq!(b.get("content").and_then(|v| v.as_str()), Some("Paris"), "{b:?}");
    assert!(b.get("reasoning_content").is_none(), "{b:?}");
    let c = p.finish();
    assert!(c.is_empty(), "{c:?}");
    // A stop string that cuts the marker short: the partial bytes are emitted
    // at finish, in the block they were in.
    let mut p = reasoning::Reasoning::for_chat(&engine, &[tags.start], false)
        .unwrap()
        .unwrap();
    let a = p.push("thought</th");
    assert_eq!(a.get("reasoning_content").and_then(|v| v.as_str()), Some("thought"));
    let c = p.finish();
    assert_eq!(c.get("reasoning_content").and_then(|v| v.as_str()), Some("</th"), "{c:?}");
    // With observations the splitter is exact by id and ignores literals.
    engine.observes = true;
    let mut p = reasoning::Reasoning::for_chat(&engine, &[tags.start], false)
        .unwrap()
        .unwrap();
    let a = p.push("literal </think> stays reasoning");
    assert_eq!(a.get("reasoning_content").and_then(|v| v.as_str()), Some("literal </think> stays reasoning"));
}
