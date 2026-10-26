//! Host-only boundary controls. Actual HTTP ID observation, Scripted CPU decode;
//! injected delivery rejection is not OS disconnect detection/native recovery.
use super::*;
use anyhow::Context;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;

const WAIT: Duration = Duration::from_secs(10);
const CHAT: &str = "/v1/chat/completions";

#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    messages: Vec<Value>,
    // Mutable rendered-prefix reuse witness, NOT the immutable accepted reply.
    ids: Vec<u32>,
    // Only this named turn: an unbound whole-history reference owns no earlier turns.
    accepted_reply: Value,
}
fn reply_fields(reply: &front::Reply) -> Value {
    json!({
        "ids":reply.ids, "text":reply.text, "block":format!("{:?}",reply.block),
        "stop":format!("{:?}",reply.stop), "content":reply.content,"tail":reply.tail,
        "reasoning_text":reply.reasoning_text,"content_text":reply.content_text,
        "partial":reply.partial,"generated":reply.generated,
        "logprobs_bits":reply.logprobs.as_ref().map(|v|v.iter().map(|x|x.to_bits()).collect::<Vec<_>>()),
        "rows_bits":reply.rows.as_ref().map(|v|v.iter().map(|r|r.iter().map(|x|x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>()),
    })
}
fn accepted_owners(shared: &Shared, id: &str) -> Vec<Arc<front::Reply>> {
    let resolved = shared
        .sessions
        .lock()
        .resolve(id, None, false, Instant::now())
        .unwrap();
    resolved.bound.into_iter().map(|b| b.reply).collect()
}
fn assert_same_accepted_owners(shared: &Shared, id: &str, before: &[Arc<front::Reply>]) {
    let after = accepted_owners(shared, id);
    assert_eq!(after.len(), before.len());
    for (a, b) in after.iter().zip(before) {
        assert!(
            Arc::ptr_eq(a, b),
            "reservation retains the original accepted Reply owner"
        );
        assert_eq!(
            reply_fields(a),
            reply_fields(b),
            "accepted Reply fields unchanged"
        );
    }
}
fn lineage_fields(mut fields: Value, thinking: bool) -> Value {
    // Tiny LFM's generation prompt opens <think>. Off appends </think>,
    // which its historical plain assistant rendering does not reproduce.
    // On + explicit existing preserve policy is the aligned fixture; default
    // stripping/last-user inference is not an unspoken precondition.
    fields["enable_thinking"] = json!(thinking);
    fields["preserve_thinking"] = json!(true);
    fields
}
fn snapshot(shared: &Shared, id: &str) -> Snapshot {
    // The mutex temporary ends at this semicolon; conversion is outside it.
    let resolved = shared
        .sessions
        .lock()
        .resolve(id, None, true, Instant::now())
        .unwrap();
    Snapshot {
        messages: message_values(resolved.messages.unwrap()),
        ids: resolved.ids,
        accepted_reply: reply_fields(&resolved.bound.last().unwrap().reply),
    }
}
fn json_reply(shared: &Shared, fields: Value) -> Value {
    let _drain = shared.drains.try_take().unwrap();
    let _turn = shared.turns.take();
    let Reply::Json(reply) = response_reply(
        chat_completions(shared, &native_response_request(fields), None),
        "boundary JSON",
    ) else {
        panic!("JSON requested")
    };
    reply
}
fn connect_post(shared: &Shared, body: &Value) -> TcpStream {
    let tiny_http::ListenAddr::IP(address) = shared.server.server_addr() else {
        panic!("IP")
    };
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(WAIT)).unwrap();
    client.set_write_timeout(Some(WAIT)).unwrap();
    let body = body.to_string();
    write!(client, "POST {CHAT} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    client
}
fn receive(shared: &Shared) -> Request {
    shared
        .server
        .recv_timeout(WAIT)
        .unwrap()
        .expect("actual HTTP request")
}
fn closed_client_request(shared: &Shared, fields: Value) -> Request {
    let client = connect_post(shared, &native_response_request(fields));
    let request = receive(shared);
    client.shutdown(Shutdown::Both).unwrap();
    request
}
#[derive(Clone, Default)]
struct Captured(Arc<ParkingMutex<Vec<u8>>>);
impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn captured_events(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .unwrap()
        .split("data: ")
        .skip(1)
        .filter_map(|s| {
            let payload = s.split_once("\n\n").unwrap().0;
            (payload != "[DONE]").then(|| serde_json::from_str(payload).unwrap())
        })
        .collect()
}
fn assert_ancestor_and_successor(shared: &Shared, a_id: &str, before: &Snapshot) {
    assert_eq!(
        &snapshot(shared, a_id),
        before,
        "latest non-divergent A preserved"
    );
    assert_successor(shared, a_id, before, true);
}
fn assert_successor(shared: &Shared, a_id: &str, before: &Snapshot, thinking: bool) {
    let b = json_reply(
        shared,
        lineage_fields(
            json!({"previous_response_id":a_id,"input":"recovery","max_tokens":8}),
            thinking,
        ),
    );
    assert!(b["timings"]["session"]["bound"].as_u64().unwrap() > 0);
    let saved = snapshot(shared, b["id"].as_str().unwrap());
    assert_eq!(saved.messages.len(), 4);
    assert_eq!(&saved.messages[..2], &before.messages);
    let reference = scripted_shared(Case::FirstFragmentRefusal);
    let a = json_reply(
        &reference,
        lineage_fields(
            json!({"messages":[{"role":"user","content":"A"}],"store":true}),
            thinking,
        ),
    );
    let rb = json_reply(
        &reference,
        lineage_fields(
            json!({"messages":[{"role":"user","content":"A"},a["choices"][0]["message"],{"role":"user","content":"recovery"}]}),
            thinking,
        ),
    );
    assert_eq!(b["prompt_token_ids"], rb["prompt_token_ids"]);
    assert_eq!(b["choices"][0]["token_ids"], rb["choices"][0]["token_ids"]);
    assert_eq!(b["choices"][0]["message"], rb["choices"][0]["message"]);
    assert_eq!(saved, snapshot(&reference, rb["id"].as_str().unwrap()));
}

// Both frontiers are checked in the divergent case: accepted history is
// immutable, while reserve is expected to forget nonmatching rendered IDs.
fn assert_divergent_ancestor(shared: &Shared, id: &str, before: &Snapshot) {
    let after = snapshot(shared, id);
    let tags = shared.engine.think_tags().unwrap();
    let mut omitted = vec![tags.start, tags.end];
    omitted.extend([97; 8]);
    let cut = before.ids.len().checked_sub(omitted.len()).unwrap();
    assert_eq!(
        &before.ids[cut..],
        omitted,
        "off-mode generation-only empty block + accepted reply"
    );
    assert_eq!(
        after.ids,
        &before.ids[..cut],
        "divergent reservation retains exact shared rendered prefix"
    );
    assert_eq!(
        after.messages, before.messages,
        "divergence preserves accepted messages"
    );
    assert_eq!(
        after.accepted_reply, before.accepted_reply,
        "divergence preserves complete accepted Reply, not only reuse slice"
    );
}
#[test]
fn first_fragment_refusal_preserves_latest_stored_ancestor_and_valid_successor() {
    assert_first_fragment_refusal(true);
}
#[test]
fn divergent_generation_prefix_preserves_accepted_reply_on_refusal_and_writer_start_error() {
    assert_first_fragment_refusal(false);
    assert_pre_decode_writer_start_failure(false);
}
fn assert_first_fragment_refusal(thinking: bool) {
    let progress = Progress::default();
    let mut shared = scripted_shared_with_progress(Case::FirstFragmentRefusal, progress.clone());
    Arc::get_mut(&mut shared).unwrap().drains =
        ResponseDrains::new(crate::options::ServerResponseConfig {
            raw_bytes: 0,
            ..Default::default()
        });
    // JSON setup/recovery do not use SSE raw-lag admission. R0 is this fixture's
    // explicit existing option value, not a new default or public switch.
    let a = json_reply(
        &shared,
        lineage_fields(
            json!({"messages":[{"role":"user","content":"A"}],"store":true}),
            thinking,
        ),
    );
    let a_id = a["id"].as_str().unwrap().to_owned();
    let before = snapshot(&shared, &a_id);
    let before_owners = accepted_owners(&shared, &a_id);
    let offered_before = progress.0 .0.lock().first_case_offered;
    let mut cancelled_id = String::new();
    let captured = Captured::default();
    let request = closed_client_request(
        &shared,
        lineage_fields(
            json!({"previous_response_id":a_id,"input":"cancelled-attempt","stream":true}),
            thinking,
        ),
    );
    let mut settled = false;
    handle_model_request_with_stream(
        &shared,
        CHAT,
        request,
        FirstToken::default(),
        |request| {
            // No automatic Request-drop 500: this transport is intentionally not
            // used for T1's event capture (T2 below observes actual received bytes).
            drop(request.into_writer());
            Ok(ChunkedWriter {
                w: Box::new(captured.clone()),
            })
        },
        |turn| {
            drop(turn.take());
            settled = true;
        },
    )
    .unwrap();
    assert!(settled);
    let events = captured_events(&captured.0.lock());
    let last = events.iter().find(|e| e.get("timings").is_some()).unwrap();
    cancelled_id.push_str(last["id"].as_str().unwrap());
    assert_ne!(cancelled_id, a_id);
    assert!(last["timings"]["session"]["bound"].as_u64().unwrap() > 0);
    assert_eq!(
        last["choices"][0]["finish_reason"], "cancelled",
        "first raw handover refusal cancels before acceptance"
    );
    assert_eq!(last["usage"]["completion_tokens"], 0);
    let f = &last["timings"]["frontier"];
    assert!(f["generated"].as_u64().unwrap() > 0);
    for name in [
        "accepted",
        "partial_bytes",
        "delivered_bytes",
        "written_bytes",
    ] {
        assert_eq!(f[name], 0, "zero frontier: {name}");
    }
    assert_eq!(f["logged"], false);
    assert_eq!(f["write_failed"], false);
    assert_eq!(last["timings"]["response_backlog_limit"], true);
    assert!(last["timings"]["session"]["id"].is_null());
    assert!(shared
        .sessions
        .lock()
        .resolve(&cancelled_id, None, true, Instant::now())
        .is_err());
    assert_eq!(shared.sessions.lock().json()["turns"], 1);
    assert_eq!(shared.drains.json()["overload"]["unresolved_bytes"], 1);
    assert_eq!(shared.drains.active.load(Ordering::Acquire), 0);
    assert!(shared.turns.try_take().is_some());
    {
        let wire = progress.0 .0.lock();
        assert_eq!(
            wire.first_case_offered,
            offered_before + 1,
            "exact first handover"
        );
        assert_eq!(wire.first_case_refused, 1, "one refusal, no later offer");
        assert_eq!(wire.model_entries, 2);
        if thinking {
            assert!(
                wire.first_case_prompt.starts_with(&before.ids),
                "D was non-divergent before refusal"
            );
        } else {
            let lcp = wire
                .first_case_prompt
                .iter()
                .zip(&before.ids)
                .take_while(|(a, b)| a == b)
                .count();
            assert_eq!(
                lcp,
                before.ids.len() - 10,
                "actual D prompt diverges at empty-block generation suffix"
            );
        }
    }
    assert_same_accepted_owners(&shared, &a_id, &before_owners);
    if thinking {
        assert_ancestor_and_successor(&shared, &a_id, &before);
    } else {
        assert_divergent_ancestor(&shared, &a_id, &before);
        assert_successor(&shared, &a_id, &before, false);
    }
    assert_eq!(progress.0 .0.lock().model_entries, 3);
}

#[test]
fn pre_decode_writer_start_failure_preserves_latest_ancestor_and_releases_ownership() {
    assert_pre_decode_writer_start_failure(true);
}
fn assert_pre_decode_writer_start_failure(thinking: bool) {
    let progress = Progress::default();
    let shared = scripted_shared_with_progress(Case::ApiLineage, progress.clone());
    let a = json_reply(
        &shared,
        lineage_fields(
            json!({"messages":[{"role":"user","content":"A"}],"store":true}),
            thinking,
        ),
    );
    let a_id = a["id"].as_str().unwrap().to_owned();
    let before = snapshot(&shared, &a_id);
    let before_owners = accepted_owners(&shared, &a_id);
    let request = closed_client_request(
        &shared,
        lineage_fields(
            json!({"previous_response_id":a_id,"input":"cancelled-attempt","stream":true}),
            thinking,
        ),
    );
    let mut factory_called = false;
    let mut settled_called = false;
    let result = handle_model_request_with_stream(
        &shared,
        CHAT,
        request,
        FirstToken::default(),
        |request| {
            factory_called = true;
            drop(request.into_writer());
            anyhow::bail!("injected pre-decode writer start failure")
        },
        |turn| {
            settled_called = true;
            drop(turn.take());
        },
    );
    let error = result
        .expect_err("writer-start refusal must remain an error before decode")
        .to_string();
    assert!(
        !settled_called,
        "no settlement callback before writer creation"
    );
    assert!(error.contains("injected pre-decode writer start failure"));
    assert!(factory_called);
    assert_eq!(
        progress.0 .0.lock().model_entries,
        1,
        "D never entered model"
    );
    assert_eq!(
        shared.sessions.lock().json()["turns"],
        1,
        "no D publication"
    );
    assert_eq!(shared.drains.active.load(Ordering::Acquire), 0);
    assert!(
        shared.turns.try_take().is_some(),
        "outer turn dropped on early Err"
    );
    assert_same_accepted_owners(&shared, &a_id, &before_owners);
    if thinking {
        assert_ancestor_and_successor(&shared, &a_id, &before);
    } else {
        assert_divergent_ancestor(&shared, &a_id, &before);
        assert_successor(&shared, &a_id, &before, false);
    }
    assert_eq!(progress.0 .0.lock().model_entries, 2);
}

#[derive(Clone, Copy, Debug)]
enum Release {
    Publication,
    FullSettlement,
}

// A real client receives headers and role ID. The next text event is withheld;
// named injected BrokenPipe controls later failure timing, not OS detection.
struct RoleThenGate {
    inner: Box<dyn Write + Send + 'static>,
    pending: Vec<u8>,
    attempted: Option<Captured>,
    role_passed: bool,
    entered: Option<mpsc::Sender<u64>>,
    expected_first_reasoning: Option<&'static str>,
    reject_after_gate: bool,
    max_pause: Duration,
    released: mpsc::Receiver<()>,
}
impl Write for RoleThenGate {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(bytes);
        if let Some(attempted) = &self.attempted {
            attempted.0.lock().extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if !self.role_passed {
            let events = captured_events(&self.pending);
            assert_eq!(events.len(), 1);
            assert_eq!(
                events[0]["choices"][0]["delta"],
                json!({"role":"assistant"})
            );
            self.inner.write_all(&self.pending)?;
            self.inner.flush()?;
            self.pending.clear();
            self.role_passed = true;
            return Ok(());
        }
        let events = captured_events(&self.pending);
        // Benchmark forward mode also carries the final HTTP zero chunk.
        // Incumbent strict T2 mode keeps its exact next-event assertion.
        if events.is_empty() && self.expected_first_reasoning.is_none() {
            self.inner.write_all(&self.pending)?;
            self.inner.flush()?;
            self.pending.clear();
            return Ok(());
        }
        assert_eq!(events.len(), 1);
        let delta = &events[0]["choices"][0]["delta"];
        let visible = ["content", "reasoning_content"]
            .iter()
            .any(|k| delta[*k].as_str().is_some_and(|s| !s.is_empty()));
        if self.entered.is_some() && (self.expected_first_reasoning.is_some() || visible) {
            if let Some(expected) = self.expected_first_reasoning {
                assert_eq!(delta["reasoning_content"], expected);
            }
            let _ = self.entered.take().unwrap().send(uptime_ns());
            self.released
                .recv_timeout(self.max_pause)
                .map_err(|e| io::Error::new(io::ErrorKind::TimedOut, e.to_string()))?;
            if self.reject_after_gate {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "injected delivery rejection after client disconnect",
                ));
            }
        }
        self.inner.write_all(&self.pending)?;
        self.inner.flush()?;
        self.pending.clear();
        Ok(())
    }
}

#[derive(Default)]
struct HttpCleanup {
    model_release: Option<mpsc::Sender<()>>,
    writer_release: Option<mpsc::Sender<()>>,
    sockets: Vec<TcpStream>,
    threads: Vec<JoinHandle<()>>,
}
impl HttpCleanup {
    fn release_model(&mut self) {
        self.model_release.take().unwrap().send(()).unwrap();
    }
    fn release_writer(&mut self) {
        self.writer_release.take().unwrap().send(()).unwrap();
    }
    fn join(&mut self) {
        self.join_observed(|_| {});
    }
    fn join_observed(&mut self, mut joined: impl FnMut(bool)) {
        // Never let an early child's panic detach the remaining owned handles.
        // All gates must have been released before the explicit success join.
        let mut panicked = false;
        while let Some(thread) = self.threads.pop() {
            let ok = thread.join().is_ok();
            panicked |= !ok;
            joined(ok);
        }
        assert!(
            !panicked,
            "HTTP cleanup joined every worker before reporting panic"
        );
    }
}
impl Drop for HttpCleanup {
    fn drop(&mut self) {
        // Unblock BOTH A gates before joining any B waiter, on every failure.
        for socket in &self.sockets {
            let _ = socket.shutdown(Shutdown::Both);
        }
        if let Some(release) = self.model_release.take() {
            let _ = release.send(());
        }
        if let Some(release) = self.writer_release.take() {
            let _ = release.send(());
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[test]
fn cleanup_joins_remaining_workers_before_reporting_child_panic() {
    let mut cleanup = HttpCleanup::default();
    cleanup.threads.push(std::thread::spawn(|| {}));
    cleanup
        .threads
        .push(std::thread::spawn(|| panic!("named cleanup child failure")));
    let mut joined = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cleanup.join_observed(|ok| joined.push(ok));
    }));
    assert!(result.is_err(), "child panic remains a test failure signal");
    assert_eq!(
        joined,
        vec![false, true],
        "both actual joins precede aggregate panic"
    );
    assert!(
        cleanup.threads.is_empty(),
        "no worker handle detached on panic"
    );
}

fn read_role(client: TcpStream) -> (BufReader<TcpStream>, Value) {
    let mut reader = BufReader::new(client);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert!(line.starts_with("HTTP/1.1 200"));
    let mut header_bytes = line.len();
    loop {
        line.clear();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        header_bytes += line.len();
        assert!(header_bytes < 8192);
        if line == "\r\n" {
            break;
        }
    }
    line.clear();
    assert!(reader.read_line(&mut line).unwrap() > 0);
    let n = usize::from_str_radix(line.trim(), 16).unwrap();
    assert!(n > 0 && n < 8192, "bounded first role chunk");
    let mut bytes = vec![0; n + 2];
    reader.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes[n..], b"\r\n");
    let payload = std::str::from_utf8(&bytes[..n]).unwrap();
    let event = serde_json::from_str(
        payload
            .strip_prefix("data: ")
            .unwrap()
            .strip_suffix("\n\n")
            .unwrap(),
    )
    .unwrap();
    (reader, event)
}
fn read_json(client: &mut TcpStream) -> Result<Value> {
    let mut bytes = Vec::new();
    client.take(1 << 16).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() < 1 << 16, "bounded B response fixture");
    let response = std::str::from_utf8(&bytes)?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("HTTP header terminator")?;
    anyhow::ensure!(
        headers.starts_with("HTTP/1.1 200"),
        "client-observed ID must continue: {response}"
    );
    Ok(serde_json::from_str(body)?)
}

#[test]
fn client_observed_id_waits_for_acceptance_and_fully_continues_after_disconnect() {
    for release_at in [Release::Publication, Release::FullSettlement] {
        assert_client_observed_continuation(release_at);
    }
}
fn assert_client_observed_continuation(release_at: Release) {
    let reference = scripted_shared(Case::PartialTail);
    let reference_a = json_reply(
        &reference,
        json!({"messages":[{"role":"user","content":"A"}],"store":true,"max_tokens":3,"enable_thinking":true}),
    );
    assert_eq!(
        reference_a["choices"][0]["message"]["reasoning_content"], "a</thi",
        "nonvacuous partial-tail oracle"
    );
    let a_expected = snapshot(&reference, reference_a["id"].as_str().unwrap());
    assert_eq!(
        a_expected.ids.last(),
        Some(&97),
        "only the complete accepted token is saved"
    );
    let reference_b = json_reply(
        &reference,
        json!({"messages":[{"role":"user","content":"A"},reference_a["choices"][0]["message"],{"role":"user","content":"B"}],"max_tokens":3,"enable_thinking":true}),
    );
    let b_expected = snapshot(&reference, reference_b["id"].as_str().unwrap());
    let progress = Progress::default();
    let shared = scripted_shared_with_progress(Case::PartialTail, progress.clone());
    assert_eq!(shared.engine.batch(), 1, "this is explicitly B=1 only");
    let mut cleanup = HttpCleanup::default();
    let (model_entered, model_entry) = mpsc::channel();
    let (model_release, model_released) = mpsc::channel();
    progress.0 .0.lock().before_decode = Some((model_entered, model_released));
    cleanup.model_release = Some(model_release);
    let a_client = connect_post(
        &shared,
        &native_response_request(
            json!({"messages":[{"role":"user","content":"A"}],"store":true,"stream":true,"max_tokens":3,"enable_thinking":true}),
        ),
    );
    cleanup.sockets.push(a_client.try_clone().unwrap());
    let a_request = receive(&shared);
    let (writer_entered, writer_entry) = mpsc::channel();
    let (writer_release, writer_released) = mpsc::channel();
    cleanup.writer_release = Some(writer_release);
    let (published, publication) = mpsc::channel();
    let (a_done, a_result) = mpsc::channel();
    let attempted = Captured::default();
    let attempted_by_writer = attempted.clone();
    let a_owner = shared.clone();
    cleanup.threads.push(std::thread::spawn(move || {
        let result = handle_model_request_with_stream(
            &a_owner,
            CHAT,
            a_request,
            FirstToken::default(),
            |request| {
                let writer = ChunkedWriter::start(request, "text/event-stream")?;
                Ok(ChunkedWriter {
                    w: Box::new(RoleThenGate {
                        inner: writer.w,
                        pending: Vec::new(),
                        attempted: Some(attempted_by_writer),
                        role_passed: false,
                        entered: Some(writer_entered),
                        expected_first_reasoning: Some("a"),
                        reject_after_gate: true,
                        max_pause: WAIT.saturating_mul(3),
                        released: writer_released,
                    }),
                })
            },
            |turn| {
                if matches!(release_at, Release::Publication) {
                    drop(turn.take());
                }
                let _ = published.send(()); // BOOL only: never supplies the client ID.
            },
        );
        // FullSettlement's outer Option<Turn> dropped on handler return.
        let _ = a_done.send(result.map_err(|e| e.to_string()));
    }));
    let (a_reader, role) = read_role(a_client);
    assert_eq!(role["choices"][0]["delta"], json!({"role":"assistant"}));
    let observed_a_id = role["id"].as_str().unwrap().to_owned();
    assert!(!observed_a_id.is_empty());
    model_entry.recv_timeout(WAIT).unwrap();
    a_reader.get_ref().shutdown(Shutdown::Both).unwrap();
    drop(a_reader);
    // B is built only from the ID actually received on connection A.
    let mut b_client = connect_post(
        &shared,
        &native_response_request(
            json!({"previous_response_id":observed_a_id,"input":"B","max_tokens":3,"enable_thinking":true}),
        ),
    );
    cleanup.sockets.push(b_client.try_clone().unwrap());
    let b_request = receive(&shared);
    let (b_response, b_received) = mpsc::channel();
    cleanup.threads.push(std::thread::spawn(move || {
        let result = read_json(&mut b_client).map_err(|e| e.to_string());
        let _ = b_response.send(result);
    }));
    let (_queue_observation, queued) = shared.turns.observe_next_take();
    let (b_done, b_result) = mpsc::channel();
    let b_owner = shared.clone();
    cleanup.threads.push(std::thread::spawn(move || {
        let result = handle_model_request(&b_owner, CHAT, b_request, FirstToken::default());
        let _ = b_done.send(result.map_err(|e| e.to_string()));
    }));
    let (ticket, serving) = queued.recv_timeout(WAIT).unwrap();
    assert!(ticket > serving, "actual B arrives before A publication");
    assert_eq!(
        progress.0 .0.lock().model_entries,
        1,
        "only paused A entered model"
    );
    {
        let missing = shared
            .sessions
            .lock()
            .resolve(&observed_a_id, None, true, Instant::now());
        assert!(
            missing.is_err(),
            "client saw an ID, not yet a committed reply"
        );
    } // Drop lookup Result, guard and any owners before releasing model.
    cleanup.release_model();
    publication
        .recv_timeout(WAIT)
        .expect("accepted publication barrier precedes held transport");
    writer_entry
        .recv_timeout(WAIT)
        .expect("next text event is held after role delivery");
    if matches!(release_at, Release::FullSettlement) {
        // B's real ticket was observed above. Read A's actual held ticket after
        // publication and while its writer is gated, not a timed non-progress
        // guess: an early-release mutant fails even if B has not been scheduled.
        let held_serving = shared.turns.serving_for_test();
        assert_eq!(
            held_serving, serving,
            "FullSettlement retains A ticket through held writer"
        );
        assert!(
            ticket > held_serving,
            "actual B remains behind held A ticket"
        );
        assert_eq!(
            progress.0 .0.lock().model_entries,
            1,
            "FullSettlement does not admit B before writer release"
        );
        cleanup.release_writer();
    }
    let b = b_received
        .recv_timeout(WAIT)
        .expect("B full HTTP response completes at this arm's release boundary")
        .expect("the ID actually received by client A continues successfully over HTTP");
    b_result.recv_timeout(WAIT).unwrap().unwrap();
    assert_eq!(progress.0 .0.lock().model_entries, 2);
    assert!(b["timings"]["session"]["bound"].as_u64().unwrap() > 0);
    let b_id = b["id"].as_str().unwrap().to_owned();
    assert_ne!(b_id, observed_a_id);
    assert_eq!(
        b["prompt_token_ids"], reference_b["prompt_token_ids"],
        "B prompt preserves accepted A including terminal partial tail"
    );
    assert_eq!(
        b["choices"][0]["token_ids"],
        reference_b["choices"][0]["token_ids"]
    );
    assert_eq!(
        b["choices"][0]["message"], reference_b["choices"][0]["message"],
        "B message matches identical accepted-history reference"
    );
    // B has already reserved/committed. Its rendering can change A's reusable
    // prefix (partial marker gets the template's own closing), but NOT A's
    // accepted Reply or saved fields. Compare the same post-continuation
    // projection, not a pre-B reuse witness to a post-B one.
    let b_prompt: Vec<u32> = serde_json::from_value(b["prompt_token_ids"].clone()).unwrap();
    let lcp = a_expected
        .ids
        .iter()
        .zip(&b_prompt)
        .take_while(|(a, b)| a == b)
        .count();
    let mut a_post_b_expected = a_expected.clone();
    a_post_b_expected.ids.truncate(lcp);
    let a_before_delivery_error = snapshot(&shared, &observed_a_id);
    assert_eq!(
        a_before_delivery_error, a_post_b_expected,
        "accepted A preserves terminal partial tail independently of delivery"
    );
    assert_eq!(
        a_before_delivery_error.accepted_reply, a_expected.accepted_reply,
        "B reservation never rewrites immutable accepted A frontier"
    );
    let a_owners_before_delivery_error = accepted_owners(&shared, &observed_a_id);
    let b_before_a_error = snapshot(&shared, &b_id);
    assert_eq!(
        b_before_a_error, b_expected,
        "B publication matches identical accepted-history reference"
    );
    if matches!(release_at, Release::Publication) {
        assert!(
            cleanup.writer_release.is_some(),
            "B fully published before A writer release"
        );
        cleanup.release_writer();
    }
    let error = a_result.recv_timeout(WAIT).unwrap().unwrap_err();
    assert!(
        error.contains("injected delivery rejection after client disconnect"),
        "{error}"
    );
    let frontier: Value = serde_json::from_str(error.split_once("; frontier ").unwrap().1).unwrap();
    assert_eq!(frontier["accepted"], 1);
    assert_eq!(frontier["partial_bytes"], 5);
    assert_eq!(frontier["delivered_bytes"], 6);
    assert_eq!(frontier["write_failed"], true);
    assert_eq!(frontier["written_bytes"], 0); // role has no completion raw credit
    assert_eq!(
        snapshot(&shared, &b_id),
        b_before_a_error,
        "late A cannot rewrite B"
    );
    assert_eq!(
        snapshot(&shared, &observed_a_id),
        a_before_delivery_error,
        "late A delivery failure cannot rewrite either accepted history or settled reuse witness"
    );
    assert_same_accepted_owners(&shared, &observed_a_id, &a_owners_before_delivery_error);
    let attempted = attempted.0.lock();
    assert!(!std::str::from_utf8(&attempted).unwrap().contains("[DONE]"));
    let events = captured_events(&attempted);
    assert!(events.iter().all(|e| e["id"] == observed_a_id));
    assert!(events.iter().all(|e| e.get("timings").is_none()));
    drop(attempted);
    cleanup.join();
    assert_eq!(shared.drains.active.load(Ordering::Acquire), 0);
    assert!(shared.turns.try_take().is_some());
}

// Private instrumented relative scheduling experiment, not a shipping CLI knob.
#[cfg(feature = "metal")]
#[path = "serve_overlap_benchmark.rs"]
mod overlap_benchmark;

#[test]
fn forwarding_writer_gate_preserves_empty_deltas_trailer_and_http_termination() {
    let actual = Captured::default();
    let expected = Captured::default();
    let (entered, observed) = mpsc::channel();
    let (release, released) = mpsc::channel();
    release.send(()).unwrap();
    let mut writer = ChunkedWriter {
        w: Box::new(RoleThenGate {
            inner: Box::new(actual.clone()),
            pending: Vec::new(),
            attempted: None,
            role_passed: false,
            entered: Some(entered),
            expected_first_reasoning: None,
            reject_after_gate: false,
            max_pause: WAIT,
            released,
        }),
    };
    let mut reference = ChunkedWriter {
        w: Box::new(expected.clone()),
    };
    for delta in [
        json!({"role":"assistant"}),
        json!({}),
        json!({"reasoning_content":"r"}),
        json!({}),
    ] {
        let event = json!({"id":"fixture","choices":[{"delta":delta}]});
        writer.event(&event).unwrap();
        reference.event(&event).unwrap();
    }
    writer.raw("data: [DONE]\n\n").unwrap();
    reference.raw("data: [DONE]\n\n").unwrap();
    writer.finish().unwrap();
    reference.finish().unwrap();
    assert!(
        observed.try_recv().is_ok(),
        "one actual visible gate observed"
    );
    assert!(matches!(
        observed.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
    assert_eq!(
        *actual.0.lock(),
        *expected.0.lock(),
        "same wire through zero chunk"
    );
}
