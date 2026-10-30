//! Response contexts at the HTTP boundary (decisions 83, 84, 85): a reply
//! with any number of reasoning markers completes and reports its
//! transitions; requests past the batch width that fit in memory wait for
//! the model rather than being refused. Scripted ids and a byte tokenizer;
//! no model, device or native work.
use super::*;
use std::io::Read;
use std::net::TcpStream;

fn joined(events: &[Value], field: &str) -> String {
    events
        .iter()
        .filter_map(|e| e.pointer(&format!("/choices/0/delta/{field}")))
        .filter_map(Value::as_str)
        .collect()
}

/// More marker tokens than the retired descriptor bound: the reply streams
/// and settles, its content is every 'a', and the trailer reports the
/// transitions.
#[test]
fn a_runaway_reply_streams_settles_and_reports_its_transitions() {
    let progress = Progress::default();
    let shared = scripted_shared_with_progress(Case::RunawayMarkers, progress.clone());
    let tags = shared.engine.think_tags().unwrap();
    let origin = shared.front.origin(
        &json!([]),
        &Default::default(),
        Some(Thinking {
            on: true,
            by: super::super::super::thinking::By::Template,
        }),
    );
    let run = StreamingRun {
        kind: StreamKind::Chat,
        prompt_ids: vec![tags.start],
        facts: Default::default(),
        images: vec![],
        params: DecodeParams::default(),
        id: "runaway".into(),
        model: "scripted-tail".into(),
        record: Some(origin),
        return_ids: false,
        reasoning_in_content: false,
        session: None,
        first: FirstToken::default(),
    };
    let result = stream_response_with_writer(&shared, run, || {
        Ok(ChunkedWriter {
            w: Box::new(Writer {
                progress: progress.clone(),
                pending: vec![],
                fail_delta: None,
            }),
        })
    });
    result.expect("a reply with any number of markers settles");
    let wire = progress.0 .0.lock();
    assert_eq!(joined(&wire.events, "content"), "a".repeat(RUNAWAY_GROUPS));
    assert_eq!(joined(&wire.events, "reasoning_content"), "");
    let trailer = wire
        .events
        .iter()
        .find(|e| e.get("timings").is_some())
        .expect("success trailer");
    assert_eq!(
        trailer["timings"]["reasoning_transitions"],
        2 * RUNAWAY_GROUPS as u64,
        "{}",
        trailer["timings"]
    );
    let health = shared.responses.json();
    assert_eq!(
        health["reasoning_transitions"],
        json!({"total": 2 * RUNAWAY_GROUPS, "max_per_response": 2 * RUNAWAY_GROUPS}),
        "{health}"
    );
}

/// B=1 with the model held by the first request: nine more requests are
/// admitted and wait for the model; none is refused while memory admits
/// them, and every one completes once the model is released.
#[test]
fn requests_past_the_batch_width_wait_for_the_model_instead_of_a_503() {
    const REQUESTS: usize = 10;
    let progress = Progress::default();
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    progress.0 .0.lock().before_decode = Some((entered_tx, released));
    let shared = scripted_shared_with_progress(Case::ApiLineage, progress.clone());
    let tiny_http::ListenAddr::IP(address) = shared.server.server_addr() else {
        panic!("IP test listener")
    };
    let body = json!({
        "messages":[{"role":"user","content":"p"}],"stream":true,"max_tokens":8
    })
    .to_string();
    std::thread::scope(|scope| {
        let mut clients = Vec::new();
        let mut handlers = Vec::new();
        for i in 0..REQUESTS {
            let mut client = TcpStream::connect(address).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(20)))
                .unwrap();
            write!(
                client,
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            let request = shared
                .server
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap();
            let shared = &shared;
            handlers.push(scope.spawn(move || {
                handle_model_request(
                    shared,
                    "/v1/chat/completions",
                    request,
                    FirstToken::default(),
                )
            }));
            if i == 0 {
                entered
                    .recv_timeout(Duration::from_secs(10))
                    .expect("the first request reached the model");
            }
            clients.push(client);
        }
        // Every refusal is written at once; a waiting request writes nothing
        // until the model is released.
        let mut refused = Vec::new();
        for (i, client) in clients.iter_mut().enumerate().skip(1) {
            client
                .set_read_timeout(Some(Duration::from_millis(300)))
                .unwrap();
            let mut head = [0u8; 12];
            if client.read_exact(&mut head).is_ok() {
                refused.push((i, String::from_utf8_lossy(&head).into_owned()));
            }
        }
        release.send(()).unwrap();
        let mut done = 0;
        for client in clients.iter_mut() {
            client
                .set_read_timeout(Some(Duration::from_secs(20)))
                .unwrap();
            let mut wire = Vec::new();
            let _ = client.read_to_end(&mut wire);
            if String::from_utf8_lossy(&wire).contains("data: [DONE]") {
                done += 1;
            }
        }
        for handler in handlers {
            let _ = handler.join().unwrap();
        }
        assert_eq!(refused, vec![], "requests within memory were refused");
        assert_eq!(done, REQUESTS, "every queued request completes");
    });
    assert_eq!(progress.0 .0.lock().model_entries, REQUESTS);
}

/// A machine registry of the test's own, so no other process's reservation
/// enters the arithmetic, and an injected availability.
struct Registry(std::path::PathBuf);

impl Registry {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "lloom-response-contexts-{name}-{}-{}",
            std::process::id(),
            uptime_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn gate(&self, available: u64) -> lloom_tensor::machine::Gate {
        lloom_tensor::machine::Gate::new(lloom_tensor::machine::GateConfig {
            reserve: Some(0),
            registry: Some(self.0.clone()),
            available: Some(available),
            ..Default::default()
        })
    }

    fn contexts(&self, raw_bytes: usize, available: u64) -> ResponseContexts {
        ResponseContexts::with_memory(
            crate::options::ServerResponseConfig {
                contexts: None,
                raw_bytes,
            },
            ResponseMemory::Machine {
                gate: self.gate(available),
                holding: parking_lot::Mutex::new(None),
            },
        )
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn reserved(contexts: &ResponseContexts) -> u64 {
    match &contexts.memory {
        ResponseMemory::Machine { holding, .. } => holding.lock().as_ref().unwrap().reserved(),
        _ => unreachable!("a machine-gated fixture"),
    }
}

/// Each context is `raw_bytes` plus its body; the server's reservation is
/// widened to exactly the bytes in use, never shrunk, and a context inside
/// it is admitted without widening.
#[test]
fn the_machine_reservation_accounts_every_context_byte() {
    const R: usize = 1000;
    let registry = Registry::new("accounting");
    let contexts = registry.contexts(R, 1 << 30);
    let a = contexts.try_take(10).unwrap();
    assert_eq!(contexts.bytes.load(Ordering::Acquire), R + 10);
    assert_eq!(reserved(&contexts), (R + 10) as u64);
    let b = contexts.try_take(20).unwrap();
    let c = contexts.try_take(30).unwrap();
    assert_eq!(contexts.active.load(Ordering::Acquire), 3);
    assert_eq!(contexts.bytes.load(Ordering::Acquire), 3 * R + 60);
    assert_eq!(reserved(&contexts), (3 * R + 60) as u64);
    drop(b);
    assert_eq!(contexts.bytes.load(Ordering::Acquire), 2 * R + 40);
    assert_eq!(
        reserved(&contexts),
        (3 * R + 60) as u64,
        "a reservation is a high-water mark"
    );
    let d = contexts.try_take(5).unwrap();
    assert_eq!(contexts.bytes.load(Ordering::Acquire), 3 * R + 45);
    assert_eq!(
        reserved(&contexts),
        (3 * R + 60) as u64,
        "admitted inside the reservation"
    );
    drop((a, c, d));
    assert_eq!(contexts.bytes.load(Ordering::Acquire), 0);
    assert_eq!(contexts.active.load(Ordering::Acquire), 0);
    assert_eq!(contexts.json()["overload"]["memory"], 0);
}

/// The gate refuses a footprint the machine cannot hold, at the boundary:
/// room for exactly one context admits it, one byte less refuses it.
#[test]
fn the_machine_gate_refuses_a_context_it_cannot_hold() {
    const R: usize = 1 << 20;
    for (available, admitted) in [(R as u64 + 7, true), (R as u64 + 6, false)] {
        let registry = Registry::new("boundary");
        let contexts = registry.contexts(R, available);
        let taken = contexts.try_take(7);
        assert_eq!(taken.is_ok(), admitted, "available={available}");
        if !admitted {
            assert!(matches!(taken, Err(Overload::Memory(_))));
            assert_eq!(contexts.active.load(Ordering::Acquire), 0);
            assert_eq!(contexts.bytes.load(Ordering::Acquire), 0);
            assert_eq!(contexts.json()["overload"]["memory"], 1);
        }
    }
}

/// `/health`'s ceiling is the contexts in flight plus what the room and the
/// reservation's slack admit at an empty-body footprint.
#[test]
fn the_ceiling_is_derived_from_the_room_and_reported() {
    const R: usize = 1 << 20;
    let registry = Registry::new("ceiling");
    let contexts = registry.contexts(R, 5 * R as u64 + 3);
    let health = contexts.json();
    assert_eq!(health["ceiling"], 5, "{health}");
    assert_eq!(health["ceiling_basis"], "memory");
    assert_eq!(health["footprint_bytes"], R);
    let one = contexts.try_take(0).unwrap();
    assert_eq!(
        contexts.json()["ceiling"],
        6,
        "one in flight beside the room for five"
    );
    assert_eq!(contexts.json()["bytes"], R);
    drop(one);
    assert_eq!(
        contexts.json()["ceiling"],
        6,
        "the released context's slack stays reserved"
    );
    let off = ResponseContexts::default();
    assert_eq!(off.json()["ceiling"], Value::Null);
    assert_eq!(off.json()["ceiling_basis"], "gate-off");
}

/// Memory refusal at the HTTP boundary: 503 with `Retry-After`, the reason
/// named, before the model is asked for and with no context left behind.
#[test]
fn a_memory_refusal_is_a_503_with_retry_after_before_the_model() {
    let registry = Registry::new("http");
    let progress = Progress::default();
    let mut shared = scripted_shared_with_progress(Case::ApiLineage, progress.clone());
    Arc::get_mut(&mut shared).unwrap().responses = registry.contexts(1 << 20, 4096);
    let tiny_http::ListenAddr::IP(address) = shared.server.server_addr() else {
        panic!("IP test listener")
    };
    let body = json!({"messages":[{"role":"user","content":"p"}],"stream":true}).to_string();
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let request = shared
        .server
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    handle_model_request(
        &shared,
        "/v1/chat/completions",
        request,
        FirstToken::default(),
    )
    .unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(
        response.to_ascii_lowercase().contains("retry-after: 1\r\n"),
        "{response}"
    );
    assert!(
        response.contains("the machine cannot hold another response context"),
        "{response}"
    );
    assert_eq!(
        progress.0 .0.lock().model_entries,
        0,
        "refused before the model"
    );
    assert!(
        shared.turns.try_take().is_some(),
        "no model ticket left behind"
    );
    let health = shared.responses.json();
    assert_eq!(health["overload"]["memory"], 1, "{health}");
    assert_eq!(health["active"], 0);
    assert_eq!(health["bytes"], 0);
}

/// On a Metal device the context's footprint is committed on the device's
/// ledger for its life and admitted on the machine through the process's
/// reservation, as a pass is (host-only: a ledger, no device).
#[cfg(feature = "metal")]
#[test]
fn a_ledger_context_is_committed_for_its_life_and_admitted_on_the_machine() {
    use lloom_tensor::kernels::metal::ledger::Ledger;
    const R: usize = 4096;
    let registry = Registry::new("ledger");
    let ledger = Ledger::new();
    let contexts = ResponseContexts::with_memory(
        crate::options::ServerResponseConfig {
            contexts: None,
            raw_bytes: R,
        },
        ResponseMemory::Ledger {
            ledger: Arc::clone(&ledger),
            gate: registry.gate(1 << 30),
        },
    );
    // No load reserved anything: nothing to check, the bytes are still committed.
    let a = contexts.try_take(4).unwrap();
    assert_eq!(ledger.committed(), R + 4);
    drop(a);
    assert_eq!(ledger.committed(), 0);
    // With a load's reservation attached, past it the machine is asked: a
    // room too small for the growth refuses, and commits nothing.
    let small = registry.gate(R as u64 - 2);
    let load = small
        .admit(
            "the model's load",
            &lloom_tensor::machine::Peak::new().with("weights", 1),
        )
        .unwrap();
    let holding = load.holding.unwrap();
    ledger.attach_machine(&holding);
    assert!(matches!(contexts.try_take(0), Err(Overload::Memory(_))));
    assert_eq!(ledger.committed(), 0);
    assert_eq!(contexts.json()["overload"]["memory"], 1);
    drop(holding);
}

/// The writer never takes the split machine: while the decode thread holds
/// it (a slow observer decode), a batch already handed over is still
/// written.
#[test]
fn a_slow_observer_decode_does_not_hold_the_writer() {
    use std::sync::mpsc;
    let engine = Scripted::new(Case::CompleteMarker, Progress::default());
    let tags = engine.think_tags().unwrap();
    let machine = Arc::new(parking_lot::Mutex::new(
        reasoning::Reasoning::for_chat(&engine, &[tags.start], false)
            .unwrap()
            .unwrap(),
    ));
    let wire = OrderingWire::default();
    let sw = StreamWriter::start_with_batches(
        ChunkedWriter {
            w: Box::new(wire.clone()),
        },
        |b: &SplitDelta| {
            (
                format!(
                    "data: {}\n\n",
                    json!({"delta": reasoning::fields(&b.content, &b.reasoning)})
                ),
                b.raw,
            )
        },
    );
    {
        let mut m = machine.lock();
        let routed = m.publish("a");
        let (content, reasoning) = m.piece(routed);
        sw.delta_with(|b| {
            b.content.push_str(content);
            b.reasoning.push_str(reasoning);
            b.raw += routed.resolved;
        });
    }
    let (entered, entry) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let observer = {
        let machine = Arc::clone(&machine);
        std::thread::spawn(move || {
            let released = parking_lot::Mutex::new(released);
            // The debug build's oracle replays the decode: block only once.
            let once = std::sync::atomic::AtomicBool::new(false);
            machine
                .lock()
                .observe(&[tags.end], "</think>", 1, &|ids| {
                    if ids.is_empty() && !once.swap(true, Ordering::AcqRel) {
                        entered.send(()).unwrap();
                        released
                            .lock()
                            .recv_timeout(Duration::from_secs(5))
                            .unwrap();
                    }
                    Ok(if ids.is_empty() {
                        String::new()
                    } else {
                        "</think>".into()
                    })
                })
                .unwrap();
        })
    };
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    let deadline = Instant::now() + Duration::from_millis(250);
    let mut written = false;
    while Instant::now() < deadline {
        if String::from_utf8_lossy(&wire.0.lock()).contains(r#""reasoning_content":"a""#) {
            written = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        machine.is_locked(),
        "the observer held the machine throughout"
    );
    release.send(()).unwrap();
    observer.join().unwrap();
    let (_, error) = sw.finish().unwrap();
    assert!(error.is_none());
    assert!(written, "the writer waited for the observer's machine lock");
}
