//! Actual loaded Metal owner through the response path. Run only under an
//! admitted native batch. No synchronization substitutes for dependency safety.
use super::stream_tail_tests::{
    held_writer::{Cleanup, PausedWriter},
    message_values, shared_with_engine,
};
use super::*;
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use parking_lot::Mutex as ParkingMutex;
use std::{path::PathBuf, sync::mpsc, time::Duration};

#[derive(Clone, Copy, Debug)]
enum Release {
    Publication,
    FullSettlement,
}

struct Checkpoint(PathBuf);
impl Drop for Checkpoint {
    fn drop(&mut self) {
        // Only this test's unique fixture, after the loaded owner has dropped.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn native_owner() -> Result<(Checkpoint, Arc<Shared>, &'static str)> {
    // TinyCheckpoint::write constructs a CPU schema model which fixes model
    // tuning/regime. Install both halves FIRST; the later model install fails.
    let mut options = crate::EngineOptions::default();
    options.load.device = "auto".into();
    options.load.warmup = false;
    options.memory.kv_capacity = 256;
    options.prefix.reuse = Some(true);
    options.speculative.method = "none".into();
    // A single deterministic owner; no process-env mutation or default activation.
    options.tuning.contract.profile = "deterministic".into();
    let api = std::env::var("LLOOM_METAL_API").unwrap_or_else(|_| "metal3".into());
    anyhow::ensure!(
        matches!(api.as_str(), "metal3" | "metal4"),
        "native fixture needs explicit metal3/metal4, not {api}"
    );
    options.tuning.metal.command_api = api;
    options.tuning.install().map_err(anyhow::Error::msg)?;
    let path = std::env::temp_dir().join(format!(
        "lloom-response-native-{}-{}",
        std::process::id(),
        short_id()
    ));
    std::fs::create_dir(&path)?;
    let checkpoint = Checkpoint(path);
    let mut cfg = Lfm2Tiny::template();
    TinyConfig::set_axis(&mut cfg, "vocab_size", Lfm2Tiny::tokenizer().vocab_size());
    TinyConfig::derive(&mut cfg);
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg, Weights::default()).write(&checkpoint.0)?;
    options.load.model = Some(model);
    let shape = super::super::Shape::of(&options, Frontend::Serve, String::new());
    let (engine, _) = super::super::load_engine(&options.load, shape)?;
    assert!(
        engine.device().is_metal(),
        "native test must not silently fall back to CPU"
    );
    let actual_api = engine
        .device()
        .as_metal_device()?
        .commands()
        .actual_command_api();
    assert_eq!(
        actual_api, options.tuning.metal.command_api,
        "explicit native API must not silently change backend"
    );
    eprintln!(
        "NATIVE_RESPONSE_OWNER actual_api={actual_api} selected_readback={}",
        engine.readback().name()
    );
    Ok((checkpoint, shared_with_engine(engine), actual_api))
}

fn request(fields: Value) -> Value {
    let mut body = json!({"max_tokens":8,"temperature":0,"seed":7,"ignore_eos":true,
        "enable_thinking":false,"return_token_ids":true,
        "logit_bias":{"97":100,"98":100,"99":100}});
    for (key, value) in fields.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn json_reply(shared: &Shared, body: Value) -> Value {
    let _turn = shared.turns.take();
    let reply = chat_completions(shared, &request(body), None)
        .ok()
        .expect("native JSON request built");
    let Reply::Json(body) = reply else {
        panic!("JSON requested")
    };
    assert_eq!(body["choices"][0]["token_ids"].as_array().unwrap().len(), 8);
    body
}

fn events(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .unwrap()
        .split("data: ")
        .skip(1)
        .filter_map(|event| {
            let payload = event.split_once("\n\n").unwrap().0;
            (payload != "[DONE]").then(|| serde_json::from_str(payload).unwrap())
        })
        .collect()
}

#[derive(Debug)]
struct Saved {
    id: String,
    prompt: Vec<u32>,
    reply: Value,
    messages: Vec<Value>,
    ids: Vec<u32>,
}
fn saved(shared: &Shared, id: &str, prompt: Vec<u32>, reply: Value) -> Saved {
    let resolved = shared
        .sessions
        .lock()
        .resolve(id, None, true, Instant::now())
        .unwrap();
    Saved {
        id: id.into(),
        prompt,
        reply,
        messages: message_values(resolved.messages.unwrap()),
        ids: resolved.ids,
    }
}

fn publish(shared: &Shared, body: Value) -> Saved {
    let _drain = shared
        .drains
        .try_take()
        .expect("bounded B/C response context");
    let by_id = body.get("previous_response_id").is_some();
    let mut turn = Some(shared.turns.take());
    let Reply::Stream(run) = chat_completions(shared, &request(body), None)
        .ok()
        .expect("native stream built")
    else {
        panic!("stream requested")
    };
    if by_id {
        assert!(run.facts.spliced.bound > 0, "native B really binds A by ID");
    }
    let (id, prompt) = (run.id.clone(), run.prompt_ids.clone());
    let bytes = Arc::new(ParkingMutex::new(Vec::new()));
    struct Capture(Arc<ParkingMutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    stream_response_settlement(
        shared,
        *run,
        || {
            Ok(ChunkedWriter {
                w: Box::new(Capture(bytes.clone())),
            })
        },
        || drop(turn.take()),
    )
    .unwrap();
    let all = events(&bytes.lock());
    let last = all
        .iter()
        .find(|event| event.get("timings").is_some())
        .unwrap();
    assert_eq!(last["id"], id);
    let reply = last["choices"][0]["token_ids"].clone();
    assert_eq!(reply.as_array().unwrap().len(), 8);
    saved(shared, &id, prompt, reply)
}

fn assert_reference(result: &Saved, reference: &Value) {
    assert_eq!(json!(result.prompt), reference["prompt_token_ids"]);
    assert_eq!(result.reply, reference["choices"][0]["token_ids"]);
    assert_eq!(
        result.messages.last().unwrap(),
        &reference["choices"][0]["message"]
    );
}

/// Same owner and accepted history in all arms; only handler ticket lifetime
/// differs. The reference retains it through the ENTIRE settlement return.
fn scenario(
    shared: &Arc<Shared>,
    actual_api: &str,
    release_at: Release,
    late_failure: bool,
    reference_a: &Value,
    reference_b: &Value,
    reference_c: &Value,
) {
    let bytes = Arc::new(ParkingMutex::new(Vec::new()));
    let (entered, entry) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (published, publication) = mpsc::channel();
    let owner = shared.clone();
    let captured = bytes.clone();
    let thread = std::thread::spawn(move || {
        let _drain = owner.drains.try_take().expect("bounded A response context");
        let mut ticket = Some(owner.turns.take());
        let Reply::Stream(run) = chat_completions(&owner, &request(json!({
            "messages":[{"role":"user","content":"A"}],"store":true,"stream":true,"reset":"cache"
        })), None).ok().expect("A built") else { panic!("stream requested") };
        let id = run.id.clone();
        let result = stream_response_settlement(
            &owner,
            *run,
            || {
                Ok(ChunkedWriter {
                    w: Box::new(PausedWriter {
                        entered: Some(entered),
                        released,
                        bytes: captured,
                        late_failure,
                        max_pause: Duration::from_secs(120),
                    }),
                })
            },
            || {
                if matches!(release_at, Release::Publication) {
                    drop(ticket.take());
                }
                published.send(id).unwrap();
            },
        );
        // FullSettlement release is deliberately after function return, not
        // merely worker join; error return follows the same ownership rule.
        drop(ticket.take());
        result
    });
    let mut cleanup = Cleanup {
        release: Some(release),
        thread: Some(thread),
    };
    entry
        .recv_timeout(Duration::from_secs(60))
        .expect("A writer reached barrier");
    let id_a = publication
        .recv_timeout(Duration::from_secs(60))
        .expect("A accepted history published");
    {
        let a = shared
            .sessions
            .lock()
            .resolve(&id_a, None, true, Instant::now())
            .unwrap();
        assert_eq!(
            message_values(a.messages.unwrap()).last().unwrap(),
            &reference_a["choices"][0]["message"]
        );
        // Do not keep Bound/Reply/native-row owners from this lookup alive
        // across B/C; only the production response context may retain A.
    }
    assert_eq!(
        shared.drains.active.load(Ordering::Acquire),
        1,
        "A retains only its response context through drain"
    );
    match release_at {
        Release::Publication => assert!(
            shared.turns.try_take().is_some(),
            "published owner released ticket"
        ),
        Release::FullSettlement => assert!(
            shared.turns.try_take().is_none(),
            "reference holds ticket through transport"
        ),
    }
    let finish = |cleanup: &mut Cleanup| {
        let result = cleanup.finish();
        if late_failure {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("late A writer rejection"));
        } else {
            result.unwrap();
        }
    };
    if matches!(release_at, Release::FullSettlement) {
        finish(&mut cleanup);
    }
    let b = publish(
        shared,
        json!({"previous_response_id":id_a,"input":[{"role":"user","content":"B"}],"stream":true}),
    );
    let c = publish(
        shared,
        json!({"messages":[{"role":"user","content":"C"}],"store":true,"stream":true}),
    );
    assert_reference(&b, reference_b);
    assert_reference(&c, reference_c);
    assert_eq!(b.messages.len(), 4);
    assert_eq!(c.messages.len(), 2);
    assert_eq!(
        shared.drains.active.load(Ordering::Acquire),
        usize::from(matches!(release_at, Release::Publication))
    );
    if matches!(release_at, Release::Publication) {
        finish(&mut cleanup);
    }
    for snapshot in [&b, &c] {
        let after = shared
            .sessions
            .lock()
            .resolve(&snapshot.id, None, true, Instant::now())
            .unwrap();
        assert_eq!(after.ids, snapshot.ids);
        assert_eq!(message_values(after.messages.unwrap()), snapshot.messages);
        assert_eq!(
            after.resident,
            Some(snapshot.id == c.id),
            "late A cannot replace C residency"
        );
    }
    assert_eq!(
        shared.drains.active.load(Ordering::Acquire),
        0,
        "all response contexts released"
    );
    let raw = bytes.lock();
    let all = events(&raw);
    assert!(!all.is_empty());
    assert!(all.iter().all(|event| event["id"] == id_a));
    if late_failure {
        assert!(all.iter().all(|event| event.get("timings").is_none()));
        assert!(!std::str::from_utf8(&raw).unwrap().contains("[DONE]"));
    } else {
        let last = all
            .iter()
            .find(|event| event.get("timings").is_some())
            .unwrap();
        assert_eq!(last["timings"]["session"]["id"], id_a);
        assert_eq!(
            last["choices"][0]["token_ids"],
            reference_a["choices"][0]["token_ids"]
        );
        assert!(std::str::from_utf8(&raw).unwrap().contains("[DONE]"));
    }
    eprintln!("NATIVE_RESPONSE_CUSTODY device=metal api={} release={release_at:?} late_failure={late_failure} b_ids={} c_ids={} c_resident=true", actual_api, b.ids.len(), c.ids.len());
}

#[test]
fn metal_lfm2_held_writer_and_full_settlement_reference_keep_identical_accepted_history(
) -> Result<()> {
    let (checkpoint, shared, actual_api) = native_owner()?;
    let user_a = json!({"role":"user","content":"A"});
    let a = json_reply(&shared, json!({"messages":[user_a],"reset":"cache"}));
    let b = json_reply(
        &shared,
        json!({"messages":[user_a,a["choices"][0]["message"],{"role":"user","content":"B"}],"reset":"cache"}),
    );
    let c = json_reply(
        &shared,
        json!({"messages":[{"role":"user","content":"C"}],"reset":"cache"}),
    );
    assert!(!a["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
    for release in [Release::Publication, Release::FullSettlement] {
        for late_failure in [false, true] {
            scenario(&shared, actual_api, release, late_failure, &a, &b, &c);
        }
    }
    drop(shared);
    drop(checkpoint);
    Ok(())
}
