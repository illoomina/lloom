//! The keep-warm's policy and its heartbeat's bound, on the real clock with
//! a counting action: the heartbeat acts while the window lasts, stops when
//! it runs out (no action past it), starts again on the next request, and
//! never stops without a bound. And its memory-pressure stop, on an
//! injected level: no action at warn or critical, one residency end per
//! spell, resumed at normal.
use super::*;
use parking_lot::Mutex;
use std::time::Instant;

fn clock() -> u64 {
    super::super::uptime_ns()
}

fn normal() -> u8 {
    1
}

/// An end that must not run: the level never rises.
fn no_end() -> Act {
    panic!("a residency end at normal pressure")
}

/// Waits up to 10 s for `cond`, so a loaded machine slows the test without
/// failing it.
fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// A heartbeat whose action records when it started.
fn counting(
    period_ms: u64,
    window_ms: Option<u64>,
) -> (Arc<Activity>, Heartbeat, Arc<Mutex<Vec<u64>>>) {
    let activity = Arc::new(Activity::new(clock()));
    let starts = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&starts);
    let beat = Heartbeat::spawn(
        Arc::clone(&activity),
        Duration::from_millis(period_ms),
        window_ms.map(Duration::from_millis),
        Beat {
            clock,
            pressure: normal,
            act: move || {
                s.lock().push(clock());
                Act::Done
            },
            end: no_end,
        },
    )
    .unwrap();
    (activity, beat, starts)
}

const MS: u64 = 1_000_000;

/// The bound: actions every period inside the window, none starting at or
/// past its end, the heartbeat stopped once it has run out; a request's end
/// starts it again for a new window; `stop` ends a parked heartbeat.
#[test]
fn the_heartbeat_stops_after_the_window_and_restarts_on_the_next_request() {
    let (period, window) = (10, 300);
    let (a, beat, starts) = counting(period, Some(window));
    let begun = a.last_request.load(Ordering::Acquire);
    wait_for("the first window to run out", || {
        a.windows.load(Ordering::Acquire) == 1
    });
    assert!(
        !a.running.load(Ordering::Acquire),
        "stopped after the window"
    );
    let first = starts.lock().clone();
    assert!(
        first.len() >= 3,
        "{} actions in a {window} ms window at {period} ms",
        first.len()
    );
    for t in &first {
        assert!(
            *t < begun + window * MS,
            "an action started {} ms after a {window} ms window began",
            (t - begun) / MS
        );
    }
    // Stopped: three windows' worth of time takes no action.
    std::thread::sleep(Duration::from_millis(3 * window));
    assert_eq!(
        starts.lock().len(),
        first.len(),
        "an action past the window"
    );
    assert_eq!(a.actions.load(Ordering::Relaxed), first.len() as u64);

    // A request ends: a new window, and the heartbeat runs again.
    let ended = clock();
    a.request_ended(ended);
    wait_for("the heartbeat to restart", || {
        starts.lock().len() >= first.len() + 3
    });
    assert!(
        starts.lock()[first.len()] > ended,
        "restarted by the request"
    );
    wait_for("the second window to run out", || {
        a.windows.load(Ordering::Acquire) == 2
    });
    for t in &starts.lock()[first.len()..] {
        assert!(
            *t < ended + window * MS,
            "an action started {} ms after the second window began",
            (t - ended) / MS
        );
    }
    let parked = Instant::now();
    beat.stop();
    assert!(
        parked.elapsed() < Duration::from_secs(5),
        "stop woke the parked heartbeat"
    );
}

/// No bound: the heartbeat keeps acting long past what a window would
/// allow, and no window ends.
#[test]
fn without_a_bound_the_heartbeat_never_stops() {
    let (a, beat, starts) = counting(10, None);
    std::thread::sleep(Duration::from_millis(600));
    let n = starts.lock().len();
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        starts.lock().len() > n,
        "the heartbeat stopped without a bound"
    );
    assert_eq!(a.windows.load(Ordering::Acquire), 0);
    assert!(a.running.load(Ordering::Acquire));
    beat.stop();
    assert!(!a.running.load(Ordering::Acquire));
}

/// A request in the window holds the slot: the heartbeat skips it, takes no
/// action, never tries one at or past the window's end (the GPU has had
/// nothing for longer than a period there, so only the window stops it), and
/// its window still runs out.
#[test]
fn a_busy_slot_takes_no_action_and_the_window_still_ends() {
    let window = 200;
    let activity = Arc::new(Activity::new(clock()));
    let begun = activity.last_request.load(Ordering::Acquire);
    let tries = Arc::new(Mutex::new(Vec::new()));
    let t = Arc::clone(&tries);
    let beat = Heartbeat::spawn(
        Arc::clone(&activity),
        Duration::from_millis(10),
        Some(Duration::from_millis(window)),
        Beat {
            clock,
            pressure: normal,
            act: move || {
                t.lock().push(clock());
                Act::Busy
            },
            end: no_end,
        },
    )
    .unwrap();
    wait_for("the window to run out", || {
        activity.windows.load(Ordering::Acquire) == 1
    });
    assert_eq!(activity.actions.load(Ordering::Relaxed), 0);
    let tries = tries.lock().clone();
    assert!(tries.len() >= 3, "{} tries in the window", tries.len());
    for t in &tries {
        assert!(
            *t < begun + window * MS,
            "a try {} ms after a {window} ms window began",
            (t - begun) / MS
        );
    }
    beat.stop();
}

/// The policy as resolved: serve's default, the opt-out, the bound's
/// spellings, the refusal at a batch and the CPU's off.
#[test]
fn the_policy_resolves_the_default_the_opt_out_and_the_refusal() {
    let on = Policy::resolve(Some(1000), 180, "residency", false, 1, true).unwrap();
    assert_eq!(
        on,
        Policy::On {
            period: Duration::from_millis(1000),
            window: Some(Duration::from_secs(180)),
            by: By::Residency,
            pressure: false,
        }
    );
    assert_eq!(on.short(), "1000ms/180s/residency");
    assert!(on.sentence().contains("for 180 s after the last request"));
    assert!(on.sentence().contains("memory pressure not read"));
    let stops = Policy::resolve(Some(1000), 180, "residency", true, 1, true).unwrap();
    assert_eq!(stops.short(), "1000ms/180s/residency/pressure");
    assert!(stops
        .sentence()
        .contains("stopped at memory pressure warn or critical"));
    assert_eq!(stops.json(&Activity::new(0), 0)["pressure_stop"], true);
    let a = Activity::new(5 * MS);
    let j = on.json(&a, 7 * MS);
    assert_eq!(j["policy"], "on");
    assert_eq!(j["period_ms"], 1000);
    assert_eq!(j["for_s"], 180);
    assert_eq!(j["by"], "residency");
    assert_eq!(j["state"]["since_request_ms"], 2);
    assert_eq!(j["state"]["heartbeat"], "stopped");

    let unbounded = Policy::resolve(Some(500), 0, "dispatch", false, 1, true).unwrap();
    assert_eq!(unbounded.short(), "500ms/unbounded/dispatch");
    assert_eq!(unbounded.json(&a, 0)["for_s"], Value::Null);

    let off = |p: Policy| match p {
        Policy::Off { why } => why,
        p => panic!("{p:?} is on"),
    };
    assert_eq!(
        off(Policy::resolve(Some(0), 180, "residency", false, 1, true).unwrap()),
        "server.keep_warm = 0"
    );
    assert_eq!(
        off(Policy::resolve(None, 180, "residency", false, 1, true).unwrap()),
        "server.keep_warm unset"
    );
    assert_eq!(
        off(Policy::resolve(None, 180, "residency", false, 4, true).unwrap()),
        "not built at --batch 4"
    );
    assert_eq!(
        off(Policy::resolve(Some(1000), 180, "residency", false, 1, false).unwrap()),
        "the device is not a GPU"
    );
    assert_eq!(
        Policy::resolve(Some(0), 180, "residency", false, 1, true)
            .unwrap()
            .short(),
        "off"
    );
    let refused = Policy::resolve(Some(1000), 180, "residency", false, 4, true).unwrap_err();
    assert!(refused.contains("--batch 4"), "{refused}");
}

/// The level the pressure tests' heartbeats read. Each test owns its
/// static: a `fn` seam carries no state.
static WARN_LEVEL: AtomicU8 = AtomicU8::new(1);
fn warn_level() -> u8 {
    WARN_LEVEL.load(Ordering::Acquire)
}

/// A heartbeat on `level`, counting its actions and its ends, with the slot
/// free (`ends_busy` false) or held for its ends.
fn pressured(
    level: fn() -> u8,
    ends_busy: Arc<AtomicBool>,
) -> (Arc<Activity>, Heartbeat, Arc<AtomicU64>, Arc<AtomicU64>) {
    let activity = Arc::new(Activity::new(clock()));
    let (acts, ends) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (a, e) = (Arc::clone(&acts), Arc::clone(&ends));
    let beat = Heartbeat::spawn(
        Arc::clone(&activity),
        Duration::from_millis(10),
        None,
        Beat {
            clock,
            pressure: level,
            act: move || {
                a.fetch_add(1, Ordering::Relaxed);
                Act::Done
            },
            end: move || {
                if ends_busy.load(Ordering::Acquire) {
                    return Act::Busy;
                }
                e.fetch_add(1, Ordering::Relaxed);
                Act::Done
            },
        },
    )
    .unwrap();
    (activity, beat, acts, ends)
}

/// At warn the heartbeat takes no action and ends the residency once; a
/// request in the spell makes it end again; at normal it acts again; each
/// change is one transition, which `/health` names.
#[test]
fn memory_pressure_stops_the_heartbeat_ends_residency_once_and_normal_resumes_it() {
    WARN_LEVEL.store(1, Ordering::Release);
    let (a, beat, acts, ends) = pressured(warn_level, Arc::new(AtomicBool::new(false)));
    wait_for("actions at normal", || acts.load(Ordering::Relaxed) >= 3);
    assert_eq!(ends.load(Ordering::Relaxed), 0, "an end at normal");

    WARN_LEVEL.store(2, Ordering::Release);
    wait_for("the end at warn", || ends.load(Ordering::Relaxed) == 1);
    let stopped = acts.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(150));
    assert!(
        acts.load(Ordering::Relaxed) <= stopped + 1,
        "{} actions at warn",
        acts.load(Ordering::Relaxed) - stopped
    );
    let at_warn = acts.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(acts.load(Ordering::Relaxed), at_warn, "an action at warn");
    assert_eq!(ends.load(Ordering::Relaxed), 1, "one end per spell");
    let j = Policy::On {
        period: Duration::from_millis(10),
        window: None,
        by: By::Residency,
        pressure: true,
    }
    .json(&a, clock());
    assert_eq!(j["state"]["pressure"], "warn", "{j}");
    assert_eq!(j["state"]["pressure_transition"]["from"], "normal", "{j}");
    assert_eq!(j["state"]["pressure_transition"]["to"], "warn", "{j}");
    assert_eq!(j["state"]["pressure_transitions"], 1, "{j}");
    assert_eq!(j["state"]["residency_ends"], 1, "{j}");

    // A request in the spell wired the set again: one more end, no action.
    a.request_ended(clock());
    wait_for("the end after a request", || {
        ends.load(Ordering::Relaxed) == 2
    });
    assert_eq!(acts.load(Ordering::Relaxed), at_warn, "an action at warn");

    // Critical is still a stop, and a change: no end without a request.
    WARN_LEVEL.store(4, Ordering::Release);
    wait_for("the change to critical", || {
        a.transitions.load(Ordering::Relaxed) == 2
    });
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ends.load(Ordering::Relaxed), 2, "an end with no request");
    assert_eq!(
        acts.load(Ordering::Relaxed),
        at_warn,
        "an action at critical"
    );

    WARN_LEVEL.store(1, Ordering::Release);
    wait_for("actions at normal again", || {
        acts.load(Ordering::Relaxed) >= at_warn + 3
    });
    assert_eq!(a.transitions.load(Ordering::Relaxed), 3);
    assert_eq!(a.ends.load(Ordering::Relaxed), 2);
    beat.stop();
}

static BUSY_LEVEL: AtomicU8 = AtomicU8::new(2);
fn busy_level() -> u8 {
    BUSY_LEVEL.load(Ordering::Acquire)
}

/// A held slot defers the end: no end while a request holds it, the end on
/// the first beat the slot is free, and still no action.
#[test]
fn under_pressure_a_busy_slot_defers_the_end_until_it_is_free() {
    BUSY_LEVEL.store(2, Ordering::Release);
    let busy = Arc::new(AtomicBool::new(true));
    let (a, beat, acts, ends) = pressured(busy_level, Arc::clone(&busy));
    wait_for("the change to warn", || {
        a.transitions.load(Ordering::Relaxed) == 1
    });
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ends.load(Ordering::Relaxed), 0, "an end on a held slot");
    busy.store(false, Ordering::Release);
    wait_for("the deferred end", || ends.load(Ordering::Relaxed) == 1);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ends.load(Ordering::Relaxed), 1, "one end per spell");
    assert_eq!(acts.load(Ordering::Relaxed), 0, "an action at warn");
    beat.stop();
}

/// The file-fed level reads the file's number and 0 when it is absent.
#[test]
fn the_pressure_file_is_read_each_beat() {
    let dir = std::env::temp_dir().join(format!("lloom-kw-pressure-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("level");
    PRESSURE_FILE.set(file.clone()).unwrap();
    assert_eq!(file_pressure(), 0, "no file reads as unknown");
    std::fs::write(&file, "2\n").unwrap();
    assert_eq!(file_pressure(), 2);
    std::fs::write(&file, "1").unwrap();
    assert_eq!(file_pressure(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The level names, and an unreadable level does not stop the heartbeat.
#[test]
fn the_pressure_levels_are_the_kernels_and_unknown_does_not_stop() {
    assert_eq!(Pressure::of(1), Pressure::Normal);
    assert_eq!(Pressure::of(2), Pressure::Warn);
    assert_eq!(Pressure::of(4), Pressure::Critical);
    assert_eq!(Pressure::of(0), Pressure::Unknown);
    assert_eq!(Pressure::of(3), Pressure::Unknown);
    assert!(Pressure::Warn.elevated() && Pressure::Critical.elevated());
    assert!(!Pressure::Normal.elevated() && !Pressure::Unknown.elevated());
    // The real reading: a documented level on macOS, unreadable elsewhere.
    let real = vm_pressure();
    if cfg!(target_os = "macos") {
        assert!(matches!(real, 1 | 2 | 4), "kern level {real}");
    } else {
        assert_eq!(real, 0);
    }
}

/// `GET`/`POST` on loopback, `Connection: close`; the body as JSON.
fn http(port: u16, method: &str, path: &str) -> std::io::Result<(u16, Value)> {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port))?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    let mut text = String::new();
    s.read_to_string(&mut text)?;
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text.split_once("\r\n\r\n").map_or("", |(_, b)| b);
    Ok((status, serde_json::from_str(body).unwrap_or(Value::Null)))
}

/// Through the accept loop: every model request's end restarts the window
/// (`/health`'s `since_request_ms` falls back), a `GET` does not, and on a
/// device that is not a GPU the policy is off and says why.
#[test]
fn a_requests_end_restarts_the_window_through_the_accept_loop() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut options = crate::EngineOptions::default();
    options.server.port = port;
    options.server.keep_warm = Some(1000);
    let engine = Box::new(crate::app::served::tests::NoVision::new(None));
    let served = std::thread::spawn(move || {
        super::super::run_loaded(options, engine, "stub".into()).map_err(|e| format!("{e:#}"))
    });
    let since = || -> u64 {
        let (status, h) = http(port, "GET", "/health").unwrap();
        assert_eq!(status, 200, "{h}");
        assert_eq!(h["keep_warm"]["policy"], "off", "{h}");
        assert_eq!(h["keep_warm"]["why"], "the device is not a GPU", "{h}");
        h["keep_warm"]["state"]["since_request_ms"]
            .as_u64()
            .unwrap()
    };
    wait_for("the server to answer", || {
        http(port, "GET", "/health").is_ok_and(|(s, _)| s == 200)
    });
    std::thread::sleep(Duration::from_millis(400));
    let loaded = since();
    assert!(loaded >= 400, "{loaded} ms since the load");
    let (status, _) = http(port, "POST", "/no-such-route").unwrap();
    assert_eq!(status, 404);
    let after = since();
    assert!(
        after < loaded && after < 400,
        "{after} ms since the request, {loaded} ms since the load before it"
    );
    std::thread::sleep(Duration::from_millis(100));
    assert!(since() >= after + 100, "a GET is not a request's end");
    let (status, _) = http(port, "POST", "/shutdown").unwrap();
    assert_eq!(status, 200);
    served.join().unwrap().unwrap();
}
