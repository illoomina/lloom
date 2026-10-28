//! The turn queue alone: no socket, no model. A body names its request
//! (`{"n": k}`); the run answers each with P = k / 100 so an answer can be
//! traced back to the body it belongs to.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn body(n: u32) -> OrderedJson {
    OrderedJson::from_slice(format!("{{\"n\": {n}}}").as_bytes()).unwrap()
}

fn number(body: &OrderedJson) -> u32 {
    match body.get("n") {
        Some(OrderedJson::Number(n)) => n.to_string().parse().unwrap(),
        other => panic!("not a test body: {other:?}"),
    }
}

fn answer(body: &OrderedJson) -> Answer {
    SystemOneResponse::new(
        vec![(
            "q".into(),
            crate::decision::Answer::Noul {
                probability: f64::from(number(body)) / 100.0,
            },
        )],
        u64::from(number(body)),
    )
    .map_err(|e| SystemOneError::Execution(e.to_string()))
}

fn traced(answer: &Answer) -> u64 {
    answer.as_ref().expect("an answer").input_tokens()
}

/// The group sizes `run` saw, in run order.
#[derive(Default)]
struct Runs(parking_lot::Mutex<Vec<Vec<u32>>>);
impl Runs {
    fn run(&self, bodies: &[&OrderedJson]) -> Vec<Answer> {
        self.0
            .lock()
            .push(bodies.iter().map(|b| number(b)).collect());
        bodies.iter().map(|b| answer(b)).collect()
    }
}

/// Spin until `queued` decision bodies wait in groups (a test-only read of
/// the queue's state), bounded so a broken queue fails rather than hangs.
fn wait_queued(turns: &Turns, queued: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let waiting: usize = turns
            .state
            .lock()
            .groups
            .values()
            .map(|g| g.bodies.len())
            .sum();
        if waiting == queued {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{waiting} of {queued} bodies queued"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_lone_decision_is_a_group_of_one() {
    let (turns, runs) = (Turns::default(), Runs::default());
    let got = turns.decide(&body(7), true, |b| runs.run(b));
    assert_eq!(traced(&got), 7);
    assert_eq!(*runs.0.lock(), vec![vec![7]]);
}

#[test]
fn decisions_that_arrive_while_the_slot_is_busy_run_as_one_group() {
    let turns = Arc::new(Turns::default());
    let runs = Arc::new(Runs::default());
    let busy = turns.take();
    let threads: Vec<_> = (1..=3)
        .map(|n| {
            let (t, r) = (turns.clone(), runs.clone());
            let handle = std::thread::spawn(move || t.decide(&body(n), true, |b| r.run(b)));
            wait_queued(&turns, n as usize);
            handle
        })
        .collect();
    drop(busy);
    let got: Vec<u64> = threads
        .into_iter()
        .map(|t| traced(&t.join().unwrap()))
        .collect();
    assert_eq!(got, vec![1, 2, 3], "an answer went to another request");
    assert_eq!(
        *runs.0.lock(),
        vec![vec![1, 2, 3]],
        "the waiting decisions were not grouped"
    );
    assert!(turns.state.lock().groups.is_empty());
}

#[test]
fn without_join_each_decision_runs_alone_in_arrival_order() {
    let turns = Arc::new(Turns::default());
    let runs = Arc::new(Runs::default());
    let busy = turns.take();
    let threads: Vec<_> = (1..=3)
        .map(|n| {
            let (t, r) = (turns.clone(), runs.clone());
            let handle = std::thread::spawn(move || t.decide(&body(n), false, |b| r.run(b)));
            wait_queued(&turns, n as usize);
            handle
        })
        .collect();
    drop(busy);
    for (n, t) in (1..=3).zip(threads) {
        assert_eq!(traced(&t.join().unwrap()), n);
    }
    assert_eq!(*runs.0.lock(), vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn a_plain_turn_waits_for_the_group_ahead_of_it() {
    let turns = Arc::new(Turns::default());
    let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let busy = turns.take();
    let decision = {
        let (turns, order) = (turns.clone(), order.clone());
        std::thread::spawn(move || {
            turns.decide(&body(1), true, |b| {
                order.lock().push("decision");
                b.iter().map(|b| answer(b)).collect()
            })
        })
    };
    wait_queued(&turns, 1);
    let chat = {
        let (turns, order) = (turns.clone(), order.clone());
        std::thread::spawn(move || {
            let _turn = turns.take();
            order.lock().push("chat");
        })
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while turns.state.lock().next < 3 {
        assert!(Instant::now() < deadline, "the chat never took a ticket");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(busy);
    decision.join().unwrap().unwrap();
    chat.join().unwrap();
    assert_eq!(*order.lock(), vec!["decision", "chat"]);
}

#[test]
fn a_failed_group_answers_its_joiners_and_frees_the_slot() {
    let turns = Arc::new(Turns::default());
    let ran = Arc::new(AtomicUsize::new(0));
    let busy = turns.take();
    let leader = {
        let turns = turns.clone();
        std::thread::spawn(move || {
            turns.decide(&body(1), true, |_| panic!("intentional group failure"))
        })
    };
    wait_queued(&turns, 1);
    let joiner = {
        let (turns, ran) = (turns.clone(), ran.clone());
        std::thread::spawn(move || {
            turns.decide(&body(2), true, |b| {
                ran.fetch_add(1, Ordering::Relaxed);
                b.iter().map(|b| answer(b)).collect()
            })
        })
    };
    wait_queued(&turns, 2);
    drop(busy);
    assert!(leader.join().is_err(), "the leader's run did not unwind");
    let joined = joiner.join().unwrap();
    assert!(
        matches!(&joined, Err(SystemOneError::Execution(m)) if m.contains("did not complete")),
        "{joined:?}"
    );
    assert_eq!(
        ran.load(Ordering::Relaxed),
        0,
        "the joiner ran its own group"
    );
    let _next = turns.take();
    assert!(turns.state.lock().groups.is_empty());
}

/// The keep-warm's turn: taken only when the slot is free and nobody waits,
/// so it never runs beside a request nor goes ahead of a queued one.
#[test]
fn try_take_never_waits_and_never_jumps_the_queue() {
    let turns = Arc::new(Turns::default());
    let warm = turns.try_take().expect("a free slot is taken");
    assert!(turns.try_take().is_none(), "a held slot is not taken twice");
    drop(warm);

    let busy = turns.take();
    let (started, has_started) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let queued = {
        let turns = turns.clone();
        std::thread::spawn(move || {
            let _turn = turns.take();
            started.send(()).unwrap();
            released.recv().unwrap();
        })
    };
    // Tickets 0 (the first try_take) and 1 (`busy`) are taken; the queued
    // request's is 2.
    let deadline = Instant::now() + Duration::from_secs(10);
    while turns.state.lock().next < 3 {
        assert!(Instant::now() < deadline, "the request never took a ticket");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(turns.try_take().is_none(), "a request holds the slot");
    drop(busy);
    assert!(
        turns.try_take().is_none(),
        "the keep-warm went ahead of the queued request"
    );
    has_started.recv().unwrap();
    assert!(
        turns.try_take().is_none(),
        "the queued request holds the slot"
    );
    release.send(()).unwrap();
    queued.join().unwrap();
    assert!(turns.try_take().is_some(), "the slot is free again");
}
