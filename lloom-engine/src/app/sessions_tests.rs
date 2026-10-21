//! The session table's contract (`design/docs/sessions.md` §3–§4), on its own:
//! ids, generations, the lifecycle, the binding and the bounds.

use super::*;
use crate::app::front::{Block, Reply, Scope};
use crate::Stop;

fn cfg(capacity: usize, one_sequence: bool) -> Config {
    Config {
        ttl: Duration::from_secs(60),
        capacity,
        ephemeral: 1,
        max_bytes: 1 << 20,
        one_sequence,
    }
}

fn reply(ids: &[u32]) -> Arc<Reply> {
    Arc::new(Reply {
        ids: ids.to_vec(),
        text: format!("{ids:?}"),
        block: Block::None,
        stop: Stop::Eos,
        content: 0,
        tail: 0,
        reasoning_text: String::new(),
        content_text: format!("{ids:?}"),
        partial: 0,
        generated: ids.len(),
        logprobs: None,
        rows: None,
    })
}

/// One turn: the request's prompt ids and messages (`n` of them, scoped
/// `0..=n`), reserved and kept with `reply_ids` as its reply.
#[allow(clippy::too_many_arguments)]
fn turn(
    t: &mut Table,
    continued: Option<(Key, usize)>,
    store: Option<bool>,
    auth: Option<[u8; 32]>,
    prompt: &[u32],
    scopes: &[Scope],
    reply_ids: &[u32],
    now: Instant,
) -> WireId {
    let r = t.reserve(
        Plan {
            continued,
            store,
            auth,
            prompt_ids: prompt,
            scopes,
        },
        now,
    );
    let n = scopes.len() - 1;
    let kept = t.commit(
        r,
        Record {
            messages: (0..n)
                .map(|i| json!({ "role": "user", "content": i }))
                .collect(),
            reply_msg: json!({ "role": "assistant", "content": "r" }),
            prompt_ids: prompt.to_vec(),
            reply: reply(reply_ids),
            eot: None,
            scope: scopes[n],
            prefix: None,
        },
        now,
    );
    assert!(kept, "the reserved turn is kept");
    r.id
}

fn scopes(n: u8) -> Vec<Scope> {
    (0..=n).map(Scope::test).collect()
}

#[test]
fn an_id_round_trips_and_nothing_else_parses() {
    let id = WireId {
        key: Key { slot: 3, gen: 7 },
        turn: 2,
        nonce: 0xdead_beef_0123_4567,
    };
    let s = id.encode();
    assert_eq!(s.len(), "chatcmpl-".len() + 40, "{s}");
    assert_eq!(WireId::parse(&s), Some(id));
    assert_eq!(id.short(), "3:7:2", "the line form carries no nonce");
    for bad in [
        "chatcmpl-0123456789abcdef",
        "",
        "chatcmpl-",
        &s[..s.len() - 1],
        &format!("{s}0"),
        &s.replace("chatcmpl-", "cmpl-"),
        &format!("chatcmpl-{}", "g".repeat(40)),
    ] {
        assert_eq!(WireId::parse(bad), None, "{bad:?}");
    }
}

#[test]
fn a_stored_session_is_resident_until_the_engine_serves_another() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2, 3],
        &scopes(1),
        &[4, 5],
        now,
    );
    let r = t.resolve(&id.encode(), None, false, now).expect("held");
    assert_eq!(r.resident, Some(true));
    assert!(r.stored);
    assert_eq!(r.ids, vec![1, 2, 3, 4, 5], "the prompt, then the reply");
    assert_eq!(r.bound.len(), 1);
    assert_eq!(r.switches, 0);
    // Another route ran on the engine's one sequence.
    t.note_other();
    let r = t
        .resolve(&id.encode(), None, false, now)
        .expect("still held");
    assert_eq!(r.resident, Some(false), "evicted, not expired");
    assert_eq!(r.switches, 1, "one switch counted");
    assert_eq!(t.json()["switches"], 1);
}

#[test]
fn a_stale_generation_misses_and_never_lands_in_the_session_that_took_the_slot() {
    let mut t = Table::new(cfg(1, true));
    let now = Instant::now();
    let a = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2],
        &scopes(1),
        &[9],
        now,
    );
    let b = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[7, 8],
        &scopes(1),
        &[9],
        now,
    );
    assert_eq!(a.key.slot, b.key.slot, "one stored slot: B took A's");
    assert_ne!(a.key.gen, b.key.gen, "and moved its generation");
    // A's id, even with B's nonce and turn, names nothing.
    let forged = WireId { key: a.key, ..b };
    for id in [a, forged] {
        assert_eq!(
            t.resolve(&id.encode(), None, false, now).err(),
            Some(Miss::Stale),
            "{id:?}"
        );
    }
    let r = t.resolve(&b.encode(), None, false, now).expect("B is held");
    assert_eq!(r.ids, vec![7, 8, 9]);
}

#[test]
fn a_wrong_nonce_or_a_turn_past_the_last_misses() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let wrong = WireId {
        nonce: id.nonce ^ 1,
        ..id
    };
    let past = WireId { turn: 1, ..id };
    for x in [wrong, past] {
        assert_eq!(
            t.resolve(&x.encode(), None, false, now).err(),
            Some(Miss::Turn),
            "{x:?}"
        );
    }
    assert!(t.resolve(&id.encode(), None, false, now).is_ok());
}

#[test]
fn a_session_bound_to_a_key_is_not_found_without_it() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let key = Some([5u8; 32]);
    let id = turn(&mut t, None, Some(true), key, &[1], &scopes(1), &[2], now);
    for other in [None, Some([6u8; 32])] {
        assert_eq!(
            t.resolve(&id.encode(), other, false, now).err(),
            Some(Miss::Binding)
        );
    }
    assert!(t.resolve(&id.encode(), key, false, now).is_ok());
}

#[test]
fn an_ephemeral_session_lasts_while_the_engine_holds_it() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(&mut t, None, None, None, &[1, 2], &scopes(1), &[3], now);
    assert!(id.key.slot as usize >= 4, "an ephemeral slot");
    let r = t.resolve(&id.encode(), None, false, now).expect("resident");
    assert!(!r.stored);
    assert_eq!(r.resident, Some(true));
    t.note_other();
    assert_eq!(
        t.resolve(&id.encode(), None, false, now).err(),
        Some(Miss::Gone),
        "its state is gone and nothing else kept it"
    );
}

#[test]
fn store_on_a_continuation_promotes_an_ephemeral_session() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let a = turn(&mut t, None, None, None, &[1], &scopes(1), &[2], now);
    let r = t.resolve(&a.encode(), None, false, now).unwrap();
    let b = turn(
        &mut t,
        Some((r.key, r.turn)),
        Some(true),
        None,
        &[1, 2, 3],
        &scopes(3),
        &[4],
        now,
    );
    assert!((b.key.slot as usize) < 4, "moved to a stored slot");
    assert_eq!(b.turn, 1, "the next turn of the same lineage");
    assert_eq!(
        t.resolve(&a.encode(), None, false, now).err(),
        Some(Miss::Stale),
        "the ephemeral slot's ids are gone"
    );
    let r = t.resolve(&b.encode(), None, false, now).unwrap();
    assert!(r.stored);
    assert_eq!(r.bound.len(), 2, "both turns' replies, bound");
    t.note_other();
    assert!(
        t.resolve(&b.encode(), None, false, now).is_ok(),
        "stored: kept past the engine's state"
    );
}

#[test]
fn a_continuation_that_leaves_a_turn_drops_it_and_its_id() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let t0 = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2],
        &scopes(1),
        &[3],
        now,
    );
    let r0 = t.resolve(&t0.encode(), None, false, now).unwrap();
    let t1 = turn(
        &mut t,
        Some((r0.key, r0.turn)),
        None,
        None,
        &[1, 2, 3, 4],
        &scopes(3),
        &[5],
        now,
    );
    assert_eq!(t1.turn, 1);
    // Continue turn 0 again, differently: a fork at turn 0's end — another
    // new message, so another conversation from there.
    let r0 = t.resolve(&t0.encode(), None, false, now).unwrap();
    let forked = [
        Scope::test(0),
        Scope::test(1),
        Scope::test(20),
        Scope::test(30),
    ];
    let fork = turn(
        &mut t,
        Some((r0.key, r0.turn)),
        None,
        None,
        &[1, 2, 3, 6],
        &forked,
        &[7],
        now,
    );
    assert_eq!(fork.turn, 1, "the fork takes turn 1's place");
    assert_eq!(
        t.resolve(&t1.encode(), None, false, now).err(),
        Some(Miss::Turn),
        "the dropped turn's id misses"
    );
    assert!(t.resolve(&t0.encode(), None, false, now).is_ok());
    let r = t.resolve(&fork.encode(), None, false, now).unwrap();
    assert_eq!(r.ids, vec![1, 2, 3, 6, 7]);
    // A history that differs before turn 0's reply keeps no turn.
    let r = t.resolve(&fork.encode(), None, false, now).unwrap();
    let other = [Scope::test(9), Scope::test(10)];
    let x = turn(
        &mut t,
        Some((r.key, r.turn)),
        None,
        None,
        &[1, 2, 3],
        &other,
        &[8],
        now,
    );
    assert_eq!(x.turn, 0, "the lineage starts anew");
}

/// The same conversation again (a regeneration): its last turn's reply is
/// not in the request, so the new reply takes that turn's place; and a turn
/// whose reply the request carries stays, even where the request's prompt
/// renders it to other ids — its ids are then known only as far as the
/// prompt agrees.
#[test]
fn a_regeneration_replaces_the_last_turn_and_a_rerendered_turn_stays() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let t0 = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2],
        &scopes(1),
        &[3],
        now,
    );
    let r0 = t.resolve(&t0.encode(), None, false, now).unwrap();
    let t1 = turn(
        &mut t,
        Some((r0.key, 0)),
        None,
        None,
        &[1, 2, 3, 4],
        &scopes(3),
        &[5],
        now,
    );
    let r1 = t.resolve(&t1.encode(), None, false, now).unwrap();
    // The same messages as turn 1's request: regenerate its reply.
    let again = turn(
        &mut t,
        Some((r1.key, 0)),
        None,
        None,
        &[1, 2, 3, 4],
        &scopes(3),
        &[6],
        now,
    );
    assert_eq!(again.turn, 1, "turn 1's place");
    assert_eq!(
        t.resolve(&t1.encode(), None, false, now).err(),
        Some(Miss::Turn)
    );
    // Turn 0's reply re-rendered to other ids (the prompt leaves the log at
    // 2): turn 0 stays, its ids known to 2.
    let r = t.resolve(&again.encode(), None, false, now).unwrap();
    let next = turn(
        &mut t,
        Some((r.key, r.turn)),
        None,
        None,
        &[1, 2, 9, 4, 6],
        &scopes(5),
        &[7],
        now,
    );
    assert_eq!(next.turn, 2, "both turns stay");
    let r0 = t.resolve(&t0.encode(), None, false, now).unwrap();
    assert_eq!(
        r0.ids,
        vec![1, 2],
        "turn 0's ids known as far as the prompt agrees"
    );
}

#[test]
fn a_session_unused_past_its_ttl_expires() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let later = now + Duration::from_secs(61);
    assert_eq!(
        t.resolve(&id.encode(), None, false, later).err(),
        Some(Miss::Ttl)
    );
    assert_eq!(t.json()["stored"], 0);
}

#[test]
fn over_the_byte_budget_the_least_recently_used_other_session_goes() {
    let mut t = Table::new(Config {
        max_bytes: 64,
        ..cfg(4, true)
    });
    let now = Instant::now();
    let a = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1; 4],
        &scopes(1),
        &[2],
        now,
    );
    let b = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[3; 4],
        &scopes(1),
        &[4],
        now + Duration::from_millis(1),
    );
    assert!(t.resolve(&a.encode(), None, false, now).is_err(), "A went");
    assert!(
        t.resolve(&b.encode(), None, false, now).is_ok(),
        "B, being kept, stays"
    );
}

#[test]
fn a_reset_drops_the_turns_and_the_next_turn_starts_the_lineage() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let a = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2],
        &scopes(1),
        &[3],
        now,
    );
    let r = t.resolve(&a.encode(), None, false, now).unwrap();
    t.reset(r.key);
    assert_eq!(
        t.resolve(&a.encode(), None, false, now).err(),
        Some(Miss::Turn)
    );
    let b = turn(
        &mut t,
        Some((r.key, r.turn)),
        None,
        None,
        &[1, 9],
        &scopes(1),
        &[3],
        now,
    );
    assert_eq!((b.key, b.turn), (a.key, 0), "the same session, turn 0");
}

#[test]
fn delete_forgets_a_session_and_inspect_shows_no_content() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let a = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[1, 2],
        &scopes(1),
        &[3],
        now,
    );
    let seen = t.inspect(&a.encode(), None, now).expect("held");
    assert_eq!(seen["turns"], 1);
    assert_eq!(seen["ids"], 3);
    assert!(
        seen.get("messages").is_some_and(Value::is_number),
        "a count, never the messages"
    );
    assert!(t.delete(&a.encode(), None, now));
    assert!(t.inspect(&a.encode(), None, now).is_none());
    assert!(!t.delete(&a.encode(), None, now));
}

#[test]
fn a_concurrent_turn_on_one_session_keeps_only_the_first() {
    let mut t = Table::new(cfg(4, false));
    let now = Instant::now();
    let a = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let r = t.resolve(&a.encode(), None, false, now).unwrap();
    assert_eq!(r.resident, None, "several sequences: the run says");
    let plan = |p: &'static [u32]| Plan {
        continued: Some((r.key, r.turn)),
        store: None,
        auth: None,
        prompt_ids: p,
        scopes: &[],
    };
    let s3 = scopes(3);
    let x = t.reserve(
        Plan {
            scopes: &s3,
            ..plan(&[1, 2, 3])
        },
        now,
    );
    let y = t.reserve(
        Plan {
            scopes: &s3,
            ..plan(&[1, 2, 4])
        },
        now,
    );
    assert_eq!(x.id.turn, y.id.turn);
    let rec = |p: Vec<u32>| Record {
        messages: Vec::new(),
        reply_msg: Value::Null,
        prompt_ids: p,
        reply: reply(&[5]),
        eot: None,
        scope: Scope::test(3),
        prefix: None,
    };
    assert!(t.commit(x, rec(vec![1, 2, 3]), now));
    assert!(
        !t.commit(y, rec(vec![1, 2, 4]), now),
        "the second names nothing"
    );
    assert!(t.resolve(&y.id.encode(), None, false, now).is_err());
}
