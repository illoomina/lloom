//! The session table's contract (`design/docs/sessions.md` §3–§4), on its own:
//! ids, generations, the lifecycle, the binding and the bounds.

use super::*;
use crate::app::front::{Block, Reply, Scope};
use crate::Stop;

fn cfg(capacity: usize, one_sequence: bool) -> Config {
    Config {
        idle: Duration::from_secs(60),
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
    assert_eq!(t.json()["switches"], 0, "a lookup counts nothing by itself");
    // Another route ran on the engine's one sequence.
    t.note_other();
    let r = t
        .resolve(&id.encode(), None, false, now)
        .expect("still held");
    assert_eq!(r.resident, Some(false), "evicted, not expired");
    assert_eq!(
        t.note_continued(r.key, false),
        1,
        "one switch, once the turn ran"
    );
    assert_eq!(t.json()["switches"], 1);
    assert_eq!(t.json()["evicted"], 1);
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
    // A's id names nothing, and says why to its holder: B took the one slot.
    assert_eq!(
        t.resolve(&a.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Capacity)
        }),
    );
    // A's key with B's nonce and turn names nothing, and is told nothing.
    let forged = WireId { key: a.key, ..b };
    assert_eq!(
        t.resolve(&forged.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: None
        })
    );
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
            Some(Lost {
                miss: Miss::Turn,
                reason: None
            }),
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
            Some(Lost {
                miss: Miss::Binding,
                reason: None
            }),
            "another key is told nothing"
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
        Some(Lost {
            miss: Miss::Gone,
            reason: Some(Reason::Capacity)
        }),
        "its state is gone and nothing else kept it"
    );
    assert_eq!(
        t.resolve(&id.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Capacity)
        }),
        "and asked again, the slot it left says why"
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
        Some(Lost {
            miss: Miss::Stale,
            reason: None
        }),
        "the ephemeral slot's ids are gone: the session moved, nothing let it go"
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
        Some(Lost {
            miss: Miss::Turn,
            reason: None
        }),
        "the dropped turn's id misses; a fork is the client's, not a reason"
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
        Some(Lost {
            miss: Miss::Turn,
            reason: None
        })
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
fn a_session_unused_past_the_idle_policy_is_reclaimed_and_says_so() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    assert!(
        t.resolve(&id.encode(), None, false, now + Duration::from_secs(60))
            .is_ok(),
        "at the policy's bound, still held"
    );
    let later = now + Duration::from_secs(121);
    assert_eq!(
        t.resolve(&id.encode(), None, false, later).err(),
        Some(Lost {
            miss: Miss::Idle,
            reason: Some(Reason::Idle)
        })
    );
    assert_eq!(
        t.resolve(&id.encode(), None, false, later).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Idle)
        }),
        "asked again, the released slot answers"
    );
    assert_eq!(t.json()["stored"], 0);
    assert_eq!(t.json()["released"]["idle"], 1);
}

/// A full table gives its slot to a new session: the session idle past the
/// policy first, else the least recently used — and each says which.
#[test]
fn a_slot_taken_for_a_new_session_says_idle_or_capacity() {
    let mut t = Table::new(cfg(2, true));
    let now = Instant::now();
    let a = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let b = turn(
        &mut t,
        None,
        Some(true),
        None,
        &[3],
        &scopes(1),
        &[4],
        now + Duration::from_secs(50),
    );
    // At 70 s A is past the 60 s policy and B is not: A's slot goes, as idle.
    let at = now + Duration::from_secs(70);
    let c = turn(&mut t, None, Some(true), None, &[5], &scopes(1), &[6], at);
    assert_eq!(c.key.slot, a.key.slot, "the idle session's slot");
    assert_eq!(
        t.resolve(&a.encode(), None, false, at).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Idle)
        })
    );
    // Nothing idle now: the least recently used, B, goes for capacity.
    let d = turn(&mut t, None, Some(true), None, &[7], &scopes(1), &[8], at);
    assert_eq!(d.key.slot, b.key.slot, "the least recently used slot");
    assert_eq!(
        t.resolve(&b.encode(), None, false, at).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Capacity)
        })
    );
    assert_eq!(t.json()["released"]["idle"], 1);
    assert_eq!(t.json()["released"]["capacity"], 1);
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
    assert_eq!(
        t.resolve(&a.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Memory)
        }),
        "A went, for the byte budget"
    );
    assert_eq!(t.json()["released"]["memory"], 1);
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
        Some(Lost {
            miss: Miss::Turn,
            reason: Some(Reason::Reset)
        })
    );
    assert_eq!(t.json()["released"]["reset"], 1);
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
    assert_eq!(seen["state"], "resident");
    assert!(
        seen.get("messages").is_some_and(Value::is_number),
        "a count, never the messages"
    );
    assert!(
        seen.get("expires_in_s").is_none(),
        "no expiry: when a session goes is the server's policy: {seen}"
    );
    assert_eq!(t.delete(&a.encode(), None, now), Ok(()));
    let gone = Some(Lost {
        miss: Miss::Stale,
        reason: Some(Reason::Delete),
    });
    assert_eq!(t.inspect(&a.encode(), None, now).err(), gone);
    assert_eq!(t.delete(&a.encode(), None, now).err(), gone);
    assert_eq!(t.resolve(&a.encode(), None, false, now).err(), gone);
    assert_eq!(t.json()["released"]["delete"], 1);
}

/// An id from an earlier server process — a generation this one never
/// issued, or a slot it does not have — says `restart`.
#[test]
fn an_id_from_an_earlier_process_says_restart() {
    let now = Instant::now();
    let mut before = Table::with_base(cfg(4, true), 1_000);
    let id = turn(
        &mut before,
        None,
        Some(true),
        None,
        &[1],
        &scopes(1),
        &[2],
        now,
    );
    let mut after = Table::with_base(cfg(4, true), 5_000);
    let restart = Some(Lost {
        miss: Miss::Stale,
        reason: Some(Reason::Restart),
    });
    assert_eq!(after.resolve(&id.encode(), None, false, now).err(), restart);
    let beyond = WireId {
        key: Key { slot: 99, ..id.key },
        ..id
    };
    assert_eq!(
        after.resolve(&beyond.encode(), None, false, now).err(),
        restart
    );
    // Across the wrap of the generation: issued just below u32::MAX, read
    // by a process whose base is past it.
    let mut wrapped = Table::with_base(cfg(1, true), u32::MAX);
    let w = turn(
        &mut wrapped,
        None,
        Some(true),
        None,
        &[1],
        &scopes(1),
        &[2],
        now,
    );
    let _ = turn(
        &mut wrapped,
        None,
        Some(true),
        None,
        &[3],
        &scopes(1),
        &[4],
        now,
    );
    assert_eq!(
        wrapped.resolve(&w.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Capacity)
        }),
        "this process's own, across the wrap"
    );
}

/// A reason is told only to the id's holder: another `Authorization`, or a
/// nonce that is not the turn's, gets none — before and after the release.
#[test]
fn a_reason_is_never_a_hint_to_anyone_but_the_ids_holder() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let key = Some([5u8; 32]);
    let id = turn(&mut t, None, Some(true), key, &[1], &scopes(1), &[2], now);
    let guessed = WireId {
        nonce: id.nonce ^ 1,
        ..id
    };
    assert_eq!(t.delete(&id.encode(), key, now), Ok(()));
    let silent = Some(Lost {
        miss: Miss::Stale,
        reason: None,
    });
    assert_eq!(t.resolve(&id.encode(), None, false, now).err(), silent);
    assert_eq!(
        t.resolve(&id.encode(), Some([6u8; 32]), false, now).err(),
        silent
    );
    assert_eq!(t.resolve(&guessed.encode(), key, false, now).err(), silent);
    assert_eq!(
        t.resolve(&id.encode(), key, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Delete)
        }),
        "the holder is told"
    );
}

/// A slot remembers its last few releases; an id older than them misses
/// without a reason rather than with a wrong one.
#[test]
fn a_release_older_than_the_slots_memory_has_no_reason() {
    let mut t = Table::new(cfg(1, true));
    let now = Instant::now();
    let first = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let mut last = first;
    // One more release than the slot remembers.
    for i in 0..=RELEASES_KEPT as u32 {
        last = turn(
            &mut t,
            None,
            Some(true),
            None,
            &[10 + i],
            &scopes(1),
            &[2],
            now,
        );
    }
    assert_eq!(
        t.resolve(&first.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: None
        })
    );
    let _ = turn(&mut t, None, Some(true), None, &[99], &scopes(1), &[2], now);
    assert_eq!(
        t.resolve(&last.encode(), None, false, now).err(),
        Some(Lost {
            miss: Miss::Stale,
            reason: Some(Reason::Capacity)
        }),
        "a recent one still answers"
    );
}

/// `/health`'s `sessions` carries the retention policy and what it let go;
/// no session carries a lifetime.
#[test]
fn health_reports_the_policy_and_the_releases_by_reason() {
    let mut t = Table::new(Config {
        max_bytes: 3 << 20,
        ..cfg(4, true)
    });
    let now = Instant::now();
    let _ = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let h = t.json();
    assert_eq!(h["held"], 1, "{h}");
    assert_eq!(h["capacity"], 4);
    assert_eq!(h["max_mib"], 3);
    assert_eq!(h["idle_s"], 60);
    assert_eq!(h["reclaim"], "lru by last use");
    assert!(h.get("ttl_s").is_none(), "{h}");
    for r in ["idle", "capacity", "memory", "reset", "delete"] {
        assert_eq!(h["released"][r], 0, "{r}: {h}");
    }
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

/// The end-of-turn id after a reply the model ended is in the session's ids —
/// the next prompt carries it — but the engine never ran it: what a
/// continuation reports as carried is the prompt and the reply.
#[test]
fn the_end_of_turn_id_is_the_sessions_but_not_the_engines() {
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let s1 = scopes(1);
    let r = t.reserve(
        Plan {
            continued: None,
            store: Some(true),
            auth: None,
            prompt_ids: &[1, 2],
            scopes: &s1,
        },
        now,
    );
    assert!(t.commit(
        r,
        Record {
            messages: vec![json!({ "role": "user", "content": 0 })],
            reply_msg: json!({ "role": "assistant", "content": "r" }),
            prompt_ids: vec![1, 2],
            reply: reply(&[3]),
            eot: Some(9),
            scope: Scope::test(1),
            prefix: None,
        },
        now,
    ));
    let x = t.resolve(&r.id.encode(), None, false, now).unwrap();
    assert_eq!(
        x.ids,
        vec![1, 2, 3],
        "the engine held the prompt and the reply"
    );
}
