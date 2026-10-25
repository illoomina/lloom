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
        keep: Keep::Prefix,
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
            keep: None,
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
    let mut t = Table::new(cfg(4, true));
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
    // Admit either complete session, but not both; logical payload bytes alone
    // no longer describe this fixture's retained allocation requirement.
    let one = t.bytes;
    t.cfg.max_bytes = one + one / 2;
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
        keep: None,
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
            keep: None,
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

#[test]
fn an_oversize_stored_session_declines_retention() {
    let mut t = Table::new(Config {
        max_bytes: 1024,
        ..cfg(1, true)
    });
    let now = Instant::now();
    let prompt = vec![1; 4096];
    let ss = scopes(1);
    let r = t.reserve(
        Plan {
            continued: None,
            store: Some(true),
            keep: None,
            auth: Some([7; 32]),
            prompt_ids: &prompt,
            scopes: &ss,
        },
        now,
    );
    let kept = t.commit(
        r,
        Record {
            messages: vec![json!({"role":"user","content":"large"})],
            reply_msg: json!({"role":"assistant","content":"r"}),
            prompt_ids: prompt,
            reply: reply(&[2]),
            eot: None,
            scope: ss[1],
            prefix: None,
        },
        now,
    );
    assert!(
        !kept,
        "successful generation must not retain an oversize lineage"
    );
    assert!(
        t.bytes <= t.cfg.max_bytes,
        "{} > {}",
        t.bytes,
        t.cfg.max_bytes
    );
    assert!(t
        .resolve(&r.id.encode(), Some([7; 32]), false, now)
        .is_err());
}

mod allocation_receipt {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
    pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    pub static LIVE: AtomicIsize = AtomicIsize::new(0);
    pub static BLOCKS: AtomicIsize = AtomicIsize::new(0);
    pub struct Allocator;
    // SAFETY: every allocation/deallocation is forwarded unchanged to System;
    // the counter observes requested live allocation bytes only.
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                BLOCKS.fetch_add(1, Ordering::Relaxed);
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
            BLOCKS.fetch_sub(1, Ordering::Relaxed);
            unsafe { System.dealloc(ptr, layout) };
        }
    }
    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
}

#[test]
fn repeated_small_turns_reset_delete_and_eviction_account_retained_allocations() {
    use std::sync::atomic::Ordering;
    let before_control = allocation_receipt::LIVE.load(Ordering::Relaxed);
    let allocation = Box::new([0u8; 4096]);
    std::hint::black_box(&allocation);
    assert_eq!(
        allocation_receipt::LIVE.load(Ordering::Relaxed) - before_control,
        4096,
        "the real allocator instrument must observe a live allocation"
    );
    drop(allocation);
    assert_eq!(
        allocation_receipt::LIVE.load(Ordering::Relaxed),
        before_control,
        "the instrument must observe its deallocation"
    );
    let mut t = Table::new(cfg(1, true));
    let now = Instant::now();
    eprintln!("retention allocator control: System requested live bytes; excludes allocator caches/rounding and fixed empty table");
    let baseline = allocation_receipt::LIVE.load(Ordering::Relaxed);
    let mut peak_undercharge = 0isize;
    for cycle in 0..12 {
        let mut previous = None;
        for n in 0..32u8 {
            let id = turn(
                &mut t,
                previous,
                Some(true),
                Some([9; 32]),
                &[1],
                &scopes(n.saturating_mul(2).saturating_add(1)),
                &[2],
                now,
            );
            previous = Some((id.key, id.turn as usize));
        }
        let key = previous.unwrap().0;
        if cycle % 2 == 0 {
            t.reset(key);
        } else {
            let id = t.slots[key.slot as usize].session.as_ref().unwrap();
            let last = id.turns.last().unwrap();
            let wire = WireId {
                key,
                turn: (id.turns.len() - 1) as u32,
                nonce: last.nonce,
            };
            t.delete(&wire.encode(), Some([9; 32]), now).unwrap();
        }
        let live = allocation_receipt::LIVE.load(Ordering::Relaxed) - baseline;
        peak_undercharge = peak_undercharge.max(live - t.bytes as isize);
        eprintln!(
            "retention cycle={cycle} allocator_live_requested={live} reported={} undercharge={}",
            t.bytes,
            live - t.bytes as isize
        );
    }
    assert!(peak_undercharge <= 0,
        "retained allocations exceed reported bytes by {peak_undercharge}; fixed empty table excluded");
}

fn assert_charge(t: &Table) {
    let slow = t
        .slots
        .iter()
        .map(|slot| {
            let session = slot.session.as_ref().map_or(0, |s| {
                s.ids.capacity() * size_of::<u32>()
                    + s.messages.capacity() * size_of::<Box<str>>()
                    + s.messages.iter().map(|v| v.as_ref().len()).sum::<usize>()
                    + s.turns.capacity() * size_of::<Turn>()
                    + s.turns
                        .iter()
                        .map(|turn| {
                            reply_bytes(&turn.reply)
                                + turn.prefix.as_ref().map_or(0, PrefixRecord::retained_bytes)
                        })
                        .sum::<usize>()
                    + s.dropped.capacity() * size_of::<Dropped>()
            });
            session
                + slot.released.capacity() * size_of::<Released>()
                + slot
                    .released
                    .iter()
                    .map(|r| {
                        r.nonces.capacity() * size_of::<u64>()
                            + r.dropped.capacity() * size_of::<Dropped>()
                    })
                    .sum::<usize>()
        })
        .sum::<usize>();
    assert_eq!(
        t.bytes, slow,
        "cached accounting equals retained allocation oracle"
    );
    assert!(
        t.bytes <= t.cfg.max_bytes,
        "{} > {}",
        t.bytes,
        t.cfg.max_bytes
    );
}

#[test]
fn release_histories_share_the_byte_bound_and_authentication() {
    let mut t = Table::new(Config {
        max_bytes: 32 * 1024,
        ..cfg(16, true)
    });
    let now = Instant::now();
    let auth = Some([17; 32]);
    let mut old = None;
    for cycle in 0..160 {
        let mut previous = None;
        for n in 0..16u8 {
            let id = turn(
                &mut t,
                previous,
                Some(true),
                auth,
                &[1],
                &scopes(2 * n + 1),
                &[2],
                now,
            );
            previous = Some((id.key, id.turn as usize));
            assert_charge(&t);
            if cycle == 0 && n == 0 {
                old = Some(id);
            }
        }
    }
    let old = old.unwrap();
    for auth in [None, Some([18; 32])] {
        assert_eq!(
            t.resolve(&old.encode(), auth, false, now)
                .err()
                .unwrap()
                .reason,
            None
        );
    }
    assert_charge(&t);
    assert!(
        t.slots.iter().any(|s| s.released.len() < RELEASES_KEPT),
        "byte pressure must limit the release history before the count-only ceiling"
    );
}

#[test]
fn unused_id_capacity_is_charged_and_an_oversize_publish_preserves_other_sessions() {
    let mut t = Table::new(Config {
        max_bytes: 32 * 1024,
        ..cfg(4, true)
    });
    let now = Instant::now();
    let a = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let mut ids = Vec::with_capacity(128 * 1024);
    ids.push(1);
    let ss = scopes(1);
    let r = t.reserve(
        Plan {
            continued: None,
            store: Some(true),
            keep: None,
            auth: Some([3; 32]),
            prompt_ids: &ids,
            scopes: &ss,
        },
        now,
    );
    let prepared = PreparedRecord::new(Record {
        messages: vec![json!({"role":"user","content":"x"})],
        reply_msg: json!({"role":"assistant","content":"r"}),
        prompt_ids: ids,
        reply: reply(&[2]),
        eot: None,
        scope: ss[1],
        prefix: None,
    })
    .unwrap();
    let result = t.publish(r, prepared, now);
    assert!(!result.retained);
    assert_eq!(result.reason, Some(Reason::Budget));
    drop(result);
    assert!(
        t.resolve(&a.encode(), None, false, now).is_ok(),
        "unrelated lineage survives impossible admission"
    );
    assert_eq!(
        t.resolve(&r.id.encode(), Some([3; 32]), false, now)
            .err()
            .unwrap()
            .reason,
        Some(Reason::Budget)
    );
    assert_eq!(
        t.resolve(&r.id.encode(), None, false, now)
            .err()
            .unwrap()
            .reason,
        None
    );
    let guessed = WireId {
        nonce: r.id.nonce ^ 1,
        ..r.id
    };
    assert_eq!(
        t.resolve(&guessed.encode(), Some([3; 32]), false, now)
            .err()
            .unwrap()
            .reason,
        None
    );
    assert_charge(&t);
}

#[test]
fn prepared_messages_roundtrip_without_a_second_value_log() {
    let messages = vec![json!({"role":"user","content":"\\ \" \n π","nested":{"a":[1,true,null]}})];
    let mut messages = messages;
    messages.push(
        json!({"$serde_json::private::RawValue": "{\"must_stay_a_string\":true}",
        "other": "ordinary key"}),
    );
    let expected = messages.clone();
    let p = PreparedRecord::new(Record {
        messages,
        reply_msg: json!({"role":"assistant","content":"r"}),
        prompt_ids: vec![1],
        reply: reply(&[2]),
        eot: None,
        scope: Scope::test(1),
        prefix: None,
    })
    .unwrap();
    let restored: Vec<Value> = p
        .messages
        .iter()
        .map(|v| serde_json::from_str(v.as_ref()).unwrap())
        .collect();
    assert_eq!(&restored[..expected.len()], &expected);
    assert_eq!(
        p.payload_bytes,
        p.ids.capacity() * size_of::<u32>()
            + p.messages.capacity() * size_of::<Box<str>>()
            + p.messages.iter().map(|v| v.as_ref().len()).sum::<usize>()
    );
}

#[test]
fn prepared_publication_with_spare_turn_capacity_allocates_nothing() {
    use std::sync::atomic::Ordering;
    let mut t = Table::new(cfg(4, true));
    let now = Instant::now();
    let id = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let ss = scopes(3);
    let reservation = t.reserve(
        Plan {
            continued: Some((id.key, 0)),
            store: None,
            keep: None,
            auth: None,
            prompt_ids: &[1, 2, 3],
            scopes: &ss,
        },
        now,
    );
    assert!(t.get(id.key).unwrap().turns.capacity() > 1);
    let record = PreparedRecord::new(Record {
        messages: vec![
            json!({"role":"user","content":"x".repeat(64*1024)}),
            json!({"role":"assistant","content":"r"}),
            json!({"role":"user","content":"next"}),
        ],
        reply_msg: json!({"role":"assistant","content":"r"}),
        prompt_ids: vec![1, 2, 3],
        reply: reply(&[4]),
        eot: None,
        scope: ss[3],
        prefix: None,
    })
    .unwrap();
    let before = allocation_receipt::ALLOCATIONS.load(Ordering::Relaxed);
    let published = t.publish(reservation, record, now);
    let allocated = allocation_receipt::ALLOCATIONS.load(Ordering::Relaxed) - before;
    eprintln!(
        "retention publication allocation_calls={allocated} retained={}",
        published.retained
    );
    assert!(published.retained);
    assert_eq!(
        allocated, 0,
        "normal publication consumes prepared buffers without serializing history"
    );
    drop(published);
    assert_charge(&t);
}

#[test]
fn reply_storage_capacities_are_not_hidden_behind_shared_arcs() {
    let now = Instant::now();
    for field in [
        "ids",
        "text",
        "reasoning",
        "content",
        "logprobs",
        "rows",
        "row",
    ] {
        let mut t = Table::new(Config {
            max_bytes: 32 * 1024,
            ..cfg(1, true)
        });
        let ss = scopes(1);
        let r = t.reserve(
            Plan {
                continued: None,
                store: Some(true),
                keep: None,
                auth: None,
                prompt_ids: &[1],
                scopes: &ss,
            },
            now,
        );
        let mut value = (*reply(&[2])).clone();
        let text = || {
            let mut s = String::with_capacity(64 * 1024);
            s.push('r');
            s
        };
        match field {
            "ids" => {
                value.ids = Vec::with_capacity(16 * 1024);
                value.ids.push(2);
            }
            "text" => value.text = text(),
            "reasoning" => value.reasoning_text = text(),
            "content" => value.content_text = text(),
            "logprobs" => value.logprobs = Some(Vec::with_capacity(16 * 1024)),
            "rows" => value.rows = Some(Vec::with_capacity(4096)),
            "row" => value.rows = Some(vec![Vec::with_capacity(16 * 1024)]),
            _ => unreachable!(),
        }
        let shared = Arc::new(value);
        let record = PreparedRecord::new(Record {
            messages: vec![json!({"role":"user","content":"q"})],
            reply_msg: json!({"role":"assistant","content":"r"}),
            prompt_ids: vec![1],
            reply: Arc::clone(&shared),
            eot: None,
            scope: ss[1],
            prefix: None,
        })
        .unwrap();
        let publication = t.publish(r, record, now);
        assert!(
            !publication.retained,
            "{field} capacity must be charged even while shared"
        );
        assert_eq!(publication.reason, Some(Reason::Budget), "{field}");
        drop(publication);
        assert_charge(&t);
    }
}

#[test]
fn stored_message_numbers_and_nested_tool_values_are_exact() {
    let mut floats = vec![
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        51.248178375505404,
        -93.31137037688033,
        -36.573994842753436,
        52.314008204106244,
        97.45365320034685,
        2.0030397744267762e-253,
        7.101215824554616e260,
        1.769268377902049e74,
        -1.6727517818542075e58,
        3.9287532173373315e299,
    ];
    let mut bits = 0x4837_12fa_93be_714du64;
    for _ in 0..4096 {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        let value = f64::from_bits(bits);
        if value.is_finite() {
            floats.push(value);
        }
    }
    let mut float_mismatches = 0;
    let mut integer_mismatches = 0;
    let mut examples = Vec::new();
    let cases = floats.len();
    for value in floats {
        let message = json!({"role":"assistant","tool_calls":[{"id":"call-π",
            "type":"function","function":{"name":"numeric-tool","arguments":{
                "value":value,"nested":[{"large":u64::MAX,"low":i64::MIN}],
                "text":"escaped \\ \" \n π"}}}]});
        let prepared = PreparedRecord::new(Record {
            messages: vec![message.clone()],
            reply_msg: json!({"role":"assistant","content":"r"}),
            prompt_ids: vec![1],
            reply: reply(&[2]),
            eot: None,
            scope: Scope::test(1),
            prefix: None,
        })
        .unwrap();
        let restored = serde_json::from_str::<Value>(prepared.messages[0].as_ref());
        let same = restored.as_ref().ok().is_some_and(|restored| {
            restored["tool_calls"][0]["function"]["arguments"]["value"]
                .as_f64()
                .is_some_and(|actual| actual.to_bits() == value.to_bits())
                && serde_json::to_string(restored).unwrap()
                    == serde_json::to_string(&message).unwrap()
        });
        if !same {
            float_mismatches += 1;
            if examples.len() < 8 {
                examples.push(format!(
                    "original={value:?} bits={:016x} restored={restored:?}",
                    value.to_bits()
                ));
            }
        }
    }
    for value in [0, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
        let message = json!({"role":"tool","content":{"values":[value,{"again":value}]}});
        let prepared = PreparedRecord::new(Record {
            messages: vec![message.clone()],
            reply_msg: Value::Null,
            prompt_ids: vec![1],
            reply: reply(&[2]),
            eot: None,
            scope: Scope::test(1),
            prefix: None,
        })
        .unwrap();
        let restored: Value = serde_json::from_str(prepared.messages[0].as_ref()).unwrap();
        if restored["content"]["values"][0].as_u64() != Some(value) || restored != message {
            integer_mismatches += 1;
            if examples.len() < 8 {
                examples.push(format!("integer={value} restored={restored}"));
            }
        }
    }
    eprintln!("retention numeric fidelity float_cases={cases} float_mismatches={float_mismatches} integer_cases=4 integer_mismatches={integer_mismatches} examples={examples:?}");
    assert_eq!(
        float_mismatches + integer_mismatches,
        0,
        "stored JSON must preserve numeric bits, negative zero and nested prompt/template values"
    );
}

#[test]
fn every_mutation_boundary_keeps_the_retention_charge_and_cap() {
    let mut t = Table::new(Config {
        max_bytes: 16 * 1024,
        ..cfg(2, true)
    });
    let now = Instant::now();
    assert_charge(&t);
    let a = turn(
        &mut t,
        None,
        None,
        Some([2; 32]),
        &[1],
        &scopes(1),
        &[2],
        now,
    );
    assert_charge(&t);
    let found = t.resolve(&a.encode(), Some([2; 32]), true, now).unwrap();
    assert_charge(&t);
    let ss = scopes(3);
    let r = t.reserve(
        Plan {
            continued: Some((found.key, found.turn)),
            store: Some(true),
            keep: None,
            auth: Some([2; 32]),
            prompt_ids: &[1, 2, 3],
            scopes: &ss,
        },
        now,
    );
    assert_charge(&t);
    let p = PreparedRecord::new(Record {
        messages: vec![json!({"role":"user","content":"q"})],
        reply_msg: json!({"role":"assistant","content":"r"}),
        prompt_ids: vec![1, 2, 3],
        reply: reply(&[4]),
        eot: None,
        scope: ss[3],
        prefix: None,
    })
    .unwrap();
    let publication = t.publish(r, p, now);
    assert!(publication.retained);
    drop(publication);
    assert_charge(&t);
    assert!(t.inspect(&r.id.encode(), Some([2; 32]), now).is_ok());
    assert_charge(&t);
    t.reset(r.id.key);
    assert_charge(&t);
    assert!(t.resolve(&r.id.encode(), Some([2; 32]), true, now).is_err());
    assert_charge(&t);
    let fork_scopes = [Scope::test(90), Scope::test(91)];
    let next = t.reserve(
        Plan {
            continued: Some((r.id.key, 0)),
            store: None,
            keep: None,
            auth: Some([2; 32]),
            prompt_ids: &[8],
            scopes: &fork_scopes,
        },
        now,
    );
    assert_charge(&t);
    assert!(t.commit(
        next,
        Record {
            messages: vec![json!({"role":"user","content":"fork"})],
            reply_msg: json!({"role":"assistant","content":"r"}),
            prompt_ids: vec![8],
            reply: reply(&[9]),
            eot: None,
            scope: fork_scopes[1],
            prefix: None
        },
        now
    ));
    assert_charge(&t);
    assert!(t.delete(&next.id.encode(), Some([2; 32]), now).is_ok());
    assert_charge(&t);
    let idle = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    assert_charge(&t);
    assert!(t
        .resolve(&idle.encode(), None, false, now + Duration::from_secs(61))
        .is_err());
    assert_charge(&t);
    let ephemeral = turn(&mut t, None, None, None, &[1], &scopes(1), &[2], now);
    assert_charge(&t);
    t.note_other();
    assert_charge(&t);
    assert!(t.resolve(&ephemeral.encode(), None, false, now).is_err());
    assert_charge(&t);
    let retained = turn(&mut t, None, Some(true), None, &[1], &scopes(1), &[2], now);
    let removed = t.free(retained.key.slot as usize, Some(Reason::Delete));
    assert_charge(&t);
    drop(removed);
    assert_charge(&t);
}

struct MaskedJson<'a>(&'a Value);

impl serde::Serialize for MaskedJson<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.0 {
            Value::Number(number) if number.is_f64() => serializer.serialize_f64(0.0),
            Value::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(&MaskedJson(value))?;
                }
                seq.end()
            }
            Value::Object(values) => {
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for (key, value) in values {
                    map.serialize_entry(key, &MaskedJson(value))?;
                }
                map.end()
            }
            value => serde::Serialize::serialize(value, serializer),
        }
    }
}

fn masked_storage_candidate(value: &Value) -> (Box<str>, Box<[u64]>) {
    fn floats(value: &Value, bits: &mut Vec<u64>) {
        match value {
            Value::Number(number) if number.is_f64() => {
                bits.push(number.as_f64().unwrap().to_bits())
            }
            Value::Array(values) => {
                for value in values {
                    floats(value, bits);
                }
            }
            Value::Object(values) => {
                for value in values.values() {
                    floats(value, bits);
                }
            }
            _ => {}
        }
    }
    let mut bits = Vec::new();
    floats(value, &mut bits);
    (
        serde_json::to_string(&MaskedJson(value))
            .unwrap()
            .into_boxed_str(),
        bits.into_boxed_slice(),
    )
}

#[test]
fn storage_representation_requested_footprint() {
    use super::storage::StoredMessage;
    use std::sync::atomic::Ordering;
    fn measure<T>(build: impl FnOnce() -> T) -> (isize, isize) {
        let before = allocation_receipt::LIVE.load(Ordering::Relaxed);
        let blocks = allocation_receipt::BLOCKS.load(Ordering::Relaxed);
        let owned = Box::new(build());
        std::hint::black_box(&owned);
        let result = (
            allocation_receipt::LIVE.load(Ordering::Relaxed) - before,
            allocation_receipt::BLOCKS.load(Ordering::Relaxed) - blocks,
        );
        drop(owned);
        assert_eq!(allocation_receipt::LIVE.load(Ordering::Relaxed), before);
        assert_eq!(allocation_receipt::BLOCKS.load(Ordering::Relaxed), blocks);
        result
    }
    eprintln!("STORAGE-COST System retained allocation requests and blocks; root boxed uniformly; inputs excluded; N=3 per case; no throughput/RSS claim");
    let cases = [
        (
            "short-chat",
            json!({"role":"user","content":"What is the capital of France?"}),
        ),
        (
            "long-chat",
            json!({"role":"user","content":"x".repeat(8192)}),
        ),
        (
            "nested-tool",
            json!({"role":"assistant","content":null,"tool_calls":[{"id":"call-π",
            "type":"function","function":{"name":"numeric-tool","arguments":{"value":51.248178375505404,
                "negative_zero":-0.0,"nested":[{"large":u64::MAX,"low":i64::MIN}],"text":"escaped \\ \" \n π"}}}]}),
        ),
        (
            "numeric-array",
            json!({"role":"tool","content":vec![0.1f64;1024]}),
        ),
        (
            "small-history",
            Value::Array(
                (0..32)
                    .map(|n| json!({"role":"user","content":format!("turn-{n}")}))
                    .collect(),
            ),
        ),
    ];
    for (name, value) in cases {
        let mut previous = None;
        for run in 0..3 {
            let baseline = measure(|| value.clone());
            let compact = measure(|| StoredMessage::new(value.clone()));
            let masked = measure(|| masked_storage_candidate(&value));
            let check = StoredMessage::new(value.clone());
            assert_eq!(
                compact.0 as usize,
                size_of::<StoredMessage>() + check.heap_bytes()
            );
            assert_eq!(
                serde_json::to_string(&check.value()).unwrap(),
                serde_json::to_string(&value).unwrap()
            );
            if let Some(prior) = previous {
                assert_eq!(prior, (baseline, compact, masked));
            }
            previous = Some((baseline, compact, masked));
            eprintln!("STORAGE-COST case={name} run={run} value_bytes={} value_blocks={} compact_bytes={} compact_blocks={} masked_bytes={} masked_blocks={}",
                baseline.0, baseline.1, compact.0, compact.1, masked.0, masked.1);
        }
    }
}
