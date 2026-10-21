//! The sessions a server keeps: the lineages of turns its chat completions'
//! ids name (`design/docs/sessions.md`). A session lives here, in host memory,
//! beside the reply log whose records its turns share; the engine holds
//! sequences, never sessions, and reuses one for a request whose ids extend
//! it.
//!
//! The wire id is `chatcmpl-` + hex(slot, gen, turn, nonce): the slot indexes
//! a fixed table, so a lookup is O(1); the slot's generation moves whenever it
//! is given to another session, so a stale id misses rather than landing in
//! that session; the nonce is per turn and random, so an id is not guessable.

use super::front::{Bound, PrefixRecord, Reply, Scope};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PREFIX: &str = "chatcmpl-";

/// How many sessions a server keeps, and for how long.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// A session not used for this long is forgotten.
    pub ttl: Duration,
    /// Stored sessions.
    pub capacity: usize,
    /// Ephemeral sessions: at most one per engine sequence.
    pub ephemeral: usize,
    /// Host bytes the stored sessions' ids and messages may hold.
    pub max_bytes: usize,
    /// Whether the engine runs one sequence (`--batch 1`), so that which
    /// session it holds is the last one served.
    pub one_sequence: bool,
}

/// A slot and its generation: one session's place in the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub slot: u32,
    pub gen: u32,
}

/// What a completion's `id` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireId {
    pub key: Key,
    pub turn: u32,
    pub nonce: u64,
}

impl WireId {
    pub fn encode(self) -> String {
        format!(
            "{PREFIX}{:08x}{:08x}{:08x}{:016x}",
            self.key.slot, self.key.gen, self.turn, self.nonce
        )
    }

    /// `None` for anything that is not an id this server could have issued.
    pub fn parse(s: &str) -> Option<Self> {
        let hex = s.strip_prefix(PREFIX)?;
        if hex.len() != 40 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let field = |r: std::ops::Range<usize>| u64::from_str_radix(&hex[r], 16).ok();
        Some(Self {
            key: Key {
                slot: field(0..8)? as u32,
                gen: field(8..16)? as u32,
            },
            turn: field(16..24)? as u32,
            nonce: field(24..40)?,
        })
    }

    /// `slot:gen:turn`, for a log line: never the nonce, which is the bearer
    /// part.
    pub fn short(self) -> String {
        format!("{}:{}:{}", self.key.slot, self.key.gen, self.turn)
    }
}

struct Turn {
    nonce: u64,
    /// The messages before this turn's reply.
    msg_len: usize,
    /// The conversation before the reply, as the front end hashes it.
    scope: Scope,
    reply: Arc<Reply>,
    prefix: Option<PrefixRecord>,
    /// The session's ids through this turn's reply.
    ids_end: usize,
}

struct Session {
    stored: bool,
    auth: Option<[u8; 32]>,
    turns: Vec<Turn>,
    /// The latest turn's conversation, its reply message last.
    messages: Vec<Value>,
    /// The latest turn's prompt ids, its reply's accepted ids and the
    /// end-of-turn id where the model stopped on EOS.
    ids: Vec<u32>,
    created: Instant,
    last_used: Instant,
    switches: u64,
    bytes: usize,
}

struct Slot {
    gen: u32,
    session: Option<Session>,
}

/// A continuation the table found.
pub struct Resolved {
    pub key: Key,
    pub turn: usize,
    pub stored: bool,
    /// Whether the engine still holds the session's state; `None` where the
    /// engine runs several sequences and the run says.
    pub resident: Option<bool>,
    /// The session's replies through the continued turn, each bound to the
    /// conversation before it.
    pub bound: Vec<Bound>,
    /// The continued turn's prefix and the ids it indexes.
    pub prefix: Option<(PrefixRecord, Vec<u32>)>,
    /// The conversation through the continued turn, its reply message last
    /// (asked for by the `input` form).
    pub messages: Option<Vec<Value>>,
    /// The session's ids through the continued turn.
    pub ids: Vec<u32>,
    /// Times the session found the engine serving another, before this
    /// request.
    pub switches: u64,
}

/// A turn reserved before its run: the id the response carries.
#[derive(Debug, Clone, Copy)]
pub struct Reservation {
    pub id: WireId,
    pub stored: bool,
}

/// What a request asks of the table before it runs.
pub struct Plan<'a> {
    /// The session and turn it continues.
    pub continued: Option<(Key, usize)>,
    /// `store` as the request said it.
    pub store: Option<bool>,
    pub auth: Option<[u8; 32]>,
    pub prompt_ids: &'a [u32],
    /// The scope before each of the request's messages and after the last.
    pub scopes: &'a [Scope],
}

/// A run's turn, to keep.
pub struct Record {
    /// The conversation the request ran.
    pub messages: Vec<Value>,
    /// The reply as the response returned it.
    pub reply_msg: Value,
    pub prompt_ids: Vec<u32>,
    pub reply: Arc<Reply>,
    /// The end-of-turn id, where the reply stopped on EOS.
    pub eot: Option<u32>,
    pub scope: Scope,
    pub prefix: Option<PrefixRecord>,
}

#[derive(Default)]
struct Counters {
    created: u64,
    continued: u64,
    resident: u64,
    evicted: u64,
    expired: u64,
    switches: u64,
    committed: u64,
}

/// The server's sessions.
pub struct Table {
    cfg: Config,
    /// `[0, capacity)` stored, then the ephemeral ones.
    slots: Vec<Slot>,
    /// The session the engine's one sequence last served.
    occupant: Option<Key>,
    bytes: usize,
    counters: Counters,
}

/// Why a lookup found nothing; every case reads as `expired` on the wire,
/// so a client cannot tell an id that never existed from one that did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Miss {
    /// Not an id this server issues.
    Malformed,
    /// The slot's generation moved, or the slot is empty.
    Stale,
    /// The turn's nonce does not match, or a fork dropped the turn.
    Turn,
    /// The session was bound to another `Authorization`.
    Binding,
    /// Unused past the TTL.
    Ttl,
    /// An ephemeral session whose state the engine no longer holds.
    Gone,
}

impl Miss {
    pub fn name(self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::Stale => "stale",
            Self::Turn => "turn",
            Self::Binding => "binding",
            Self::Ttl => "ttl",
            Self::Gone => "gone",
        }
    }
}

fn same(a: u64, b: u64) -> bool {
    (a ^ b) == 0
}

fn json_bytes(v: &[Value]) -> usize {
    v.iter().map(|m| m.to_string().len()).sum()
}

impl Table {
    pub fn new(cfg: Config) -> Self {
        let n = cfg.capacity + cfg.ephemeral.max(1);
        Self {
            cfg,
            slots: (0..n)
                .map(|_| Slot {
                    gen: 0,
                    session: None,
                })
                .collect(),
            occupant: None,
            bytes: 0,
            counters: Counters::default(),
        }
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    fn stored_slots(&self) -> std::ops::Range<usize> {
        0..self.cfg.capacity
    }

    fn ephemeral_slots(&self) -> std::ops::Range<usize> {
        self.cfg.capacity..self.slots.len()
    }

    fn expired(&self, s: &Session, now: Instant) -> bool {
        now.saturating_duration_since(s.last_used) > self.cfg.ttl
    }

    /// Empty slot `i` and move its generation: every id naming it misses.
    fn free(&mut self, i: usize) {
        let slot = &mut self.slots[i];
        if let Some(s) = slot.session.take() {
            if s.stored {
                self.bytes -= s.bytes;
            }
        }
        slot.gen = slot.gen.wrapping_add(1);
        if self.occupant.is_some_and(|k| k.slot as usize == i) {
            self.occupant = None;
        }
    }

    /// A slot for a new session among `range`: an empty one, else one past
    /// its TTL, else the least recently used.
    fn take_slot(&mut self, range: std::ops::Range<usize>, now: Instant) -> usize {
        let pick = range
            .clone()
            .find(|&i| self.slots[i].session.is_none())
            .or_else(|| {
                range.clone().find(|&i| {
                    self.slots[i]
                        .session
                        .as_ref()
                        .is_some_and(|s| self.expired(s, now))
                })
            })
            .unwrap_or_else(|| {
                range
                    .min_by_key(|&i| self.slots[i].session.as_ref().map(|s| s.last_used))
                    .expect("a non-empty range")
            });
        if self.slots[pick].session.is_some() {
            self.counters.expired += 1;
        }
        self.free(pick);
        pick
    }

    fn get(&self, key: Key) -> Option<&Session> {
        let slot = self.slots.get(key.slot as usize)?;
        if slot.gen != key.gen {
            return None;
        }
        slot.session.as_ref()
    }

    fn get_mut(&mut self, key: Key) -> Option<&mut Session> {
        let slot = self.slots.get_mut(key.slot as usize)?;
        if slot.gen != key.gen {
            return None;
        }
        slot.session.as_mut()
    }

    /// Find the turn `id` names, as `auth` may see it.
    fn find(
        &mut self,
        id: &str,
        auth: Option<[u8; 32]>,
        now: Instant,
    ) -> Result<(Key, usize), Miss> {
        let w = WireId::parse(id).ok_or(Miss::Malformed)?;
        let key = w.key;
        let i = key.slot as usize;
        let ephemeral = self.ephemeral_slots().contains(&i);
        let (expired, gone) = {
            let s = self.get(key).ok_or(Miss::Stale)?;
            if s.auth != auth {
                return Err(Miss::Binding);
            }
            let turn = s.turns.get(w.turn as usize).ok_or(Miss::Turn)?;
            if !same(turn.nonce, w.nonce) {
                return Err(Miss::Turn);
            }
            let gone = ephemeral && self.cfg.one_sequence && self.occupant != Some(key);
            (self.expired(s, now), gone)
        };
        if expired {
            self.counters.expired += 1;
            self.free(i);
            return Err(Miss::Ttl);
        }
        if gone {
            self.free(i);
            return Err(Miss::Gone);
        }
        Ok((key, w.turn as usize))
    }

    /// The turn `id` names, to continue: its bindings and its prefix, and its
    /// conversation where `messages` is asked. Counts a switch where the
    /// engine's sequence has served another session since.
    pub fn resolve(
        &mut self,
        id: &str,
        auth: Option<[u8; 32]>,
        messages: bool,
        now: Instant,
    ) -> Result<Resolved, Miss> {
        let found = self.find(id, auth, now);
        let (key, t) = match found {
            Ok(k) => k,
            Err(m) => {
                self.counters.expired += u64::from(m != Miss::Ttl);
                return Err(m);
            }
        };
        let resident = self.cfg.one_sequence.then(|| self.occupant == Some(key));
        let s = self.get_mut(key).expect("found above");
        s.last_used = now;
        let turn = &s.turns[t];
        let bound = s.turns[..=t]
            .iter()
            .map(|u| Bound {
                scope: u.scope,
                reply: Arc::clone(&u.reply),
            })
            .collect();
        let prefix = turn
            .prefix
            .as_ref()
            .filter(|p| p.ids_len <= turn.ids_end && turn.ids_end <= s.ids.len())
            .map(|p| (p.clone(), s.ids[..p.ids_len].to_vec()));
        let msgs = messages.then(|| s.messages[..=turn.msg_len.min(s.messages.len() - 1)].to_vec());
        Ok(Resolved {
            key,
            turn: t,
            stored: s.stored,
            resident,
            bound,
            prefix,
            messages: msgs,
            ids: s.ids[..turn.ids_end.min(s.ids.len())].to_vec(),
            switches: s.switches,
        })
    }

    /// A continuation of the session at `key` ran: count it, and the switch
    /// where the engine had served another session since its last turn.
    /// Returns the session's switches so far.
    pub fn note_continued(&mut self, key: Key, resident: bool) -> u64 {
        self.counters.continued += 1;
        if resident {
            self.counters.resident += 1;
        } else {
            self.counters.evicted += 1;
            self.counters.switches += 1;
        }
        match self.get_mut(key) {
            Some(s) => {
                s.switches += u64::from(!resident);
                s.switches
            }
            None => 0,
        }
    }

    /// Forget a session's turns (`reset`): the next turn starts its lineage
    /// anew under the same session.
    pub fn reset(&mut self, key: Key) {
        let Some(s) = self.get_mut(key) else {
            return;
        };
        s.turns.clear();
        s.ids.clear();
        s.messages.clear();
        let (stored, old) = (s.stored, s.bytes);
        s.bytes = 0;
        if stored {
            self.bytes -= old;
        }
    }

    /// Reserve the turn a request will leave, before it runs: its session
    /// (the continued one, promoted to stored where the request asks; else a
    /// new one) with every turn the request no longer extends dropped, and
    /// the id the response carries.
    pub fn reserve(&mut self, plan: Plan<'_>, now: Instant) -> Reservation {
        let mut key = plan
            .continued
            .map(|(k, _)| k)
            .filter(|&k| self.get(k).is_some());
        if let Some(k) = key {
            let promote = plan.store == Some(true) && !self.get(k).is_some_and(|s| s.stored);
            if promote {
                let i = self.take_slot(self.stored_slots(), now);
                let mut s = self.slots[k.slot as usize]
                    .session
                    .take()
                    .expect("present above");
                self.free(k.slot as usize);
                s.stored = true;
                s.bytes = s.ids.len() * 4 + json_bytes(&s.messages);
                self.bytes += s.bytes;
                self.slots[i].session = Some(s);
                key = Some(Key {
                    slot: i as u32,
                    gen: self.slots[i].gen,
                });
            }
        }
        let key = match key {
            Some(k) => k,
            None => {
                let stored = plan.store == Some(true);
                let range = if stored {
                    self.stored_slots()
                } else {
                    self.ephemeral_slots()
                };
                let i = self.take_slot(range, now);
                self.slots[i].session = Some(Session {
                    stored,
                    auth: plan.auth,
                    turns: Vec::new(),
                    messages: Vec::new(),
                    ids: Vec::new(),
                    created: now,
                    last_used: now,
                    switches: 0,
                    bytes: 0,
                });
                self.counters.created += 1;
                Key {
                    slot: i as u32,
                    gen: self.slots[i].gen,
                }
            }
        };
        let s = self.get_mut(key).expect("reserved above");
        // A turn stays in the lineage while the request's conversation
        // carries its reply: the messages before it are the ones it answered
        // and a message stands where its reply did. Of its ids, only those
        // the request's prompt starts with stay known; a prefix past them is
        // forgotten (the next use verifies what is left anyway).
        let lcp = s
            .ids
            .iter()
            .zip(plan.prompt_ids)
            .take_while(|(a, b)| a == b)
            .count();
        let n = plan.scopes.len().saturating_sub(1);
        let kept = s
            .turns
            .iter()
            .take_while(|t| t.msg_len < n && plan.scopes.get(t.msg_len) == Some(&t.scope))
            .count();
        s.turns.truncate(kept);
        for t in &mut s.turns {
            t.ids_end = t.ids_end.min(lcp);
            if t.prefix.as_ref().is_some_and(|p| p.ids_len > lcp) {
                t.prefix = None;
            }
        }
        s.last_used = now;
        Reservation {
            id: WireId {
                key,
                turn: kept as u32,
                nonce: rand::random::<u64>() | 1,
            },
            stored: s.stored,
        }
    }

    /// Keep a reserved turn. `false` where the session moved under it (a
    /// concurrent request on the same session, or its slot given away): the
    /// id then names nothing.
    pub fn commit(&mut self, r: Reservation, rec: Record, now: Instant) -> bool {
        let max_bytes = self.cfg.max_bytes;
        let Some(s) = self.get_mut(r.id.key) else {
            return false;
        };
        if s.turns.len() != r.id.turn as usize {
            return false;
        }
        let mut ids = rec.prompt_ids;
        ids.extend_from_slice(&rec.reply.ids);
        ids.extend(rec.eot);
        let mut messages = rec.messages;
        let msg_len = messages.len();
        messages.push(rec.reply_msg);
        s.turns.push(Turn {
            nonce: r.id.nonce,
            msg_len,
            scope: rec.scope,
            reply: rec.reply,
            prefix: rec.prefix,
            ids_end: ids.len(),
        });
        let bytes = ids.len() * 4 + json_bytes(&messages);
        s.ids = ids;
        s.messages = messages;
        s.last_used = now;
        let (stored, old) = (s.stored, s.bytes);
        s.bytes = bytes;
        if stored {
            self.bytes = self.bytes - old + bytes;
        }
        self.counters.committed += 1;
        if self.cfg.one_sequence {
            self.occupant = Some(r.id.key);
        }
        // Over the byte budget, the least recently used other stored
        // sessions go first.
        while self.bytes > max_bytes {
            let victim = self
                .stored_slots()
                .filter(|&i| i != r.id.key.slot as usize)
                .filter_map(|i| self.slots[i].session.as_ref().map(|s| (i, s.last_used)))
                .min_by_key(|&(_, t)| t)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    self.counters.expired += 1;
                    self.free(i);
                }
                None => break,
            }
        }
        true
    }

    /// The engine's sequence served something no session holds (another
    /// route, a failed run): no session is resident any more.
    pub fn note_other(&mut self) {
        self.occupant = None;
    }

    /// `GET /v1l/sessions/{id}`: what the server holds of it — never its
    /// content.
    pub fn inspect(&mut self, id: &str, auth: Option<[u8; 32]>, now: Instant) -> Option<Value> {
        let (key, t) = self.find(id, auth, now).ok()?;
        let resident = self.cfg.one_sequence.then(|| self.occupant == Some(key));
        let ttl = self.cfg.ttl;
        let s = self.get(key)?;
        Some(json!({
            "id": id,
            "turn": t,
            "turns": s.turns.len(),
            "stored": s.stored,
            "resident": resident,
            "ids": s.ids.len(),
            "messages": s.messages.len(),
            "bytes": s.bytes.max(s.ids.len() * 4),
            "switches": s.switches,
            "age_s": now.saturating_duration_since(s.created).as_secs(),
            "idle_s": now.saturating_duration_since(s.last_used).as_secs(),
            "expires_in_s": ttl.saturating_sub(now.saturating_duration_since(s.last_used)).as_secs(),
        }))
    }

    /// `DELETE /v1l/sessions/{id}`: forget the session now.
    pub fn delete(&mut self, id: &str, auth: Option<[u8; 32]>, now: Instant) -> bool {
        match self.find(id, auth, now) {
            Ok((key, _)) => {
                self.counters.expired += 1;
                self.free(key.slot as usize);
                true
            }
            Err(_) => false,
        }
    }

    /// `/health`'s `sessions`.
    pub fn json(&self) -> Value {
        let held =
            |r: std::ops::Range<usize>| r.filter(|&i| self.slots[i].session.is_some()).count();
        json!({
            "stored": held(self.stored_slots()),
            "ephemeral": held(self.ephemeral_slots()),
            "capacity": self.cfg.capacity,
            "ttl_s": self.cfg.ttl.as_secs(),
            "bytes": self.bytes,
            "max_bytes": self.cfg.max_bytes,
            "created": self.counters.created,
            "continued": self.counters.continued,
            "resident": self.counters.resident,
            "evicted": self.counters.evicted,
            "expired": self.counters.expired,
            "switches": self.counters.switches,
            "turns": self.counters.committed,
            "occupant": self.occupant.map(|k| format!("{}:{}", k.slot, k.gen)),
        })
    }
}

#[cfg(test)]
#[path = "sessions_tests.rs"]
mod tests;
