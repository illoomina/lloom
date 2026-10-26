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
//!
//! A session has no lifetime of its own: it is held until the server
//! reclaims it — unused past the idle policy, the least recently used past
//! the capacity or the byte budget — or a client ends it, and an id that
//! misses says which (`Reason`).

use super::front::{Bound, PrefixRecord, Reply, Scope};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PREFIX: &str = "chatcmpl-";
pub const ACCOUNTING: &str = "retained allocation requests; stored + ephemeral + release history; shared replies charged per owner; excludes fixed table, allocator overhead and transient copies";

/// What a session keeps of the engine's state when the engine's state must
/// give way (`docs/sessions.md` §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// The session's ids and messages.
    None,
    /// And its head's state, pinned in the engine.
    Prefix,
}

impl Keep {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Prefix => "prefix",
        }
    }
}

/// Releases a slot remembers, so an id of a session let go can still be told
/// why; an older one misses without a reason.
const RELEASES_KEPT: usize = 8;

/// Turns a session remembers dropping by `reset`, for the same purpose.
const DROPPED_KEPT: usize = 64;

/// The server's retention policy for its sessions.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// A session unused this long is reclaimed (`server.sessions.idle_s`).
    pub idle: Duration,
    /// Stored sessions.
    pub capacity: usize,
    /// Ephemeral sessions: at most one per engine sequence.
    pub ephemeral: usize,
    /// Dynamic allocation requests retained by sessions and release records.
    /// Fixed table slots and transient request/response copies are separate.
    pub max_bytes: usize,
    /// Whether the engine runs one sequence (`--batch 1`), so that which
    /// session it holds is the last one served.
    pub one_sequence: bool,
    /// What a session keeps when no request of it says (`session.keep`).
    pub keep: Keep,
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

/// Why the server let a session, or a turn of it, go
/// (`docs/sessions.md` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Unused past the idle policy.
    Idle,
    /// Its slot went to another session: the least recently used past the
    /// table's capacity, or an ephemeral session whose state the engine gave
    /// to another conversation.
    Capacity,
    /// The least recently used past the byte budget.
    Memory,
    /// `reset` dropped the turn.
    Reset,
    /// `DELETE /v1l/sessions/{id}`.
    Delete,
    /// Issued by an earlier server process.
    Restart,
    /// The complete lineage cannot fit the retention budget.
    Budget,
}

impl Reason {
    pub const ALL: [Reason; 7] = [
        Reason::Idle,
        Reason::Capacity,
        Reason::Memory,
        Reason::Reset,
        Reason::Delete,
        Reason::Restart,
        Reason::Budget,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Capacity => "capacity",
            Self::Memory => "memory",
            Self::Reset => "reset",
            Self::Delete => "delete",
            Self::Restart => "restart",
            Self::Budget => "budget",
        }
    }
}

/// A session this slot let go, kept to answer for its ids: the reason, and
/// what verifies an id against it (the binding, each turn's nonce).
struct Released {
    gen: u32,
    reason: Reason,
    auth: Option<[u8; 32]>,
    nonces: Vec<u64>,
    dropped: VecDeque<Dropped>,
    bytes: usize,
}

/// A turn `reset` dropped from a session that lives on.
#[derive(Clone, Copy)]
struct Dropped {
    turn: u32,
    nonce: u64,
    reason: Reason,
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
    /// The last of them is the end-of-turn id the template writes for the
    /// position the model's EOS took: the engine never ran it.
    eot: bool,
    bytes: usize,
}

struct Session {
    stored: bool,
    auth: Option<[u8; 32]>,
    /// The latest `keep` a request of the session asked.
    keep: Option<Keep>,
    turns: Vec<Turn>,
    /// The latest turn's conversation, its reply message last.
    messages: Vec<StoredMessage>,
    /// The latest turn's prompt ids, its reply's accepted ids and the
    /// end-of-turn id where the model stopped on EOS.
    ids: Vec<u32>,
    created: Instant,
    last_used: Instant,
    switches: u64,
    bytes: usize,
    payload_bytes: usize,
    turn_bytes: usize,
    dropped: VecDeque<Dropped>,
}

struct Slot {
    gen: u32,
    session: Option<Session>,
    released: VecDeque<Released>,
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
    pub messages: Option<Vec<StoredMessage>>,
    /// The session's ids the engine held after the continued turn: its
    /// prompt and its reply, not the end-of-turn id after them.
    pub ids: Vec<u32>,
    /// Times the session found the engine serving another, before this
    /// request.
    pub switches: u64,
    /// The latest `keep` a request of the session asked.
    pub keep: Option<Keep>,
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
    /// `keep` as the request said it: the session's from now on.
    pub keep: Option<Keep>,
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
    /// Sessions let go, and `reset`s, by reason (`Reason::ALL`'s order).
    released: [u64; 7],
}

/// The server's sessions.
pub struct Table {
    cfg: Config,
    /// `[0, capacity)` stored, then the ephemeral ones.
    slots: Vec<Slot>,
    /// Every slot's first generation, drawn per process: an id whose
    /// generation this process has not reached was issued by an earlier one.
    base: u32,
    /// The session the engine's one sequence last served.
    occupant: Option<Key>,
    bytes: usize,
    counters: Counters,
}

/// Why a lookup found nothing, for the server's own log. On the wire every
/// case is `expired`, and only `reason` — present where the id's holder can
/// be told why — says more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Miss {
    /// Not an id this server issues.
    Malformed,
    /// The slot's generation moved, or the slot is empty.
    Stale,
    /// The turn's nonce does not match, or a fork or a reset dropped the turn.
    Turn,
    /// The session was bound to another `Authorization`.
    Binding,
    /// Unused past the idle policy.
    Idle,
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
            Self::Idle => "idle",
            Self::Gone => "gone",
        }
    }
}

/// A lookup that found nothing, and why the server let the session go where
/// the id verifies against what it let go: its binding and its turn's nonce.
/// An id that does not — another `Authorization`, a guessed nonce, a release
/// too old to remember — gets no reason, so a reason is never a hint that an
/// id exists to anyone but its holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lost {
    pub miss: Miss,
    pub reason: Option<Reason>,
}

impl Lost {
    fn of(miss: Miss, reason: Option<Reason>) -> Self {
        Self { miss, reason }
    }
}

fn same(a: u64, b: u64) -> bool {
    (a ^ b) == 0
}

/// Incoming variable-sized work is prepared before taking the table lock.
pub struct PreparedRecord {
    ids: Vec<u32>,
    messages: Vec<StoredMessage>,
    turn: Turn,
    payload_bytes: usize,
}

impl PreparedRecord {
    pub fn new(rec: Record) -> serde_json::Result<Self> {
        let mut ids = rec.prompt_ids;
        ids.extend_from_slice(&rec.reply.ids);
        ids.extend(rec.eot);
        let msg_len = rec.messages.len();
        let messages = rec
            .messages
            .into_iter()
            .chain([rec.reply_msg])
            .map(StoredMessage::new)
            .collect::<Vec<_>>();
        let payload_bytes = ids.capacity() * size_of::<u32>()
            + messages.capacity() * size_of::<StoredMessage>()
            + messages
                .iter()
                .map(StoredMessage::heap_bytes)
                .sum::<usize>();
        let bytes =
            reply_bytes(&rec.reply) + rec.prefix.as_ref().map_or(0, PrefixRecord::retained_bytes);
        Ok(Self {
            turn: Turn {
                nonce: 0,
                msg_len,
                scope: rec.scope,
                reply: rec.reply,
                prefix: rec.prefix,
                ids_end: ids.len(),
                eot: rec.eot.is_some(),
                bytes,
            },
            ids,
            messages,
            payload_bytes,
        })
    }
}

fn reply_bytes(r: &Reply) -> usize {
    size_of::<Reply>()
        + 2 * size_of::<usize>()
        + r.ids.capacity() * size_of::<u32>()
        + r.text.capacity()
        + r.reasoning_text.capacity()
        + r.content_text.capacity()
        + r.logprobs
            .as_ref()
            .map_or(0, |v| v.capacity() * size_of::<f32>())
        + r.rows.as_ref().map_or(0, |v| {
            v.capacity() * size_of::<Vec<f32>>()
                + v.iter()
                    .map(|row| row.capacity() * size_of::<f32>())
                    .sum::<usize>()
        })
}

impl Session {
    fn account(&mut self) {
        self.bytes = self.payload_bytes
            + self.turn_bytes
            + self.turns.capacity() * size_of::<Turn>()
            + self.dropped.capacity() * size_of::<Dropped>();
    }
}

/// Dropping this result outside the table lock reclaims replaced histories.
#[must_use]
pub struct Publication {
    pub retained: bool,
    pub reason: Option<Reason>,
    _retired: Vec<Session>,
    _replaced: Option<(Vec<u32>, Vec<StoredMessage>)>,
    _unpublished: Option<PreparedRecord>,
}

fn push_dropped(q: &mut VecDeque<Dropped>, d: Dropped) {
    if q.len() == DROPPED_KEPT {
        q.pop_front();
    }
    q.push_back(d);
}

fn dropped_reason(q: &VecDeque<Dropped>, w: WireId) -> Option<Reason> {
    q.iter()
        .find(|d| d.turn == w.turn && same(d.nonce, w.nonce))
        .map(|d| d.reason)
}

impl Table {
    pub fn new(cfg: Config) -> Self {
        Self::with_base(cfg, rand::random::<u32>())
    }

    /// A table whose slots start at generation `base`; `new` draws it.
    pub fn with_base(cfg: Config, base: u32) -> Self {
        let n = cfg.capacity + cfg.ephemeral.max(1);
        Self {
            cfg,
            slots: (0..n)
                .map(|_| Slot {
                    gen: base,
                    session: None,
                    released: VecDeque::new(),
                })
                .collect(),
            base,
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

    fn idle(&self, s: &Session, now: Instant) -> bool {
        now.saturating_duration_since(s.last_used) > self.cfg.idle
    }

    /// Whether this process has issued generation `gen` in slot `i`.
    fn issued(&self, i: usize, gen: u32) -> bool {
        gen.wrapping_sub(self.base) <= self.slots[i].gen.wrapping_sub(self.base)
    }

    /// Empty slot `i` and move its generation: every id naming it misses.
    /// With a `reason`, the session is let go, and its ids are answered for.
    fn free(&mut self, i: usize, reason: Option<Reason>) -> Option<Session> {
        let slot = &mut self.slots[i];
        let old = Self::slot_bytes(slot);
        let retired = slot.session.take();
        if let Some(s) = retired.as_ref() {
            if let Some(reason) = reason {
                if slot.released.len() == RELEASES_KEPT {
                    slot.released.pop_front();
                }
                let nonces: Vec<_> = s.turns.iter().map(|t| t.nonce).collect();
                let dropped = s.dropped.clone();
                let bytes = nonces.capacity() * size_of::<u64>()
                    + dropped.capacity() * size_of::<Dropped>();
                slot.released.push_back(Released {
                    gen: slot.gen,
                    reason,
                    auth: s.auth,
                    nonces,
                    dropped,
                    bytes,
                });
                self.counters.released[reason as usize] += 1;
            }
        }
        slot.gen = slot.gen.wrapping_add(1);
        if self.occupant.is_some_and(|k| k.slot as usize == i) {
            self.occupant = None;
        }
        self.bytes = self.bytes - old + Self::slot_bytes(&self.slots[i]);
        self.trim_releases();
        retired
    }

    fn slot_bytes(slot: &Slot) -> usize {
        slot.session.as_ref().map_or(0, |s| s.bytes)
            + slot.released.capacity() * size_of::<Released>()
            + slot.released.iter().map(|r| r.bytes).sum::<usize>()
    }

    /// Release reasons are best-effort, authenticated history, never a budget bypass.
    fn trim_releases(&mut self) {
        while self.bytes > self.cfg.max_bytes {
            let Some(slot) = self.slots.iter_mut().find(|s| !s.released.is_empty()) else {
                break;
            };
            let old = Self::slot_bytes(slot);
            slot.released.pop_front();
            if slot.released.is_empty() {
                slot.released = VecDeque::new();
            }
            self.bytes = self.bytes - old + Self::slot_bytes(slot);
        }
    }

    /// Why slot `w.key.slot` let go the session `w` names, where `w`
    /// verifies against what it let go.
    fn released_reason(&self, w: WireId, auth: Option<[u8; 32]>) -> Option<Reason> {
        let r = self.slots[w.key.slot as usize]
            .released
            .iter()
            .find(|r| r.gen == w.key.gen)?;
        if r.auth != auth {
            return None;
        }
        match r.nonces.get(w.turn as usize) {
            Some(&n) if same(n, w.nonce) => Some(r.reason),
            _ => dropped_reason(&r.dropped, w),
        }
    }

    /// A slot for a new session among `range`: an empty one, else one past
    /// the idle policy, else the least recently used.
    fn take_slot(&mut self, range: std::ops::Range<usize>, now: Instant) -> usize {
        let empty = range.clone().find(|&i| self.slots[i].session.is_none());
        let idle = range.clone().find(|&i| {
            self.slots[i]
                .session
                .as_ref()
                .is_some_and(|s| self.idle(s, now))
        });
        let (pick, reason) = match (empty, idle) {
            (Some(i), _) => (i, None),
            (None, Some(i)) => (i, Some(Reason::Idle)),
            (None, None) => (
                range
                    .min_by_key(|&i| self.slots[i].session.as_ref().map(|s| s.last_used))
                    .expect("a non-empty range"),
                Some(Reason::Capacity),
            ),
        };
        if self.slots[pick].session.is_some() {
            self.counters.expired += 1;
        }
        drop(self.free(pick, reason));
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
    ) -> Result<(Key, usize), Lost> {
        let w = WireId::parse(id).ok_or(Lost::of(Miss::Malformed, None))?;
        let key = w.key;
        let i = key.slot as usize;
        if i >= self.slots.len() || !self.issued(i, key.gen) {
            return Err(Lost::of(Miss::Stale, Some(Reason::Restart)));
        }
        let ephemeral = self.ephemeral_slots().contains(&i);
        let (idle, gone) = {
            let Some(s) = self.get(key) else {
                return Err(Lost::of(Miss::Stale, self.released_reason(w, auth)));
            };
            if s.auth != auth {
                return Err(Lost::of(Miss::Binding, None));
            }
            match s.turns.get(w.turn as usize) {
                Some(t) if same(t.nonce, w.nonce) => {}
                _ => return Err(Lost::of(Miss::Turn, dropped_reason(&s.dropped, w))),
            }
            let gone = ephemeral && self.cfg.one_sequence && self.occupant != Some(key);
            (self.idle(s, now), gone)
        };
        if idle {
            self.counters.expired += 1;
            drop(self.free(i, Some(Reason::Idle)));
            return Err(Lost::of(Miss::Idle, Some(Reason::Idle)));
        }
        if gone {
            drop(self.free(i, Some(Reason::Capacity)));
            return Err(Lost::of(Miss::Gone, Some(Reason::Capacity)));
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
    ) -> Result<Resolved, Lost> {
        let found = self.find(id, auth, now);
        let (key, t) = match found {
            Ok(k) => k,
            Err(l) => {
                self.counters.expired += u64::from(l.miss != Miss::Idle);
                return Err(l);
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
            ids: s.ids[..(turn.ids_end - usize::from(turn.eot)).min(s.ids.len())].to_vec(),
            switches: s.switches,
            keep: s.keep,
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
    /// anew under the same session, and the dropped turns' ids miss with
    /// `reset` as their reason.
    pub fn reset(&mut self, key: Key) {
        let Some(s) = self.get_mut(key) else {
            return;
        };
        let old = s.bytes;
        for (turn, t) in s.turns.iter().enumerate() {
            let d = Dropped {
                turn: turn as u32,
                nonce: t.nonce,
                reason: Reason::Reset,
            };
            push_dropped(&mut s.dropped, d);
        }
        s.turns = Vec::new();
        s.ids = Vec::new();
        s.messages = Vec::new();
        s.payload_bytes = 0;
        s.turn_bytes = 0;
        s.account();
        let bytes = s.bytes;
        self.bytes = self.bytes - old + bytes;
        self.trim_releases();
        self.counters.released[Reason::Reset as usize] += 1;
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
                // Moving the lineage invalidates the old slot without a release reason.
                self.bytes -= s.bytes;
                drop(self.free(k.slot as usize, None));
                s.stored = true;
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
                    keep: None,
                    turns: Vec::new(),
                    messages: Vec::new(),
                    ids: Vec::new(),
                    created: now,
                    last_used: now,
                    switches: 0,
                    bytes: 0,
                    payload_bytes: 0,
                    turn_bytes: 0,
                    dropped: VecDeque::new(),
                });
                self.counters.created += 1;
                Key {
                    slot: i as u32,
                    gen: self.slots[i].gen,
                }
            }
        };
        let s = self.get_mut(key).expect("reserved above");
        let old = s.bytes;
        if plan.keep.is_some() {
            s.keep = plan.keep;
        }
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
        s.turn_bytes -= s.turns[kept..].iter().map(|t| t.bytes).sum::<usize>();
        s.turns.truncate(kept);
        for t in &mut s.turns {
            if t.ids_end > lcp {
                t.ids_end = lcp;
                t.eot = false;
            }
            if let Some(p) = t.prefix.take_if(|p| p.ids_len > lcp) {
                let bytes = p.retained_bytes();
                t.bytes -= bytes;
                s.turn_bytes -= bytes;
            }
        }
        s.last_used = now;
        s.account();
        let stored = s.stored;
        let bytes = s.bytes;
        self.bytes = self.bytes - old + bytes;
        self.trim_releases();
        Reservation {
            id: WireId {
                key,
                turn: kept as u32,
                nonce: rand::random::<u64>() | 1,
            },
            stored,
        }
    }

    /// Keep a reserved turn. `false` where the session moved under it (a
    /// concurrent request on the same session, or its slot given away): the
    /// id then names nothing.
    #[cfg(test)]
    pub fn commit(&mut self, r: Reservation, rec: Record, now: Instant) -> bool {
        self.publish(r, PreparedRecord::new(rec).expect("Value serializes"), now)
            .retained
    }

    pub fn publish(
        &mut self,
        r: Reservation,
        mut rec: PreparedRecord,
        now: Instant,
    ) -> Publication {
        let mut result = Publication {
            retained: false,
            reason: None,
            _retired: Vec::new(),
            _replaced: None,
            _unpublished: None,
        };
        let Some(s) = self.get_mut(r.id.key) else {
            result._unpublished = Some(rec);
            return result;
        };
        if s.turns.len() != r.id.turn as usize {
            result._unpublished = Some(rec);
            return result;
        }
        let old = s.bytes;
        rec.turn.nonce = r.id.nonce;
        s.turn_bytes += rec.turn.bytes;
        s.turns.push(rec.turn);
        result._replaced = Some((
            std::mem::replace(&mut s.ids, rec.ids),
            std::mem::replace(&mut s.messages, rec.messages),
        ));
        s.payload_bytes = rec.payload_bytes;
        s.last_used = now;
        s.account();
        let bytes = s.bytes;
        self.bytes = self.bytes - old + bytes;
        self.counters.committed += 1;
        if self.cfg.one_sequence {
            self.occupant = Some(r.id.key);
        }
        if bytes > self.cfg.max_bytes {
            self.counters.expired += 1;
            result.reason = Some(Reason::Budget);
            result
                ._retired
                .extend(self.free(r.id.key.slot as usize, Some(Reason::Budget)));
            return result;
        }
        self.trim_releases();
        while self.bytes > self.cfg.max_bytes {
            let victim = self
                .stored_slots()
                .filter(|&i| i != r.id.key.slot as usize)
                .filter_map(|i| self.slots[i].session.as_ref().map(|s| (i, s.last_used)))
                .min_by_key(|&(_, t)| t)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    self.counters.expired += 1;
                    result._retired.extend(self.free(i, Some(Reason::Memory)));
                }
                None => {
                    self.counters.expired += 1;
                    result.reason = Some(Reason::Budget);
                    result
                        ._retired
                        .extend(self.free(r.id.key.slot as usize, Some(Reason::Budget)));
                    return result;
                }
            }
        }
        result.retained = true;
        result
    }

    /// The engine's sequence served something no session holds (another
    /// route, a failed run): no session is resident any more.
    pub fn note_other(&mut self) {
        self.occupant = None;
    }

    /// `GET /v1l/sessions/{id}`: what the server holds of it — never its
    /// content. Its state, never an expiry: when a session goes is the
    /// server's policy (`/health`'s `sessions`), not the session's.
    pub fn inspect(
        &mut self,
        id: &str,
        auth: Option<[u8; 32]>,
        now: Instant,
    ) -> Result<Value, Lost> {
        let (key, t) = self.find(id, auth, now)?;
        let resident = self.cfg.one_sequence.then(|| self.occupant == Some(key));
        let s = self.get(key).expect("found above");
        Ok(json!({
            "id": id,
            "state": resident.map(|r| if r { "resident" } else { "evicted" }),
            "turn": t,
            "turns": s.turns.len(),
            "stored": s.stored,
            "resident": resident,
            "ids": s.ids.len(),
            "messages": s.messages.len(),
            "bytes": s.bytes,
            "switches": s.switches,
            "keep": s.keep.unwrap_or(self.cfg.keep).name(),
            "age_s": now.saturating_duration_since(s.created).as_secs(),
            "unused_s": now.saturating_duration_since(s.last_used).as_secs(),
        }))
    }

    /// `DELETE /v1l/sessions/{id}`: let the session go now.
    pub fn delete(&mut self, id: &str, auth: Option<[u8; 32]>, now: Instant) -> Result<(), Lost> {
        let (key, _) = self.find(id, auth, now)?;
        self.counters.expired += 1;
        drop(self.free(key.slot as usize, Some(Reason::Delete)));
        Ok(())
    }

    /// `/health`'s `sessions`: what is held, the retention policy that lets
    /// sessions go (`held` against `capacity` and `max_mib`, reclaimed after
    /// `idle_s` unused, the least recently used first), and what the policy
    /// and the clients have let go, by reason.
    pub fn json(&self) -> Value {
        let held =
            |r: std::ops::Range<usize>| r.filter(|&i| self.slots[i].session.is_some()).count();
        let released: serde_json::Map<String, Value> = Reason::ALL
            .iter()
            .filter(|r| **r != Reason::Restart)
            .map(|r| {
                (
                    r.name().to_string(),
                    json!(self.counters.released[*r as usize]),
                )
            })
            .collect();
        json!({
            "held": held(0..self.slots.len()),
            "stored": held(self.stored_slots()),
            "ephemeral": held(self.ephemeral_slots()),
            "capacity": self.cfg.capacity,
            "max_mib": self.cfg.max_bytes >> 20,
            "idle_s": self.cfg.idle.as_secs(),
            "reclaim": "lru by last use",
            "released": released,
            "keep": self.cfg.keep.name(),
            "bytes": self.bytes,
            "max_bytes": self.cfg.max_bytes,
            "accounting": ACCOUNTING,
            "base_bytes": size_of::<Self>() + self.slots.capacity() * size_of::<Slot>(),
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

mod storage;
pub use storage::StoredMessage;
