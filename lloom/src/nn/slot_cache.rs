//! `SlotCache`: keys resident in slots, with the liveness and the policy a
//! streamed tier needs (`profiles-and-memory-design.md` §4.4, L3
//! `SlotCache<K, B, E>`): which key is in which slot, the free slots, the
//! victim the policy names, and evicted slots held in a [`Retire`] until
//! their last GPU reader completes — or deferred while a device plan that
//! may read any slot is open. The slot *storage* (the expert planes, the PLE
//! rows) stays the owner's: the cache is the map and the rule, not the
//! bytes.
//!
//! Two axes are types (§4.4's `K` and `E`):
//! - the key index, [`KeyIndex`]: [`Dense`] for keys that are small
//!   integers (an expert id: a `Vec` indexed by the key, so a lookup on the
//!   decode path is an index, and the map is what the device-side plan
//!   reads as its residency table, [`SlotCache::slot_table`]); [`Hashed`]
//!   for sparse keys (a PLE row's token id);
//! - the eviction policy, [`Evict`]: [`TickLru`] for the expert tier (least
//!   recently used by the pass that used it, with pins, slots in flight and
//!   the epoch of each slot's last GPU reader); [`ListLru`] for a host-side
//!   row cache (an intrusive list, O(1) per lookup).
//!
//! The policy-specific operations are inherent on the concrete policy type,
//! so each owner sees only its own vocabulary and nothing is dispatched.

use lloom_tensor::retire::{Completed, Epoch, Retire, Tick};
use std::collections::HashMap;
use std::hash::Hash;

/// A slot of the cache. `u32`: the device table holds slot ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SlotId(pub u32);

impl SlotId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A routed expert's id, the streamed tier's key. Dense in `0..n_experts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ExpertId(pub u32);

/// A weight group's id (`profiles-and-memory-design.md` §4.8): a set of
/// tensors that load and evict together. Dense in `0..n_groups`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct WeightGroupId(pub u32);

impl WeightGroupId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A key a [`Dense`] index can index a `Vec` by.
pub trait DenseKey: Copy + Eq {
    fn index(self) -> usize;
    fn from_index(i: usize) -> Self;
}

impl DenseKey for ExpertId {
    fn index(self) -> usize {
        self.0 as usize
    }
    fn from_index(i: usize) -> Self {
        ExpertId(i as u32)
    }
}

impl DenseKey for WeightGroupId {
    fn index(self) -> usize {
        self.0 as usize
    }
    fn from_index(i: usize) -> Self {
        WeightGroupId(i as u32)
    }
}

/// "No slot" / "no key" in the dense tables, as the device reads them.
pub const NONE: u32 = u32::MAX;

/// The key → slot map.
pub trait KeyIndex<K>: std::fmt::Debug + Clone {
    fn get(&self, key: K) -> Option<SlotId>;
    fn set(&mut self, key: K, slot: SlotId);
    fn remove(&mut self, key: K);
}

/// A `Vec` indexed by the key: the dense index, and the device's table.
#[derive(Debug, Clone)]
pub struct Dense {
    slot_of: Vec<u32>,
}

impl Dense {
    pub fn new(n_keys: usize) -> Self {
        Self {
            slot_of: vec![NONE; n_keys],
        }
    }
}

impl<K: DenseKey> KeyIndex<K> for Dense {
    fn get(&self, key: K) -> Option<SlotId> {
        match self.slot_of.get(key.index()) {
            Some(&NONE) | None => None,
            Some(&s) => Some(SlotId(s)),
        }
    }
    fn set(&mut self, key: K, slot: SlotId) {
        self.slot_of[key.index()] = slot.0;
    }
    fn remove(&mut self, key: K) {
        self.slot_of[key.index()] = NONE;
    }
}

/// A hash map from the key: for sparse keys.
#[derive(Debug, Clone)]
pub struct Hashed<K> {
    slot_of: HashMap<K, u32>,
}

impl<K: Hash + Eq> Hashed<K> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            slot_of: HashMap::with_capacity(n),
        }
    }
}

impl<K: Hash + Eq + Copy + std::fmt::Debug> KeyIndex<K> for Hashed<K> {
    fn get(&self, key: K) -> Option<SlotId> {
        self.slot_of.get(&key).map(|&s| SlotId(s))
    }
    fn set(&mut self, key: K, slot: SlotId) {
        self.slot_of.insert(key, slot.0);
    }
    fn remove(&mut self, key: K) {
        self.slot_of.remove(&key);
    }
}

/// The eviction policy's per-slot state. Which slot is the victim is the
/// concrete policy's own operation (it takes the policy's own arguments).
pub trait Evict: std::fmt::Debug + Clone {
    fn new(n_slots: usize) -> Self;
    /// `slot` now holds a key.
    fn installed(&mut self, slot: SlotId);
    /// `slot` holds nothing.
    fn uninstalled(&mut self, slot: SlotId);
}

/// Least recently used by the pass that used it (a [`Tick`] per `ensure`),
/// with pins, slots in flight (a load issued, not landed) and the epoch of
/// each slot's last GPU reader: the expert tier's policy. A victim search is
/// a scan over the slots (a few hundred at most, once per miss).
#[derive(Debug, Clone)]
pub struct TickLru {
    last_use: Vec<Tick>,
    pinned: Vec<bool>,
    /// Per slot, the loader's event value its key lands on — in flight — or
    /// 0 when the slot holds a landed key or nothing.
    inflight: Vec<u64>,
    /// Per slot, the epoch of the last command buffer that read or wrote it.
    last_epoch: Vec<Epoch>,
    /// The epoch of the last device plan, which may read any slot.
    plan_epoch: Epoch,
    tick: Tick,
}

impl Evict for TickLru {
    fn new(n_slots: usize) -> Self {
        Self {
            last_use: vec![Tick::ZERO; n_slots],
            pinned: vec![false; n_slots],
            inflight: vec![0; n_slots],
            last_epoch: vec![Epoch::ZERO; n_slots],
            plan_epoch: Epoch::ZERO,
            tick: Tick::ZERO,
        }
    }
    fn installed(&mut self, slot: SlotId) {
        self.inflight[slot.index()] = 0;
    }
    fn uninstalled(&mut self, slot: SlotId) {
        self.inflight[slot.index()] = 0;
    }
}

/// Belady (§4.8): the plan's order of use is known, so the victim is the
/// resident slot whose key's next use is furthest away — never the one
/// needed next, which is what LRU evicts on a cyclic order (its hit rate
/// collapses to 0; this keeps `C − d` groups permanently resident and
/// streams the rest through a `d`-deep ring). Per slot: the plan position
/// of its key's next use, and the epoch of its last GPU reader.
#[derive(Debug, Clone)]
pub struct Belady {
    next_use: Vec<u64>,
    pinned: Vec<bool>,
    last_epoch: Vec<Epoch>,
}

impl Evict for Belady {
    fn new(n_slots: usize) -> Self {
        Self {
            next_use: vec![u64::MAX; n_slots],
            pinned: vec![false; n_slots],
            last_epoch: vec![Epoch::ZERO; n_slots],
        }
    }
    fn installed(&mut self, slot: SlotId) {
        self.next_use[slot.index()] = u64::MAX;
    }
    fn uninstalled(&mut self, slot: SlotId) {
        self.next_use[slot.index()] = u64::MAX;
    }
}

/// Least recently used, exactly: an intrusive doubly-linked list over the
/// slots (`prev` / `next` by slot; `head` most recent, `tail` least), O(1)
/// per lookup. A host-side row cache's policy.
#[derive(Debug, Clone)]
pub struct ListLru {
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    /// Slots never filled yet: a miss takes the next one before any eviction.
    fill: usize,
}

impl ListLru {
    fn unlink(&mut self, s: u32) {
        let (p, n) = (self.prev[s as usize], self.next[s as usize]);
        if p == NONE {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NONE {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.prev[s as usize] = NONE;
        self.next[s as usize] = NONE;
    }

    fn push_front(&mut self, s: u32) {
        self.prev[s as usize] = NONE;
        self.next[s as usize] = self.head;
        if self.head != NONE {
            self.prev[self.head as usize] = s;
        }
        self.head = s;
        if self.tail == NONE {
            self.tail = s;
        }
    }
}

impl Evict for ListLru {
    fn new(n_slots: usize) -> Self {
        Self {
            prev: vec![NONE; n_slots],
            next: vec![NONE; n_slots],
            head: NONE,
            tail: NONE,
            fill: 0,
        }
    }
    fn installed(&mut self, slot: SlotId) {
        self.push_front(slot.0);
    }
    fn uninstalled(&mut self, slot: SlotId) {
        self.unlink(slot.0);
    }
}

#[derive(Debug, Clone)]
pub struct SlotCache<K, I: KeyIndex<K>, E: Evict> {
    index: I,
    /// Per slot, its key.
    key_in: Vec<Option<K>>,
    policy: E,
    /// Slots kept free (the quarantined arm; 0: the in-place arm).
    spare: usize,
    /// Slots holding nothing whose last reader has completed.
    free: Vec<SlotId>,
    /// Slots evicted and not yet safe to write.
    quarantine: Retire<SlotId>,
    /// Slots evicted while a device plan was open: their readers are not
    /// all encoded yet, so the epoch they must outlive is known only once it
    /// settles (`release_deferred`).
    deferred: Vec<SlotId>,
    recycled: Vec<SlotId>,
    /// Slots in service: `0..active`. A slot at or past `active` is parked
    /// — never a victim, never taken; when its last reader completes it
    /// goes to `parked`, not to `free` (an elastic tenant's shrink,
    /// `profiles-and-memory-design.md` §4.2: the owner frees the storage
    /// past `active` once every parked slot's reader has completed, and
    /// allocates it again on a grow).
    active: usize,
    /// Parked slots whose last reader has completed (the quarantined arm's
    /// free entries past `active` go here at once).
    parked: Vec<SlotId>,
}

/// The expert tier's cache: dense ids, the tick policy.
pub type ExpertSlots = SlotCache<ExpertId, Dense, TickLru>;

impl ExpertSlots {
    pub fn new(n_experts: usize, n_slots: usize) -> Self {
        Self::with(Dense::new(n_experts), n_slots)
    }
}

impl<K: Copy + Eq, I: KeyIndex<K>, E: Evict> SlotCache<K, I, E> {
    pub fn with(index: I, n_slots: usize) -> Self {
        Self {
            index,
            key_in: vec![None; n_slots],
            policy: E::new(n_slots),
            spare: 0,
            free: Vec::new(),
            quarantine: Retire::new(),
            deferred: Vec::new(),
            recycled: Vec::new(),
            active: n_slots,
            parked: Vec::new(),
        }
    }

    pub fn n_slots(&self) -> usize {
        self.key_in.len()
    }

    /// Slots in service (`0..active`).
    pub fn active(&self) -> usize {
        self.active
    }

    /// Take slots `active..` out of service. Returns the resident keys in
    /// them (the caller evicts those, through its liveness); the empty ones
    /// among them are parked at once, the evicted ones when their last
    /// reader completes (`recycle`; `settled` says when every slot past
    /// `active` has). `active` never drops below `floor`.
    pub fn shrink_to(&mut self, active: usize, floor: usize) -> Vec<K> {
        let active = active.max(floor).min(self.n_slots());
        if active >= self.active {
            return Vec::new();
        }
        self.active = active;
        // Free-list entries past `active` are parked, not taken.
        let (keep, park): (Vec<SlotId>, Vec<SlotId>) =
            self.free.drain(..).partition(|s| s.index() < active);
        self.free = keep;
        self.parked.extend(park);
        let mut evict = Vec::new();
        for s in active..self.n_slots() {
            match self.key_in[s] {
                Some(k) => evict.push(k),
                None if !self.parked.contains(&SlotId(s as u32))
                    && !self.quarantine.iter().any(|q| q.index() == s)
                    && !self.deferred.contains(&SlotId(s as u32)) =>
                {
                    self.parked.push(SlotId(s as u32))
                }
                None => {}
            }
        }
        evict
    }

    /// Put slots `..active` back in service: parked slots below `active`
    /// are free again (the quarantined arm's list; the in-place arm takes
    /// an empty slot by scan). Returns the slots brought back.
    pub fn grow_to(&mut self, active: usize) -> Vec<SlotId> {
        let active = active.min(self.n_slots());
        if active <= self.active {
            return Vec::new();
        }
        self.active = active;
        let (mut back, stay): (Vec<SlotId>, Vec<SlotId>) =
            self.parked.drain(..).partition(|s| s.index() < active);
        self.parked = stay;
        // Descending, as `set_spare` lists them: pops fill from the low end.
        back.sort_by(|a, b| b.cmp(a));
        if self.quarantined() {
            self.free.extend(back.iter().copied());
        }
        back
    }

    /// Whether every slot past `active` is settled: none resident, none in
    /// the quarantine or deferred — their storage may be freed.
    pub fn settled_past_active(&self) -> bool {
        (self.active..self.n_slots()).all(|s| {
            self.key_in[s].is_none()
                && !self.quarantine.iter().any(|q| q.index() == s)
                && !self.deferred.contains(&SlotId(s as u32))
        })
    }

    /// Select the arm: `spare` slots kept free (0: in place). Only on an
    /// empty cache.
    pub fn set_spare(&mut self, spare: usize) {
        assert!(!self.any_resident(), "set_spare on a cache that holds keys");
        self.spare = spare.min(self.n_slots().saturating_sub(1));
        self.free.clear();
        if self.spare > 0 {
            // Descending, so the pops fill slot 0, 1, 2, … — the order the
            // in-place arm fills an empty cache in.
            self.free
                .extend((0..self.n_slots() as u32).rev().map(SlotId));
        }
    }

    /// Whether misses land in free slots (the quarantined arm).
    pub fn quarantined(&self) -> bool {
        self.spare > 0
    }

    pub fn spare(&self) -> usize {
        self.spare
    }

    /// Slots evicted and not yet reusable (quarantined or deferred).
    pub fn held(&self) -> usize {
        self.quarantine.held() + self.deferred.len()
    }

    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    pub fn any_resident(&self) -> bool {
        self.key_in.iter().any(|k| k.is_some())
    }

    /// Move every quarantined slot whose epoch has completed to the free
    /// list; how many.
    pub fn recycle(&mut self, completed: Completed) -> usize {
        self.quarantine.drain_into(completed, &mut self.recycled);
        let n = self.recycled.len();
        for s in self.recycled.drain(..) {
            if s.index() < self.active {
                self.free.push(s);
            } else {
                self.parked.push(s);
            }
        }
        n
    }

    /// A free slot, if any.
    pub fn take_free(&mut self) -> Option<SlotId> {
        self.free.pop()
    }

    /// Evict `slot`'s key out of the cache; the slot is quarantined at
    /// `epoch` — every reader encoded so far is in an epoch at most `epoch`
    /// — or deferred while a device plan is open. Returns the key evicted.
    pub fn evict(&mut self, slot: SlotId, epoch: Epoch, plan_open: bool) -> Option<K> {
        let k = self.uninstall(slot);
        if plan_open {
            self.deferred.push(slot);
        } else {
            self.quarantine.retire(epoch, slot);
        }
        k
    }

    /// The device plans are settled: every reader of a slot evicted while
    /// one was open is encoded now, in an epoch at most `epoch`.
    pub fn release_deferred(&mut self, epoch: Epoch) {
        for s in self.deferred.drain(..) {
            self.quarantine.retire(epoch, s);
        }
    }

    /// Empty `slot`, returning the key it held.
    pub fn uninstall(&mut self, slot: SlotId) -> Option<K> {
        let old = self.key_in[slot.index()].take();
        if let Some(k) = old {
            self.index.remove(k);
            self.policy.uninstalled(slot);
        }
        old
    }

    /// Put `key` in `slot`, displacing what was there.
    pub fn install(&mut self, slot: SlotId, key: K) {
        if let Some(old) = self.key_in[slot.index()].replace(key) {
            self.index.remove(old);
            self.policy.uninstalled(slot);
        }
        self.index.set(key, slot);
        self.policy.installed(slot);
    }

    /// The slot holding `key`, if resident.
    pub fn slot_of(&self, key: K) -> Option<SlotId> {
        self.index.get(key)
    }

    /// The key in `slot`, if any.
    pub fn key_in(&self, slot: SlotId) -> Option<K> {
        self.key_in[slot.index()]
    }

    /// Resident keys, in slot order.
    pub fn resident(&self) -> Vec<K> {
        self.key_in.iter().filter_map(|k| *k).collect()
    }

    /// Slots a load can take without evicting: holding nothing (the in-place
    /// arm), or free with the last reader complete (the quarantined arm).
    pub fn free_slots(&self) -> usize {
        if self.quarantined() {
            return self.free.len();
        }
        self.key_in[..self.active]
            .iter()
            .filter(|k| k.is_none())
            .count()
    }
}

impl<K: DenseKey, E: Evict> SlotCache<K, Dense, E> {
    /// The key → slot table as the device reads it (`NONE` for absent).
    pub fn slot_table(&self) -> &[u32] {
        &self.index.slot_of
    }
}

impl<K: Copy + Eq, I: KeyIndex<K>> SlotCache<K, I, TickLru> {
    /// Open a new use epoch (one per `ensure`).
    pub fn begin(&mut self) -> Tick {
        self.policy.tick = self.policy.tick.next();
        self.policy.tick
    }

    pub fn tick(&self) -> Tick {
        self.policy.tick
    }

    pub fn touch(&mut self, slot: SlotId, tick: Tick) {
        self.policy.last_use[slot.index()] = tick;
    }

    /// The least-recently-used resident unpinned slot not used in `tick`
    /// and not in flight.
    pub fn lru_resident(&self, tick: Tick) -> Option<SlotId> {
        let p = &self.policy;
        let mut best: Option<(usize, Tick)> = None;
        for s in 0..self.active {
            if self.key_in[s].is_none()
                || p.pinned[s]
                || p.last_use[s] == tick
                || p.inflight[s] != 0
            {
                continue;
            }
            match best {
                Some((_, t)) if t <= p.last_use[s] => {}
                _ => best = Some((s, p.last_use[s])),
            }
        }
        best.map(|(s, _)| SlotId(s as u32))
    }

    /// Evict the LRU key (not used in `tick`) out of the cache
    /// ([`Self::evict`]). Returns the key evicted.
    pub fn evict_lru(&mut self, tick: Tick, epoch: Epoch, plan_open: bool) -> Option<K> {
        let s = self.lru_resident(tick)?;
        self.evict(s, epoch, plan_open)
    }

    /// [`Self::victim_in`] preferring a slot whose last GPU reader's epoch
    /// `settled_before` covers (a direct read into it waits for nothing),
    /// skipping in-flight slots while any other serves; falling back to any.
    pub fn victim_settled(&self, tick: Tick, settled_before: Completed) -> Option<(SlotId, bool)> {
        self.victim_in(tick, true, settled_before)
            .or_else(|| self.victim_in(tick, true, Completed::ALL))
            .or_else(|| self.victim_in(tick, false, Completed::ALL))
    }

    /// The least-recently-used unpinned slot not used in `tick`, and whether
    /// it held a key (an eviction): over the slots not in flight
    /// (`skip_inflight`) or all of them, and with a last reader's epoch
    /// `settled_before` covers. An empty slot wins outright.
    pub fn victim_in(
        &self,
        tick: Tick,
        skip_inflight: bool,
        settled_before: Completed,
    ) -> Option<(SlotId, bool)> {
        let p = &self.policy;
        let mut best: Option<(usize, Tick)> = None;
        for s in 0..self.active {
            if p.pinned[s]
                || p.last_use[s] == tick
                || (skip_inflight && p.inflight[s] != 0)
                || !settled_before.covers(self.reader_epoch(SlotId(s as u32)))
            {
                continue;
            }
            if self.key_in[s].is_none() {
                return Some((SlotId(s as u32), false));
            }
            match best {
                Some((_, t)) if t <= p.last_use[s] => {}
                _ => best = Some((s, p.last_use[s])),
            }
        }
        best.map(|(s, _)| (SlotId(s as u32), true))
    }

    /// The epoch of `slot`'s last GPU reader or writer.
    pub fn reader_epoch(&self, slot: SlotId) -> Epoch {
        self.policy.last_epoch[slot.index()].max(self.policy.plan_epoch)
    }

    /// `slot` was read or written by a command buffer in `epoch`.
    pub fn stamp(&mut self, slot: SlotId, epoch: Epoch) {
        self.policy.last_epoch[slot.index()] = epoch;
    }

    /// A device plan in `epoch` may read every slot.
    pub fn plan_read_all(&mut self, epoch: Epoch) {
        self.policy.plan_epoch = self.policy.plan_epoch.max(epoch);
    }

    /// The epoch of the last device plan ([`Self::plan_read_all`]).
    pub fn plan_epoch(&self) -> Epoch {
        self.policy.plan_epoch
    }

    pub fn inflight(&self, slot: SlotId) -> u64 {
        self.policy.inflight[slot.index()]
    }

    pub fn set_inflight(&mut self, slot: SlotId, value: u64) {
        self.policy.inflight[slot.index()] = value;
    }

    /// Slots whose key is in flight.
    pub fn inflight_count(&self) -> usize {
        self.policy.inflight.iter().filter(|&&v| v != 0).count()
    }

    /// Forget the in-flight state of every slot whose load landed (event
    /// value ≤ `done`); how many.
    pub fn settle_inflight(&mut self, done: u64) -> u64 {
        let mut n = 0;
        for v in self.policy.inflight.iter_mut() {
            if *v != 0 && *v <= done {
                *v = 0;
                n += 1;
            }
        }
        n
    }

    pub fn set_pinned(&mut self, slot: SlotId, pinned: bool) {
        self.policy.pinned[slot.index()] = pinned;
    }

    pub fn is_pinned(&self, slot: SlotId) -> bool {
        self.policy.pinned[slot.index()]
    }

    pub fn pinned_count(&self) -> usize {
        self.policy.pinned.iter().filter(|&&p| p).count()
    }
}

impl<K: Copy + Eq, I: KeyIndex<K>> SlotCache<K, I, Belady> {
    /// `slot`'s key is next used at plan position `at` (`u64::MAX`: never
    /// again).
    pub fn set_next_use(&mut self, slot: SlotId, at: u64) {
        self.policy.next_use[slot.index()] = at;
    }

    pub fn next_use(&self, slot: SlotId) -> u64 {
        self.policy.next_use[slot.index()]
    }

    pub fn set_pinned(&mut self, slot: SlotId, pinned: bool) {
        self.policy.pinned[slot.index()] = pinned;
    }

    /// `slot` was read by a command buffer in `epoch`.
    pub fn stamp(&mut self, slot: SlotId, epoch: Epoch) {
        self.policy.last_epoch[slot.index()] = epoch;
    }

    pub fn reader_epoch(&self, slot: SlotId) -> Epoch {
        self.policy.last_epoch[slot.index()]
    }

    /// The slot a load takes: an empty slot in service, else the resident
    /// unpinned slot whose key's next use is furthest away (ties: the
    /// lowest slot), and whether it held a key. `settled_before` restricts
    /// the choice to slots whose last reader it covers; `None` when nothing
    /// qualifies.
    pub fn victim(&self, settled_before: Completed) -> Option<(SlotId, bool)> {
        let p = &self.policy;
        let mut best: Option<(usize, u64)> = None;
        for s in 0..self.active {
            if p.pinned[s] || !settled_before.covers(p.last_epoch[s]) {
                continue;
            }
            if self.key_in[s].is_none() {
                return Some((SlotId(s as u32), false));
            }
            match best {
                Some((_, far)) if far >= p.next_use[s] => {}
                _ => best = Some((s, p.next_use[s])),
            }
        }
        best.map(|(s, _)| (SlotId(s as u32), true))
    }
}

impl<K: Copy + Eq, I: KeyIndex<K>> SlotCache<K, I, ListLru> {
    /// `slot` is the most recently used.
    pub fn touch(&mut self, slot: SlotId) {
        if self.policy.head != slot.0 {
            self.policy.unlink(slot.0);
            self.policy.push_front(slot.0);
        }
    }

    /// The slot holding `key`, moved to most recent.
    pub fn lookup(&mut self, key: K) -> Option<SlotId> {
        let s = self.index.get(key)?;
        self.touch(s);
        Some(s)
    }

    /// The slot a new key takes: a never-filled slot while one remains,
    /// else the least recently used, whose key is displaced. Returns the
    /// slot and the key it displaced; the caller writes the slot's storage
    /// and installs the key.
    pub fn claim(&mut self) -> (SlotId, Option<K>) {
        if self.policy.fill < self.n_slots() {
            let s = SlotId(self.policy.fill as u32);
            self.policy.fill += 1;
            return (s, None);
        }
        let s = SlotId(self.policy.tail);
        let old = self.uninstall(s);
        (s, old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dense_cache_installs_evicts_lru_and_recycles_through_the_quarantine() {
        let mut c = ExpertSlots::new(8, 3);
        let t = c.begin();
        c.install(SlotId(0), ExpertId(5));
        c.touch(SlotId(0), t);
        c.install(SlotId(1), ExpertId(6));
        assert_eq!(c.slot_of(ExpertId(5)), Some(SlotId(0)));
        assert_eq!(c.key_in(SlotId(1)), Some(ExpertId(6)));
        assert_eq!(c.slot_table()[5], 0);
        assert_eq!(c.slot_table()[7], NONE);
        assert_eq!(c.free_slots(), 1, "the in-place arm counts empty slots");
        // The LRU is slot 1 (never touched in this tick); slot 0 was.
        assert_eq!(c.evict_lru(t, Epoch(3), false), Some(ExpertId(6)));
        assert_eq!(c.slot_table()[6], NONE);
        assert_eq!(c.held(), 1);
        assert_eq!(c.recycle(Completed(3)), 0, "its epoch is still open");
        assert_eq!(c.recycle(Completed(4)), 1);
        assert_eq!(c.take_free(), Some(SlotId(1)));
        // Pinned slots are never victims; an empty slot wins outright — the
        // first by index (slot 1, emptied by the eviction above, before 2).
        c.set_pinned(SlotId(0), true);
        let t = c.begin();
        assert_eq!(
            c.victim_in(t, true, Completed::ALL),
            Some((SlotId(1), false))
        );
        c.install(SlotId(1), ExpertId(2));
        c.touch(SlotId(1), t);
        assert_eq!(
            c.victim_in(t, true, Completed::ALL),
            Some((SlotId(2), false))
        );
        c.install(SlotId(2), ExpertId(3));
        assert_eq!(
            c.victim_in(t, true, Completed::ALL),
            Some((SlotId(2), true)),
            "slot 0 pinned, slot 1 used this tick: slot 2 is the LRU"
        );
        assert_eq!(c.resident(), vec![ExpertId(5), ExpertId(2), ExpertId(3)]);
    }

    #[test]
    fn a_shrink_parks_the_top_slots_and_a_grow_brings_them_back() {
        let mut c = ExpertSlots::new(8, 4);
        c.set_spare(1);
        let t = c.begin();
        for (s, e) in [(0u32, 5u32), (1, 6), (2, 7)] {
            let got = c.take_free().unwrap();
            assert_eq!(got, SlotId(s));
            c.install(got, ExpertId(e));
        }
        c.touch(SlotId(2), t);
        // Shrink to two slots: 7 (slot 2) and nothing in slot 3 (free)
        // leave service; slot 3 is parked at once, slot 2 once evicted.
        let evict = c.shrink_to(2, 1);
        assert_eq!(evict, vec![ExpertId(7)]);
        assert_eq!(c.active(), 2);
        assert_eq!(c.free_len(), 0, "the free slot past `active` was parked");
        c.evict(SlotId(2), Epoch(4), false);
        assert!(!c.settled_past_active(), "slot 2 is in the quarantine");
        assert_eq!(c.recycle(Completed(5)), 1);
        assert!(c.settled_past_active(), "recycled past `active`: parked");
        assert_eq!(c.free_len(), 0);
        let t = c.begin();
        assert_eq!(
            c.lru_resident(t),
            Some(SlotId(0)),
            "victims come from the active slots"
        );
        assert!(c.victim_in(t, true, Completed::ALL).unwrap().0.index() < 2);
        // Grow back to four: both parked slots return to the free list,
        // low slot first on pop.
        let back = c.grow_to(4);
        assert_eq!(back.len(), 2);
        assert_eq!(c.take_free(), Some(SlotId(2)));
        assert_eq!(c.take_free(), Some(SlotId(3)));
        // Never below the floor: a shrink to 0 with a floor of 1 keeps slot
        // 0 (expert 5) in service and evicts slot 1's.
        assert_eq!(
            c.shrink_to(0, 1),
            vec![ExpertId(6)],
            "never below the floor"
        );
        assert_eq!(c.active(), 1);
    }

    #[test]
    fn a_deferred_eviction_waits_for_the_settle() {
        let mut c = ExpertSlots::new(4, 2);
        let t = c.begin();
        c.install(SlotId(0), ExpertId(1));
        assert_eq!(c.evict_lru(t, Epoch(1), true), Some(ExpertId(1)));
        assert_eq!(c.recycle(Completed::ALL), 0, "deferred, not quarantined");
        c.release_deferred(Epoch(9));
        assert_eq!(c.recycle(Completed(9)), 0);
        assert_eq!(c.recycle(Completed(10)), 1);
        assert_eq!(c.settle_inflight(0), 0);
        c.set_inflight(SlotId(1), 7);
        assert_eq!(c.inflight_count(), 1);
        assert_eq!(c.settle_inflight(7), 1);
        assert_eq!(c.inflight(SlotId(1)), 0);
    }

    /// Belady on a cyclic order (§4.8, and §11's deliberate break): with N
    /// groups used in a fixed cycle and C < N slots, the victim is always a
    /// group needed further ahead than the next one, so at least C − 1 of
    /// every N hit once the cycle is warm — where LRU on the same order
    /// evicts exactly the group needed next every time and hits nothing.
    #[test]
    fn belady_hits_on_a_cyclic_order_where_lru_hits_nothing() {
        let (n, c) = (6u32, 4usize);
        let cycle: Vec<WeightGroupId> = (0..n).map(WeightGroupId).collect();
        let next_use = |g: WeightGroupId, pos: u64| -> u64 {
            let idx = g.0 as u64;
            let n = n as u64;
            let at = (pos / n) * n + idx;
            if at > pos {
                at
            } else {
                at + n
            }
        };
        let rounds = 5u64;
        // Belady.
        let mut cache: SlotCache<WeightGroupId, Dense, Belady> =
            SlotCache::with(Dense::new(n as usize), c);
        let (mut hits, mut misses) = (0u64, 0u64);
        for pos in 0..(rounds * n as u64) {
            let g = cycle[(pos % n as u64) as usize];
            match cache.slot_of(g) {
                Some(s) => {
                    hits += 1;
                    cache.set_next_use(s, next_use(g, pos));
                }
                None => {
                    misses += 1;
                    let (s, evicted) = cache.victim(Completed::ALL).expect("a slot");
                    if evicted {
                        // Never the group needed next.
                        let needed_next = cycle[((pos + 1) % n as u64) as usize];
                        assert_ne!(cache.key_in(s), Some(needed_next), "evicted the next group");
                        assert!(cache.next_use(s) > pos + 1, "evicted a group needed next");
                    }
                    cache.install(s, g);
                    cache.set_next_use(s, next_use(g, pos));
                }
            }
        }
        assert_eq!(hits + misses, rounds * n as u64);
        assert!(
            hits >= (rounds - 1) * (c as u64 - 1),
            "Belady hit {hits} of {} after the first cycle: below the (C − 1) / N bound",
            rounds * n as u64
        );
        assert!(
            misses >= n as u64 + (rounds - 1),
            "{misses}: a C-slot cache cannot hit every N"
        );
        // LRU on the same order: nothing hits after the first fill.
        let mut lru = ExpertSlots::new(n as usize, c);
        let (mut lru_hits, mut lru_misses) = (0u64, 0u64);
        for pos in 0..(rounds * n as u64) {
            let g = ExpertId(cycle[(pos % n as u64) as usize].0);
            let tick = lru.begin();
            match lru.slot_of(g) {
                Some(s) => {
                    lru_hits += 1;
                    lru.touch(s, tick);
                }
                None => {
                    lru_misses += 1;
                    let (s, _) = lru.victim_in(tick, false, Completed::ALL).expect("a slot");
                    lru.install(s, g);
                    lru.touch(s, tick);
                }
            }
        }
        assert_eq!(
            (lru_hits, lru_misses),
            (0, rounds * n as u64),
            "LRU on a cycle hits nothing"
        );
    }

    #[test]
    fn a_hashed_list_cache_fills_then_displaces_the_least_recent() {
        let mut c: SlotCache<i64, Hashed<i64>, ListLru> =
            SlotCache::with(Hashed::with_capacity(3), 3);
        for id in [10i64, 20, 30] {
            let (s, old) = c.claim();
            assert_eq!(old, None);
            c.install(s, id);
        }
        assert_eq!(c.resident(), vec![10, 20, 30]);
        // 10 is touched: 20 is now the least recent.
        assert_eq!(c.lookup(10), Some(SlotId(0)));
        assert_eq!(c.lookup(99), None);
        let (s, old) = c.claim();
        assert_eq!((s, old), (SlotId(1), Some(20)));
        c.install(s, 40);
        assert_eq!(c.slot_of(20), None);
        assert_eq!(c.slot_of(40), Some(SlotId(1)));
        // Order now: 40, 10, 30 → 30 is the least recent.
        assert_eq!(c.claim(), (SlotId(2), Some(30)));
    }
}
