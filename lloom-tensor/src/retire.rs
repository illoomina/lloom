//! The one "safe to reuse" rule (`profiles-and-memory-design.md` §4.5): an
//! item retired at an epoch is reusable once the GPU has finished that epoch.
//!
//! [`GpuClock`](crate::kernels::metal::GpuClock) counts command buffers: a
//! buffer is committed in an [`Epoch`], and its completion handler raises the
//! count of [`Completed`] epochs. Anything the host gives up while a buffer
//! may still read it — a pool buffer, a KV page, a streamed expert's slot, a
//! staging entry — is held in a [`Retire`] against the epoch open at the
//! moment it was given up, and comes back only once that epoch is complete.
//! That is the whole liveness discipline, written once; the users differ only
//! in what they hold. `MTLFence` orders work the encoder can see; this hazard
//! is between a read already encoded and a write not yet encoded, possibly in
//! another command buffer, and no barrier expresses it (§10.3h, §6.7 L4).
//!
//! Why the epoch is taken when the item is retired and not when its refcount
//! fell: the item must outlive every command buffer that could read it, and
//! the last of those is the one open when the host gives it up. Later is
//! always safe; earlier is not.
//!
//! Backend-neutral on purpose, like `kv_pages`: a CPU test drives it with a
//! counter, and the Metal clock is one producer of its numbers.

use std::collections::VecDeque;

/// A command buffer's epoch: the value of the clock's submit counter when the
/// buffer was open. A newtype so an epoch is never added to a tick, an IO
/// event value or a count of completed epochs by accident.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

impl Epoch {
    /// Before any command buffer: the epoch a fresh clock is in.
    pub const ZERO: Epoch = Epoch(0);

    /// `n` epochs later: the pool's `LLOOM_POOL_QUARANTINE=K` instrument
    /// holds a buffer `K` command buffers past its own.
    pub fn plus(self, n: u64) -> Epoch {
        Epoch(self.0.saturating_add(n))
    }
}

impl std::fmt::Display for Epoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// How many epochs the GPU has finished. Zero means none; epoch `e` is
/// finished iff the count exceeds it — the clock stores `epoch + 1` so that
/// "nothing has completed" and "epoch 0 has" are different states.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Completed(pub u64);

impl Completed {
    /// Nothing finished.
    pub const NONE: Completed = Completed(0);
    /// Everything finished: what a holder without a clock (a CPU run) uses,
    /// so a retired item is reusable at once.
    pub const ALL: Completed = Completed(u64::MAX);

    /// Whether work encoded in `epoch` is known to have finished.
    pub fn covers(self, epoch: Epoch) -> bool {
        self.0 > epoch.0
    }
}

impl std::fmt::Display for Completed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A use epoch of a cache's own: the streamed tier opens one per `ensure`
/// and marks what it touched in it, so a slot used in this tick is never a
/// victim in it. Not a [`Epoch`]: it counts calls, not command buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tick(pub u64);

impl std::fmt::Display for Tick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Tick {
    pub const ZERO: Tick = Tick(0);
    /// A tick no use is ever in: "protect nothing" for a victim search.
    pub const NEVER: Tick = Tick(u64::MAX);

    /// The next tick.
    pub fn next(self) -> Tick {
        Tick(self.0 + 1)
    }
}

/// Items given up on the host and not yet safe to reuse, each held against
/// the epoch it must outlive — the queue the whole tree retires through.
///
/// Oldest first: the epoch is non-decreasing by construction (the clock only
/// grows), so draining is a prefix walk that stops at the first entry still
/// outstanding — never a scan, which §15.2 #10 forbids on the per-token path.
#[derive(Clone, Debug)]
pub struct Retire<I> {
    pending: VecDeque<(Epoch, I)>,
}

impl<I> Default for Retire<I> {
    fn default() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }
}

impl<I> Retire<I> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold `item` until `epoch` — the one open at the moment of the call,
    /// which a command buffer being encoded right now belongs to — has
    /// completed. Allocates nothing once the queue has reached its working
    /// size.
    pub fn retire(&mut self, epoch: Epoch, item: I) {
        self.pending.push_back((epoch, item));
    }

    /// [`Self::retire`] for every item of `items`. Empty input records
    /// nothing, so an idle step costs nothing.
    pub fn retire_all(&mut self, epoch: Epoch, items: Vec<I>) {
        self.pending.extend(items.into_iter().map(|i| (epoch, i)));
    }

    /// The oldest item whose epoch `ready` accepts, removed; `None` when the
    /// oldest is not ready (then no later one is).
    pub fn pop_ready(&mut self, ready: impl FnOnce(Epoch) -> bool) -> Option<I> {
        let (epoch, _) = self.pending.front()?;
        if !ready(*epoch) {
            return None;
        }
        self.pending.pop_front().map(|(_, i)| i)
    }

    /// Every item whose epoch `completed` covers, appended to `out` (which
    /// keeps its capacity across calls) and removed from the queue.
    pub fn drain_into(&mut self, completed: Completed, out: &mut Vec<I>) {
        while let Some(i) = self.pop_ready(|e| completed.covers(e)) {
            out.push(i);
        }
    }

    /// [`Self::drain_into`] into a fresh vector.
    pub fn drain(&mut self, completed: Completed) -> Vec<I> {
        let mut out = Vec::new();
        self.drain_into(completed, &mut out);
        out
    }

    /// How many items a drain at `completed` would return, without draining:
    /// what a planner may count as available before the drain runs.
    pub fn ready(&self, completed: Completed) -> usize {
        self.pending
            .iter()
            .take_while(|(epoch, _)| completed.covers(*epoch))
            .count()
    }

    /// Items held and not yet eligible.
    ///
    /// The quantity a leak would grow without bound, so it is worth reporting
    /// rather than only asserting — §3.4a-iv's *"a curve that climbs without
    /// plateauing is a leak or an allocator defect"*.
    pub fn held(&self) -> usize {
        self.pending.len()
    }

    /// Whether anything is held.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// The epoch the oldest item waits for, if any.
    pub fn head_epoch(&self) -> Option<Epoch> {
        self.pending.front().map(|(e, _)| *e)
    }

    /// The held items, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &I> {
        self.pending.iter().map(|(_, i)| i)
    }
}

/// A future use a retired item still owes, which keeps it from reuse past
/// its epoch (`docs/dependency-ordered-reuse-design.md` §3, §14.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FutureUse {
    /// No owner, view, export or replay can issue another use: the item is
    /// reusable once its epoch completes.
    Closed,
    /// ONE use declared ahead of encoding, not an arbitrary set of future
    /// uses. Its receipt is stamped at discharge; actual additional uses need
    /// a separately agreed join/closure contract before encoding. The receipt
    /// must name the same clock domain and cannot predate retirement.
    Declared,
}

/// One retired item's place in a [`RetireIndex`]: the record and the
/// generation the item was retired under. A ticket whose generation the
/// record no longer has names an item that was withdrawn or reused, and
/// every operation on it leaves the index unchanged. A ticket is bookkeeping,
/// not physical custody, source authority or permission to replay a use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    slot: u32,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum DischargePhase {
    Awaiting,
    Closed,
    Discharged,
    Conflicted,
}

#[repr(C)]
struct DischargeReceipt {
    // Retirement epoch while Awaiting/Closed; first accepted use thereafter.
    epoch: u64,
    phase: DischargePhase,
}

/// Disposition of ONE declared future use, not authority for storage reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "handle the receipt disposition; it is not storage-reuse authority"]
pub enum DischargeDisposition {
    /// The first valid receipt queued this retirement's only notice.
    Queued,
    /// The identical receipt was already queued; nothing changed.
    AlreadyQueued,
    /// Closed retirement; no declared use can be discharged.
    NotDeclared { retired: Epoch },
    /// No current item has this ticket's generation.
    Stale,
    /// Invalid first receipt; the declaration and its notice credit remain
    /// awaiting a corrected receipt.
    BeforeRetirement { retired: Epoch, attempted: Epoch },
    /// A different repeated receipt quarantined the item. The attempted epoch
    /// is returned, not retained as per-record history.
    Incompatible { accepted: Epoch, attempted: Epoch },
    /// Already conflicted; no further receipt can release the item.
    Quarantined { accepted: Epoch, attempted: Epoch },
}

struct Record<I> {
    item: Option<I>,
    generation: u64,
    receipt: DischargeReceipt,
}

/// Items given up and not yet safe to reuse, as [`Retire`] holds them, with
/// one owner per item: a stable record. The epoch queue holds versioned
/// notices into the records rather than the items, so an item can leave
/// before its epoch completes ([`Self::withdraw`]) and its notice then
/// finds a newer generation and does nothing: a late completion never moves,
/// poisons or frees what took the record's place.
///
/// At most one notice is queued per retirement. Identical discharge receipts
/// are idempotent; incompatible receipts quarantine the current item rather
/// than select an epoch for multiple uses. A conflicted notice is consumed
/// without releasing its item. Withdrawal still requires independent ordering.
/// Capacity for each declared notice is reserved at retirement, not discharge.
/// Records, vacancies and notices retain their capacity.
pub struct RetireIndex<I> {
    records: Vec<Record<I>>,
    vacant: Vec<u32>,
    notices: Retire<Ticket>,
    parked: usize,
    /// Reserved notice credits for current Awaiting declarations.
    unqueued: usize,
    stale: u64,
}

impl<I> Default for RetireIndex<I> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            vacant: Vec::new(),
            notices: Retire::new(),
            parked: 0,
            unqueued: 0,
            stale: 0,
        }
    }
}

impl<I> RetireIndex<I> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold `item` until `epoch` — the one open at the call — completes, and,
    /// under [`FutureUse::Declared`], until it is discharged as well. Notice
    /// capacity is reserved here so discharge never grows the queue.
    pub fn retire(&mut self, epoch: Epoch, item: I, future: FutureUse) -> Ticket {
        self.notices
            .pending
            .reserve(self.unqueued.checked_add(1).expect("notice capacity"));
        let slot = match self.vacant.pop() {
            Some(slot) => slot,
            None => {
                self.records.push(Record {
                    item: None,
                    generation: 0,
                    receipt: DischargeReceipt {
                        epoch: 0,
                        phase: DischargePhase::Awaiting,
                    },
                });
                u32::try_from(self.records.len() - 1).expect("fewer than 2^32 parked records")
            }
        };
        let record = &mut self.records[slot as usize];
        debug_assert!(record.item.is_none(), "a vacant record holds an item");
        record.item = Some(item);
        record.receipt = DischargeReceipt {
            epoch: epoch.0,
            phase: if future == FutureUse::Closed {
                DischargePhase::Closed
            } else {
                DischargePhase::Awaiting
            },
        };
        if future == FutureUse::Declared {
            self.unqueued += 1;
        }
        self.parked += 1;
        let ticket = Ticket {
            slot,
            generation: record.generation,
        };
        if future == FutureUse::Closed {
            self.notices.retire(epoch, ticket);
        }
        ticket
    }

    /// Record the epoch open at ONE declared use, in the retirement's clock
    /// domain. An earlier first receipt is refused without consuming the
    /// declaration or its reserved credit; a corrected receipt may retry.
    /// Numeric comparison does not verify domain identity or native safety.
    ///
    /// An identical accepted receipt is idempotent. A different repeated epoch
    /// quarantines this retirement: neither epoch is a join of multiple uses.
    /// This receipt is not physical, source or replay authority.
    pub fn discharge(&mut self, ticket: Ticket, epoch: Epoch) -> DischargeDisposition {
        if !self.current(ticket) {
            return DischargeDisposition::Stale;
        }
        let receipt = &mut self.records[ticket.slot as usize].receipt;
        let accepted = Epoch(receipt.epoch);
        match receipt.phase {
            DischargePhase::Awaiting if epoch < accepted => {
                DischargeDisposition::BeforeRetirement {
                    retired: accepted,
                    attempted: epoch,
                }
            }
            DischargePhase::Awaiting => {
                receipt.epoch = epoch.0;
                receipt.phase = DischargePhase::Discharged;
                self.unqueued -= 1;
                self.notices.retire(epoch, ticket);
                DischargeDisposition::Queued
            }
            DischargePhase::Closed => DischargeDisposition::NotDeclared { retired: accepted },
            DischargePhase::Discharged if accepted == epoch => DischargeDisposition::AlreadyQueued,
            DischargePhase::Discharged => {
                receipt.phase = DischargePhase::Conflicted;
                DischargeDisposition::Incompatible {
                    accepted,
                    attempted: epoch,
                }
            }
            DischargePhase::Conflicted => DischargeDisposition::Quarantined {
                accepted,
                attempted: epoch,
            },
        }
    }

    /// The oldest item whose notice `ready` accepts, removed; `None` when the
    /// oldest notice is not ready (then no later one is). A ready notice whose
    /// record moved on is dropped and counted ([`Self::stale_notices`]). A
    /// conflicted current notice is consumed without taking its held item.
    pub fn pop_ready(&mut self, mut ready: impl FnMut(Epoch) -> bool) -> Option<I> {
        loop {
            let ticket = self.notices.pop_ready(&mut ready)?;
            if self.current(ticket)
                && self.records[ticket.slot as usize].receipt.phase == DischargePhase::Conflicted
            {
                continue;
            }
            if let Some(item) = self.take(ticket) {
                return Some(item);
            }
            self.stale += 1;
        }
    }

    /// Take the item `ticket` names before its epoch completes, for a caller
    /// whose reuse is ordered by other means. `None` when the ticket is stale
    /// or its item was already taken. Its notice stays queued and finds the
    /// record moved on. Awaiting declarations release their reserved credit.
    /// Conflicted items can also be withdrawn, but the caller must separately
    /// prove future-use closure, ordering, initialization and physical/native
    /// custody. A ticket or receipt alone proves none of those obligations.
    pub fn withdraw(&mut self, ticket: Ticket) -> Option<I> {
        self.take(ticket)
    }

    fn current(&self, ticket: Ticket) -> bool {
        self.records
            .get(ticket.slot as usize)
            .is_some_and(|r| r.generation == ticket.generation && r.item.is_some())
    }

    fn take(&mut self, ticket: Ticket) -> Option<I> {
        if !self.current(ticket) {
            return None;
        }
        let record = &mut self.records[ticket.slot as usize];
        if record.receipt.phase == DischargePhase::Awaiting {
            self.unqueued -= 1;
        }
        let item = record.item.take();
        record.generation += 1;
        self.vacant.push(ticket.slot);
        self.parked -= 1;
        item
    }

    /// Items held and not yet taken.
    pub fn held(&self) -> usize {
        self.parked
    }

    /// Notices queued, stale ones included: the metadata the epoch queue
    /// holds.
    pub fn notices(&self) -> usize {
        self.notices.held()
    }

    /// Records allocated, held or vacant: the metadata the slab holds.
    pub fn records(&self) -> usize {
        self.records.len()
    }

    /// Ready notices dropped because their record had moved on, since
    /// construction.
    pub fn stale_notices(&self) -> u64 {
        self.stale
    }

    pub fn is_empty(&self) -> bool {
        self.parked == 0
    }

    /// The epoch the oldest notice waits for, if any.
    pub fn head_epoch(&self) -> Option<Epoch> {
        self.notices.head_epoch()
    }

    /// The held items, in record order.
    pub fn iter(&self) -> impl Iterator<Item = &I> {
        self.records.iter().filter_map(|r| r.item.as_ref())
    }

    /// The held items with their tickets, in record order.
    pub fn tickets(&self) -> impl Iterator<Item = (Ticket, &I)> {
        self.records.iter().enumerate().filter_map(|(slot, r)| {
            r.item.as_ref().map(|item| {
                (
                    Ticket {
                        slot: slot as u32,
                        generation: r.generation,
                    },
                    item,
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_item_comes_back_only_once_its_epoch_is_covered() {
        let mut r = Retire::new();
        r.retire(Epoch(3), "a");
        r.retire_all(Epoch(5), vec!["b", "c"]);
        assert_eq!(r.held(), 3);
        assert_eq!(
            r.ready(Completed(3)),
            0,
            "a count equal to the epoch does not cover it"
        );
        assert_eq!(r.ready(Completed(4)), 1);
        assert_eq!(r.drain(Completed(4)), vec!["a"]);
        assert_eq!(r.head_epoch(), Some(Epoch(5)));
        assert!(r.drain(Completed(5)).is_empty());
        assert_eq!(r.drain(Completed::ALL), vec!["b", "c"]);
        assert!(r.is_empty());
    }

    #[test]
    fn a_pop_stops_at_the_first_unready_entry() {
        let mut r = Retire::new();
        r.retire(Epoch(1), 1u8);
        r.retire(Epoch(2), 2u8);
        assert_eq!(r.pop_ready(|e| e <= Epoch(1)), Some(1));
        assert_eq!(r.pop_ready(|e| e <= Epoch(1)), None);
        assert_eq!(r.held(), 1);
    }

    /// A withdrawn item's notice finds its record moved on: the later
    /// completion takes nothing, and in particular not the item retired
    /// into the record after it, which waits for its own epoch. The record is
    /// reused, so the notice's generation is the only thing between the two.
    #[test]
    fn a_stale_notice_never_releases_the_records_next_item() {
        let mut r = RetireIndex::new();
        let old = r.retire(Epoch(5), "old", FutureUse::Closed);
        assert_eq!(r.withdraw(old), Some("old"));
        assert_eq!(r.withdraw(old), None, "a ticket withdraws once");
        let new = r.retire(Epoch(9), "new", FutureUse::Closed);
        assert_eq!(r.records(), 1, "the vacated record was not reused");
        assert_eq!(r.notices(), 2);
        assert_eq!(
            r.pop_ready(|e| Completed(6).covers(e)),
            None,
            "epoch 5's notice released the item retired after it"
        );
        assert_eq!(r.stale_notices(), 1);
        assert_eq!(r.held(), 1);
        assert_eq!(r.pop_ready(|e| Completed(10).covers(e)), Some("new"));
        assert!(r.is_empty());
        assert_eq!(r.withdraw(new), None, "a drained ticket is stale");
    }

    /// Closed items come back in epoch order once their epoch completes, as
    /// `Retire` gives them; a drain stops at the first incomplete notice.
    #[test]
    fn closed_items_come_back_once_their_epoch_is_covered() {
        let mut r = RetireIndex::new();
        r.retire(Epoch(3), 'a', FutureUse::Closed);
        r.retire(Epoch(5), 'b', FutureUse::Closed);
        assert_eq!(r.pop_ready(|e| Completed(3).covers(e)), None);
        assert_eq!(r.pop_ready(|e| Completed(4).covers(e)), Some('a'));
        assert_eq!(r.pop_ready(|e| Completed(4).covers(e)), None);
        assert_eq!(r.head_epoch(), Some(Epoch(5)));
        assert_eq!(r.pop_ready(|e| Completed::ALL.covers(e)), Some('b'));
    }

    /// A declared future use holds its item past its epoch: no notice until
    /// it is discharged, and then the discharge's epoch decides. A stale
    /// ticket's discharge is ignored.
    #[test]
    fn a_declared_future_use_holds_its_item_until_discharged() {
        let mut r = RetireIndex::new();
        let carry = r.retire(Epoch(2), "carry", FutureUse::Declared);
        assert_eq!(r.notices(), 0);
        assert_eq!(r.pop_ready(|e| Completed::ALL.covers(e)), None);
        assert_eq!(r.held(), 1);
        assert_eq!(r.discharge(carry, Epoch(7)), DischargeDisposition::Queued);
        assert_eq!(r.pop_ready(|e| Completed(7).covers(e)), None);
        assert_eq!(r.pop_ready(|e| Completed(8).covers(e)), Some("carry"));
        assert_eq!(r.discharge(carry, Epoch(9)), DischargeDisposition::Stale);
        assert_eq!(r.notices(), 0, "a stale ticket's discharge queued a notice");
    }

    /// The metadata stays bounded under withdraw-and-retire churn: records
    /// at the most items held at once, notices at the retirements since the
    /// oldest incomplete epoch, each dropped when its epoch completes.
    #[test]
    fn churn_keeps_the_records_and_notices_bounded() {
        let mut r = RetireIndex::new();
        let in_flight = 3u64;
        for epoch in 0..2000u64 {
            for i in 0..4 {
                let t = r.retire(Epoch(epoch), (epoch, i), FutureUse::Closed);
                if i % 2 == 0 {
                    assert!(r.withdraw(t).is_some());
                }
            }
            let done = Completed(epoch.saturating_sub(in_flight));
            while r.pop_ready(|e| done.covers(e)).is_some() {}
            assert!(
                r.records() <= 4 * (in_flight as usize + 1) + 2,
                "{}",
                r.records()
            );
            assert!(
                r.notices() <= 4 * (in_flight as usize + 1),
                "{}",
                r.notices()
            );
        }
        assert_eq!(r.stale_notices(), 2 * (2000 - (in_flight + 1)));
    }

    #[test]
    fn duplicate_current_discharge_is_idempotent_without_notice_growth() {
        let mut r = RetireIndex::new();
        let t = r.retire(Epoch(1), "carry", FutureUse::Declared);
        let capacity = r.notices.pending.capacity();
        assert_eq!(r.discharge(t, Epoch(7)), DischargeDisposition::Queued);
        for _ in 0..1000 {
            assert_eq!(
                r.discharge(t, Epoch(7)),
                DischargeDisposition::AlreadyQueued
            );
        }
        assert_eq!(r.notices(), 1);
        assert_eq!(r.notices.pending.capacity(), capacity);
        assert_eq!(r.pop_ready(|e| Completed(7).covers(e)), None);
        assert_eq!(r.pop_ready(|e| Completed(8).covers(e)), Some("carry"));
        assert_eq!(r.pop_ready(|_| true), None);
    }

    #[test]
    fn closed_ticket_cannot_queue_a_discharge_notice() {
        let mut r = RetireIndex::new();
        let t = r.retire(Epoch(3), "closed", FutureUse::Closed);
        let before = snapshot(&r);
        for e in [0, 3, 4, 99] {
            assert_eq!(
                r.discharge(t, Epoch(e)),
                DischargeDisposition::NotDeclared { retired: Epoch(3) }
            );
            assert_eq!(snapshot(&r), before);
        }
        assert_eq!(r.notices(), 1);
        assert_eq!(r.head_epoch(), Some(Epoch(3)));
        assert_eq!(r.pop_ready(|e| Completed(4).covers(e)), Some("closed"));
        assert_eq!(r.discharge(t, Epoch(99)), DischargeDisposition::Stale);
    }

    #[test]
    fn stale_discharge_cannot_change_a_reused_records_receipt() {
        let mut r = RetireIndex::new();
        let old = r.retire(Epoch(1), "old", FutureUse::Declared);
        assert_eq!(r.withdraw(old), Some("old"));
        let new = r.retire(Epoch(2), "new", FutureUse::Declared);
        assert_eq!(old.slot, new.slot);
        assert_eq!(new.generation, old.generation + 1);
        assert_eq!(r.discharge(old, Epoch(8)), DischargeDisposition::Stale);
        assert_eq!(r.notices(), 0);
        assert_eq!(r.unqueued, 1);
        assert_eq!(r.discharge(new, Epoch(9)), DischargeDisposition::Queued);
        assert_eq!(r.pop_ready(|e| Completed(10).covers(e)), Some("new"));
    }

    #[test]
    fn incompatible_epochs_quarantine_instead_of_selecting_earliest_or_latest() {
        for incompatible in [4, 9] {
            let mut r = RetireIndex::new();
            let t = r.retire(Epoch(1), "held", FutureUse::Declared);
            let capacity = r.notices.pending.capacity();
            assert_eq!(r.discharge(t, Epoch(7)), DischargeDisposition::Queued);
            assert_eq!(
                r.discharge(t, Epoch(incompatible)),
                DischargeDisposition::Incompatible {
                    accepted: Epoch(7),
                    attempted: Epoch(incompatible),
                }
            );
            for attempted in [7, incompatible, 99] {
                assert_eq!(
                    r.discharge(t, Epoch(attempted)),
                    DischargeDisposition::Quarantined {
                        accepted: Epoch(7),
                        attempted: Epoch(attempted),
                    }
                );
            }
            assert_eq!(r.notices(), 1);
            assert_eq!(r.notices.pending.capacity(), capacity);
            assert_eq!(r.pop_ready(|_| true), None);
            assert_eq!(r.held(), 1, "no epoch substitutes for the missing contract");
            assert_eq!(r.notices(), 0);
            assert_eq!(r.stale_notices(), 0, "conflict is not a stale generation");
            assert_eq!(r.withdraw(t), Some("held"));
            assert_eq!(r.discharge(t, Epoch(99)), DischargeDisposition::Stale);
        }
    }

    #[test]
    fn declared_churn_reserves_notice_capacity_before_discharge() {
        let mut r = RetireIndex::new();
        for epoch in 0..2000u64 {
            let a = r.retire(Epoch(epoch), 1, FutureUse::Declared);
            let b = r.retire(Epoch(epoch), 2, FutureUse::Declared);
            let c = r.retire(Epoch(epoch), 3, FutureUse::Closed);
            let capacity = r.notices.pending.capacity();
            for t in [a, b] {
                assert_eq!(r.discharge(t, Epoch(epoch)), DischargeDisposition::Queued);
                assert_eq!(
                    r.discharge(t, Epoch(epoch)),
                    DischargeDisposition::AlreadyQueued
                );
            }
            assert_eq!(r.notices.pending.capacity(), capacity);
            assert_eq!(r.notices(), 3);
            assert_eq!(r.unqueued, 0);
            assert_eq!(r.withdraw(b), Some(2));
            let mut total = 0;
            while let Some(item) = r.pop_ready(|e| Completed(epoch + 1).covers(e)) {
                total += item;
            }
            assert_eq!(total, 4);
            assert_eq!(r.discharge(c, Epoch(epoch)), DischargeDisposition::Stale);
            assert_eq!(r.held(), 0);
            assert_eq!(r.notices(), 0);
            assert_eq!(r.records(), 3);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct IndexSnapshot {
        records: Vec<(Option<&'static str>, u64, u64, DischargePhase)>,
        vacant: Vec<u32>,
        notices: VecDeque<(Epoch, Ticket)>,
        parked: usize,
        unqueued: usize,
        stale: u64,
        capacities: [usize; 3],
    }

    fn snapshot(r: &RetireIndex<&'static str>) -> IndexSnapshot {
        IndexSnapshot {
            records: r
                .records
                .iter()
                .map(|record| {
                    (
                        record.item,
                        record.generation,
                        record.receipt.epoch,
                        record.receipt.phase,
                    )
                })
                .collect(),
            vacant: r.vacant.clone(),
            notices: r.notices.pending.clone(),
            parked: r.parked,
            unqueued: r.unqueued,
            stale: r.stale,
            capacities: [
                r.records.capacity(),
                r.vacant.capacity(),
                r.notices.pending.capacity(),
            ],
        }
    }

    fn assert_notice_credits<I>(r: &RetireIndex<I>) {
        let awaiting = r
            .records
            .iter()
            .filter(|record| {
                record.item.is_some() && record.receipt.phase == DischargePhase::Awaiting
            })
            .count();
        assert_eq!(r.unqueued, awaiting);
        assert!(r.notices.pending.capacity() >= r.notices() + r.unqueued);
    }

    /// Refusal keeps the original lower bound and every credit/state field.
    /// Equal and later corrections both retain the ordinary completion gate.
    #[test]
    fn before_retirement_is_retryable_without_state_or_credit_mutation() {
        for valid in [5, 9] {
            let mut r = RetireIndex::new();
            let t = r.retire(Epoch(5), "carry", FutureUse::Declared);
            let before = snapshot(&r);
            for attempted in [0, 4, 4] {
                assert_eq!(
                    r.discharge(t, Epoch(attempted)),
                    DischargeDisposition::BeforeRetirement {
                        retired: Epoch(5),
                        attempted: Epoch(attempted),
                    }
                );
                assert_eq!(snapshot(&r), before);
                assert_eq!(r.pop_ready(|_| true), None);
                assert_eq!(snapshot(&r), before);
                assert_notice_credits(&r);
            }
            assert_eq!(r.discharge(t, Epoch(valid)), DischargeDisposition::Queued);
            assert_eq!(r.unqueued, 0);
            assert_eq!(r.notices(), 1);
            assert_eq!(r.head_epoch(), Some(Epoch(valid)));
            assert_eq!(r.notices.pending.capacity(), before.capacities[2]);
            assert_eq!(r.pop_ready(|e| Completed(valid).covers(e)), None);
            assert_eq!(
                r.pop_ready(|e| Completed(valid + 1).covers(e)),
                Some("carry")
            );
            assert_eq!(r.discharge(t, Epoch(4)), DischargeDisposition::Stale);
            assert_notice_credits(&r);
        }
    }

    /// Only a first receipt can be corrected. A repeated receipt below even
    /// the original retirement epoch is incompatible, not a retryable refusal.
    #[test]
    fn earlier_repeated_receipt_quarantines_an_already_accepted_use() {
        let mut r = RetireIndex::new();
        let t = r.retire(Epoch(5), "carry", FutureUse::Declared);
        assert_eq!(r.discharge(t, Epoch(7)), DischargeDisposition::Queued);
        let capacity = r.notices.pending.capacity();
        assert_eq!(
            r.discharge(t, Epoch(4)),
            DischargeDisposition::Incompatible {
                accepted: Epoch(7),
                attempted: Epoch(4),
            }
        );
        assert_eq!(r.unqueued, 0);
        assert_eq!(r.notices(), 1);
        assert_eq!(r.notices.pending.capacity(), capacity);
        assert_eq!(r.head_epoch(), Some(Epoch(7)));
        assert_eq!(r.pop_ready(|_| true), None);
        assert_eq!(r.held(), 1);
        assert_eq!(
            r.discharge(t, Epoch(7)),
            DischargeDisposition::Quarantined {
                accepted: Epoch(7),
                attempted: Epoch(7),
            }
        );
        assert_eq!(r.withdraw(t), Some("carry"));
    }

    /// This exercises bookkeeping recovery, not a native ordering proof.
    /// Every withdrawal caller still owes the independent reuse obligations.
    #[test]
    fn withdraw_recovers_awaiting_queued_and_conflicted_bookkeeping() {
        for phase in 0..4 {
            let mut r = RetireIndex::new();
            let old = r.retire(Epoch(2), "old", FutureUse::Declared);
            if phase != 0 {
                assert_eq!(r.discharge(old, Epoch(7)), DischargeDisposition::Queued);
            }
            if phase >= 2 {
                assert_eq!(
                    r.discharge(old, Epoch(8)),
                    DischargeDisposition::Incompatible {
                        accepted: Epoch(7),
                        attempted: Epoch(8),
                    }
                );
            }
            if phase == 3 {
                assert_eq!(r.pop_ready(|_| true), None);
                assert_eq!(r.notices(), 0);
                assert_eq!(r.held(), 1);
            }
            assert_notice_credits(&r);
            assert_eq!(r.withdraw(old), Some("old"));
            assert_eq!(r.withdraw(old), None);
            assert_eq!(r.unqueued, 0);
            let new = r.retire(Epoch(9), "new", FutureUse::Declared);
            assert_eq!(new.slot, old.slot);
            assert_eq!(new.generation, old.generation + 1);
            let before = snapshot(&r);
            for attempted in [1, 7, 99] {
                assert_eq!(
                    r.discharge(old, Epoch(attempted)),
                    DischargeDisposition::Stale
                );
                assert_eq!(snapshot(&r), before);
            }
            assert_eq!(r.pop_ready(|_| true), None);
            assert_eq!(
                r.stale_notices(),
                if phase == 1 || phase == 2 { 1 } else { 0 }
            );
            assert_eq!(r.held(), 1);
            assert_eq!(r.unqueued, 1);
            assert_eq!(r.discharge(new, Epoch(9)), DischargeDisposition::Queued);
            assert_eq!(r.pop_ready(|e| Completed(10).covers(e)), Some("new"));
            assert_notice_credits(&r);
            assert_eq!(r.notices(), 0);
            assert_eq!(r.held(), 0);
            assert_eq!(r.records(), 1);
        }
    }

    #[test]
    fn mixed_notice_credits_survive_withdrawal_and_held_awaiting_batches() {
        let mut r = RetireIndex::new();
        let mut tickets = [Ticket {
            slot: 0,
            generation: 0,
        }; 64];
        for round in 0..64 {
            for (i, ticket) in tickets.iter_mut().enumerate() {
                *ticket = r.retire(Epoch(round), i, FutureUse::Declared);
                r.retire(Epoch(round), i + 64, FutureUse::Closed);
                assert_notice_credits(&r);
            }
            assert_eq!(r.unqueued, 64);
            assert_eq!(r.notices(), 64);
            let capacity = r.notices.pending.capacity();
            for (i, &t) in tickets.iter().enumerate() {
                if i % 2 == 0 {
                    assert_eq!(r.withdraw(t), Some(i));
                    assert_eq!(r.discharge(t, Epoch(round)), DischargeDisposition::Stale);
                } else {
                    assert_eq!(r.discharge(t, Epoch(round)), DischargeDisposition::Queued);
                    assert_eq!(
                        r.discharge(t, Epoch(round)),
                        DischargeDisposition::AlreadyQueued
                    );
                }
                assert_notice_credits(&r);
            }
            assert_eq!(r.unqueued, 0);
            assert_eq!(r.notices(), 96);
            assert_eq!(r.notices.pending.capacity(), capacity);
            let mut count = 0;
            while r.pop_ready(|e| Completed(round + 1).covers(e)).is_some() {
                count += 1;
                assert_notice_credits(&r);
            }
            assert_eq!(count, 96);
            assert_eq!(r.notices(), 0);
            assert_eq!(r.held(), 0);
            assert_eq!(r.records(), 128);
        }
        eprintln!(
            "retire capacity observation: records={} notices={} vacancies={}",
            r.records.capacity(),
            r.notices.pending.capacity(),
            r.vacant.capacity()
        );
    }

    /// Actual compiler/target layouts are printed, not inferred from capacity
    /// or a whole-record byte guess. The baseline shape is diagnostic only.
    #[test]
    fn compact_receipt_layout_observation() {
        use std::mem::{align_of, size_of};
        #[allow(dead_code)]
        struct BaselineRecord<I> {
            item: Option<I>,
            generation: u64,
        }
        eprintln!("retire layout target={}/{} pointer={} epoch={}/{} ticket={}/{} receipt={}/{} disposition={}/{} record_u8={}/{} baseline_record_u8={}/{} index_u8={}/{}",
            std::env::consts::ARCH, std::env::consts::OS, size_of::<usize>(),
            size_of::<Epoch>(), align_of::<Epoch>(),
            size_of::<Ticket>(), align_of::<Ticket>(),
            size_of::<DischargeReceipt>(), align_of::<DischargeReceipt>(),
            size_of::<DischargeDisposition>(), align_of::<DischargeDisposition>(),
            size_of::<Record<u8>>(), align_of::<Record<u8>>(),
            size_of::<BaselineRecord<u8>>(), align_of::<BaselineRecord<u8>>(),
            size_of::<RetireIndex<u8>>(), align_of::<RetireIndex<u8>>());
        assert_eq!(std::mem::offset_of!(DischargeReceipt, epoch), 0);
        assert_eq!(
            std::mem::offset_of!(DischargeReceipt, phase),
            size_of::<u64>()
        );
    }

    #[test]
    fn epochs_ticks_and_counts_do_not_mix() {
        assert!(!Completed::NONE.covers(Epoch::ZERO));
        assert!(Completed::ALL.covers(Epoch(u64::MAX - 1)));
        assert_eq!(Epoch(7).plus(2), Epoch(9));
        assert_eq!(Tick::ZERO.next(), Tick(1));
    }
}
