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

    #[test]
    fn epochs_ticks_and_counts_do_not_mix() {
        assert!(!Completed::NONE.covers(Epoch::ZERO));
        assert!(Completed::ALL.covers(Epoch(u64::MAX - 1)));
        assert_eq!(Epoch(7).plus(2), Epoch(9));
        assert_eq!(Tick::ZERO.next(), Tick(1));
    }
}
