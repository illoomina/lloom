//! What an unretained command buffer cannot hold for itself.
//!
//! A retaining command buffer takes a strong reference to every object
//! encoded into it and drops them on completion. An unretained one
//! (`commandBufferWithUnretainedReferences`, `commands::cb_unretained`)
//! takes none, so every object it reads must be held by the host until it
//! completes. Pool buffers already are (`Retire` in `BufferPool`), arena and
//! weight buffers live for their owner, pipelines for the process. The objects
//! whose last host handle can drop while a command buffer still names them —
//! an encoder's fence, a buffer allocated outside every pool — are handed to an
//! [`InFlightHold`] instead of being released, and come back to Metal once the
//! epoch open at the drop has completed: the pool's own rule, through the same
//! [`Retire`] (`docs/profiles-and-memory-design.md` §4.5).
//!
//! Present only on the unretained arm: a retaining queue gets no hold, and
//! every drop is the plain release it always was.

use super::{Buffer, GpuClock};
use crate::retire::{Completed, Retire};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFence;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// An object a command buffer may still name after its last host handle
/// dropped.
pub enum Held {
    /// An encoder's `MTLFence`: updated by its own encoder, waited on by later
    /// ones, possibly in later command buffers.
    Fence(Retained<ProtocolObject<dyn MTLFence>>),
    /// A buffer no pool parks: an unpooled upload, a persistent allocation,
    /// the base of an arena or a session region.
    Buffer(Buffer),
}

// SAFETY: Metal objects are thread-safe to retain and release from any
// thread; the hold only ever releases them.
unsafe impl Send for Held {}

/// Objects held against the epoch open when they were given up, released once
/// the queue's clock shows that epoch complete.
pub struct InFlightHold {
    clock: Arc<GpuClock>,
    held: parking_lot::Mutex<Retire<Held>>,
    fences: AtomicU64,
    buffers: AtomicU64,
}

/// What a hold has taken, for a run to show the arm engaged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HoldCounts {
    /// Fences held past their last host handle, since construction.
    pub fences: u64,
    /// Buffers held past their last host handle, since construction.
    pub buffers: u64,
    /// Objects held now, their epoch not yet complete.
    pub pending: usize,
}

impl InFlightHold {
    pub fn new(clock: Arc<GpuClock>) -> Self {
        Self {
            clock,
            held: parking_lot::Mutex::new(Retire::new()),
            fences: AtomicU64::new(0),
            buffers: AtomicU64::new(0),
        }
    }

    /// Hold `item` until every command buffer that could name it has
    /// completed: the last of those is the one open now.
    ///
    /// Takes a lock the encoding thread also takes, so it must not run on a
    /// Metal completion thread (`CONTRIBUTING.md` §3.3a); every caller drops
    /// its object on the host.
    pub fn hold(&self, item: Held) {
        match &item {
            Held::Fence(_) => self.fences.fetch_add(1, Ordering::Relaxed),
            Held::Buffer(_) => self.buffers.fetch_add(1, Ordering::Relaxed),
        };
        retire_at_current(&self.held, || self.clock.current_epoch(), item);
    }

    /// Release everything whose epoch has completed. A prefix walk that stops
    /// at the first entry still outstanding; the releases happen outside the
    /// lock.
    pub fn release_completed(&self) -> usize {
        self.release_completed_up_to(self.clock.completed_count())
    }

    /// The same completed-prefix drain with a caller's earlier snapshot.
    fn release_completed_up_to(&self, completed: Completed) -> usize {
        release_completed_prefix(&self.held, completed)
    }

    /// Borrowed cleanup for Commands' off-EntryState custody. Captures the
    /// cutoff now, but takes no hold lock until the cleanup itself is dropped.
    pub(super) fn defer_release_completed(&self) -> CompletedRelease<'_, Held> {
        CompletedRelease::new(&self.held, self.clock.completed_count())
    }

    pub fn counts(&self) -> HoldCounts {
        HoldCounts {
            fences: self.fences.load(Ordering::Relaxed),
            buffers: self.buffers.load(Ordering::Relaxed),
            pending: self.held.lock().held(),
        }
    }
}

/// Serialize stamp acquisition with insertion: reading a monotone clock before
/// taking this mutex would let a delayed older stamp follow a newer queue entry.
/// The production closure is one atomic clock load; no new lock or queue scan.
fn retire_at_current<I>(
    held: &parking_lot::Mutex<Retire<I>>,
    current_epoch: impl FnOnce() -> crate::retire::Epoch,
    item: I,
) {
    let mut queue = held.lock();
    queue.retire(current_epoch(), item);
}

/// Snapshot-bearing borrowed cleanup. No Arc clone, allocation or extra
/// producer lock: the caller keeps its hold alive and drops this after its
/// outer state guard. Generic only so host controls exercise the actual loop
/// and cleanup with nonempty fake payloads, not a second implementation.
pub(super) struct CompletedRelease<'a, I> {
    held: &'a parking_lot::Mutex<Retire<I>>,
    completed: Completed,
}
impl<'a, I> CompletedRelease<'a, I> {
    pub(super) fn new(held: &'a parking_lot::Mutex<Retire<I>>, completed: Completed) -> Self {
        Self { held, completed }
    }
}
impl<I> Drop for CompletedRelease<'_, I> {
    fn drop(&mut self) {
        release_completed_prefix(self.held, self.completed);
    }
}

fn release_completed_prefix<I>(
    held: &parking_lot::Mutex<Retire<I>>,
    completed: Completed,
) -> usize {
    let mut n = 0;
    loop {
        let item = held.lock().pop_ready(|e| completed.covers(e));
        match item {
            Some(item) => {
                drop(item);
                n += 1;
            }
            None => return n,
        }
    }
}

/// Keep `buffer` alive until every command buffer encoded so far has run, when
/// there is a hold: for an owner giving up a buffer that is not a pool's (an
/// arena or region base it replaces). A no-op with no hold — the retaining kind,
/// whose command buffers hold it themselves.
pub fn hold_buffer(hold: Option<&Arc<InFlightHold>>, buffer: &Buffer) {
    if let Some(hold) = hold {
        hold.hold(Held::Buffer(buffer.clone()));
    }
}

impl std::fmt::Debug for InFlightHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InFlightHold {:?}", self.counts())
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use crate::retire::Epoch;

    #[test]
    fn retirement_stamp_is_sampled_under_the_insertion_lock() {
        let clock = GpuClock::new();
        let held = parking_lot::Mutex::new(Retire::new());
        let sample = || {
            assert!(
                held.try_lock().is_none(),
                "retirement epoch sampled before insertion lock"
            );
            clock.current_epoch()
        };
        retire_at_current(&held, sample, 10u8);
        assert_eq!(clock.commit_epoch(), Epoch(0));
        retire_at_current(&held, sample, 20u8);
        clock.mark_completed(Epoch(0));
        let mut queue = held.lock();
        assert_eq!(
            queue.pop_ready(|e| clock.completed_count().covers(e)),
            Some(10)
        );
        assert_eq!(queue.head_epoch(), Some(Epoch(1)));
        assert_eq!(queue.pop_ready(|e| clock.completed_count().covers(e)), None);
        assert_eq!(queue.held(), 1);
    }
}
