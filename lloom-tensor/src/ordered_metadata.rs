//! Persistent generation metadata for a queue-associated command marker.
//!
//! The owner mailbox's live pin excludes reset; the owner payload can already
//! have been detached. Successor custody is metadata-only and is consumed once
//! by the elected completion drainer, never by native arrival order.
use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

/// `native_key` before a commit names its native command buffer.
const NO_KEY: usize = 0;

const EMPTY: u8 = 0;
const WRITING: u8 = 1;
const PUBLISHED: u8 = 2;
const TAKEN: u8 = 3;

pub(crate) struct OrderedMetadata<P> {
    successor_phase: AtomicU8,
    successor: UnsafeCell<Option<P>>,
    native_ready: AtomicBool,
    ordered_ready: AtomicBool,
    callbacks_finished: AtomicBool,
    wrapper_handles: AtomicUsize,
    /// The native object this generation's commit handed to the queue (its
    /// address), so a queue-wide completion handler can find the record.
    native_key: AtomicUsize,
    command_lifecycle: crate::command_lifecycle::CommandLifecycle,
}
// SAFETY: publication owns WRITING, extraction owns TAKEN after winning the
// PUBLISHED CAS. A shared reference to P is exposed only by `peek_successor`,
// whose caller excludes extraction (see there) and which requires P: Sync.
// Reset requires exclusive generation custody from the containing registry,
// not just a native ticket.
unsafe impl<P: Send> Send for OrderedMetadata<P> {}
unsafe impl<P: Send> Sync for OrderedMetadata<P> {}
impl<P> OrderedMetadata<P> {
    pub(crate) fn new() -> Self {
        Self {
            successor_phase: AtomicU8::new(EMPTY),
            successor: UnsafeCell::new(None),
            native_ready: AtomicBool::new(false),
            ordered_ready: AtomicBool::new(false),
            callbacks_finished: AtomicBool::new(false),
            wrapper_handles: AtomicUsize::new(0),
            native_key: AtomicUsize::new(NO_KEY),
            command_lifecycle: crate::command_lifecycle::CommandLifecycle::new(),
        }
    }
    /// Atomic-only prerequisite for final host metadata disposal. Together
    /// with the containing Slab's exclusive Arc census, EMPTY/TAKEN prove no
    /// successor value remains and zero wrappers prove no release is pending.
    pub(crate) fn quiescent(&self) -> bool {
        matches!(self.successor_phase.load(Ordering::Acquire), EMPTY | TAKEN)
            && self.wrapper_handles.load(Ordering::Acquire) == 0
    }
    /// Caller owns a newly prepared mailbox generation, before returning its
    /// first Registration. Previous generation has no pins or initial pinners.
    /// Every linked predecessor stays pinned until its successor was taken.
    pub(crate) unsafe fn reset_generation(&self) {
        assert_eq!(
            self.wrapper_handles.load(Ordering::Acquire),
            0,
            "live wrapper at metadata reuse"
        );
        let phase = self.successor_phase.load(Ordering::Acquire);
        assert!(
            matches!(phase, EMPTY | TAKEN),
            "unconsumed successor at metadata reuse"
        );
        // SAFETY: required generation exclusion; consumed/unused link is None.
        debug_assert!(unsafe { (&*self.successor.get()).is_none() });
        // SAFETY: the same exclusive generation/pin boundary protects lifecycle tickets.
        unsafe {
            self.command_lifecycle.reset_generation();
        }
        self.native_key.store(NO_KEY, Ordering::Relaxed);
        self.native_ready.store(false, Ordering::Relaxed);
        self.ordered_ready.store(false, Ordering::Relaxed);
        self.callbacks_finished.store(false, Ordering::Relaxed);
        self.successor_phase.store(EMPTY, Ordering::Release);
    }
    /// One commit-order producer publishes the next committed record. Rejection
    /// returns the pin; no ownership can disappear on duplicate publication.
    pub(crate) fn append(&self, next: P) -> Result<(), P> {
        if self
            .successor_phase
            .compare_exchange(EMPTY, WRITING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(next);
        }
        // SAFETY: WRITING excludes extraction and all other publishers.
        unsafe {
            *self.successor.get() = Some(next);
        }
        self.successor_phase.store(PUBLISHED, Ordering::Release);
        Ok(())
    }
    /// The published successor, left in place.
    ///
    /// # Safety
    /// The caller holds, for as long as it uses the reference, the one lock
    /// under which every [`Self::take_successor`] of this chain runs, so the
    /// value cannot be taken from under it; publication is complete once
    /// PUBLISHED is observed, and a published value is never rewritten.
    pub(crate) unsafe fn peek_successor(&self) -> Option<&P>
    where
        P: Sync,
    {
        if self.successor_phase.load(Ordering::Acquire) != PUBLISHED {
            return None;
        }
        // SAFETY: PUBLISHED, and extraction is excluded by the caller.
        unsafe { (*self.successor.get()).as_ref() }
    }
    /// Name the native object this generation's commit hands to the queue.
    pub(crate) fn set_native_key(&self, key: usize) {
        debug_assert_ne!(key, NO_KEY);
        self.native_key.store(key, Ordering::Release);
    }
    pub(crate) fn native_key(&self) -> usize {
        self.native_key.load(Ordering::Acquire)
    }
    pub(crate) fn take_successor(&self) -> Option<P> {
        self.successor_phase
            .compare_exchange(PUBLISHED, TAKEN, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        // SAFETY: exactly one consumer won PUBLISHED. An independent metadata
        // pin held by the caller excludes reset through this take.
        unsafe { (*self.successor.get()).take() }
    }
    // Each counted consumer holds a metadata pin. Associated wrappers and
    // host-anchor permits use this counter on distinct registry records. Internal
    // link/prefix pins are metadata-only and never prolong payload ownership.
    pub(crate) fn add_wrapper_handle(&self) -> Result<(), ()> {
        self.wrapper_handles
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map(|_| ())
            .map_err(|_| ())
    }
    pub(crate) fn release_wrapper_handle(&self) -> bool {
        let previous = self.wrapper_handles.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "one release per associated wrapper");
        previous == 1
    }
    pub(crate) fn publish_native_ready(&self) {
        self.native_ready.store(true, Ordering::Release);
    }
    pub(crate) fn native_ready(&self) -> bool {
        self.native_ready.load(Ordering::Acquire)
    }
    pub(crate) fn publish_ordered_ready(&self) {
        self.ordered_ready.store(true, Ordering::Release);
    }
    pub(crate) fn ordered_ready(&self) -> bool {
        self.ordered_ready.load(Ordering::Acquire)
    }
    pub(crate) fn publish_callbacks_finished(&self) {
        self.callbacks_finished.store(true, Ordering::Release);
    }
    pub(crate) fn command_lifecycle(&self) -> &crate::command_lifecycle::CommandLifecycle {
        &self.command_lifecycle
    }
    pub(crate) fn callbacks_finished(&self) -> bool {
        self.callbacks_finished.load(Ordering::Acquire)
    }
}
