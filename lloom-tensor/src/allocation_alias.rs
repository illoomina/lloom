//! Reusable external-alias/park handshake for allocation eligibility.
//!
//! This is the production protocol core, not native terminality. Its adapter
//! must keep state and parked node in an independently rooted host inbox. Raw
//! native retains and cold same-raw identity installation are separate duties.
//! Node storage is the existing pool node; no per-checkout Box is introduced.
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

const ACTIVE: usize = 0;
const PARKED: usize = 1;
const CLAIMING: usize = 2;
const IDLE: usize = 3;
const PHASE_BITS: usize = 2;
const PHASE_MASK: usize = 3;

struct FinalRelease {
    generation: usize,
}

pub(crate) struct AliasPins<N> {
    external: AtomicUsize,
    primary: AtomicBool,
    // Generation and phase are ONE atomic claim; a stale final-alias notifier
    // cannot steal a pointer parked by the next reuse generation.
    control: AtomicUsize,
    parked: AtomicPtr<N>,
}
impl<N> AliasPins<N> {
    /// One external Buffer pin at birth. All pre-adoption clones share it.
    pub(crate) fn new() -> Self {
        Self {
            external: AtomicUsize::new(1),
            primary: AtomicBool::new(false),
            control: AtomicUsize::new(ACTIVE),
            parked: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
    /// Adapter requires an existing external pin OR an active primary handle.
    /// This is NOT the cold raw-wrapper lookup/installation operation.
    pub(crate) fn clone_external(&self) {
        let previous = self.external.fetch_add(1, Ordering::Relaxed);
        // Matches catastrophic Arc count-overflow policy, not quota refusal.
        if previous >= isize::MAX as usize {
            std::process::abort();
        }
    }
    /// Convert first adopter's external pin into the unique internal baseline.
    /// Secondary adoption keeps its pin and does not create another owner.
    pub(crate) fn claim_primary(&self) -> bool {
        if self
            .primary
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        assert_ne!(self.external.fetch_sub(1, Ordering::AcqRel), 0);
        true
    }
    pub(crate) fn has_external(&self) -> bool {
        self.external.load(Ordering::Acquire) != 0
    }
    pub(crate) fn generation(&self) -> usize {
        self.control.load(Ordering::Acquire) >> PHASE_BITS
    }
    /// A free node begins its next checkout. Overflow refuses reuse; the caller
    /// keeps ownership, never wraps a generation back onto an old notifier.
    pub(crate) fn begin_checkout(&self) -> bool {
        if self.has_external() {
            return false;
        }
        let old = self.control.load(Ordering::Acquire);
        if old & PHASE_MASK != IDLE {
            return false;
        }
        let Some(next) = (old >> PHASE_BITS)
            .checked_add(1)
            .filter(|next| *next <= usize::MAX >> PHASE_BITS)
        else {
            return false;
        };
        self.control
            .compare_exchange(
                old,
                next << PHASE_BITS | ACTIVE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
    /// Preserve the ordinary no-external-alias epoch FIFO path. Caller already
    /// owns the last primary handle's node; no new Buffer borrow can arise from
    /// that handle. A false result takes the alias-park route instead.
    pub(crate) fn finish_without_alias(&self, generation: usize) -> bool {
        !self.has_external()
            && self
                .control
                .compare_exchange(
                    generation << PHASE_BITS | ACTIVE,
                    generation << PHASE_BITS | IDLE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
    }
    /// Drop one external pin. Capture generation BEFORE publishing zero; another
    /// host may then return/reuse this node before the notification continues.
    pub(crate) fn release_external(&self) -> Option<NonNull<N>> {
        self.begin_release()
            .and_then(|notice| self.finish_release(notice))
    }
    /// Nonallocating split solely to expose the real production suspension
    /// boundary to deterministic controls; callers still consume exactly one pin.
    fn begin_release(&self) -> Option<FinalRelease> {
        let generation = self.generation();
        let previous = self.external.fetch_sub(1, Ordering::AcqRel);
        assert_ne!(previous, 0, "release owns an external pin");
        (previous == 1).then_some(FinalRelease { generation })
    }
    fn finish_release(&self, notice: FinalRelease) -> Option<NonNull<N>> {
        self.take_ready(notice.generation)
    }
    /// # Safety
    /// Transfers an exclusively owned node to this state until returned by park
    /// or release_external. No outstanding primary handle may dereference it.
    /// State/node remain independently host-rooted if the pool disappears.
    pub(crate) unsafe fn park(
        &self,
        generation: usize,
        node: NonNull<N>,
    ) -> Result<Option<NonNull<N>>, NonNull<N>> {
        let active = generation << PHASE_BITS | ACTIVE;
        if self.control.load(Ordering::Acquire) != active {
            return Err(node);
        }
        if self
            .parked
            .compare_exchange(
                std::ptr::null_mut(),
                node.as_ptr(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Err(node);
        }
        // Only the unique node owner transitions ACTIVE; final alias notifiers
        // may claim PARKED only after this release publishes the pointer.
        // Acquires a prior final-release no-op RMW even when its phase value
        // stayed ACTIVE. Thus either this side observes zero, or that notifier
        // observes our PARKED publication and claims the node. Release-only
        // store/CAS would leave a cross-atomic lost-notification execution.
        self.control
            .compare_exchange(
                active,
                generation << PHASE_BITS | PARKED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .expect("exclusive primary node owns ACTIVE transition");
        if !self.has_external() {
            Ok(self.take_ready(generation))
        } else {
            Ok(None)
        }
    }
    fn take_ready(&self, generation: usize) -> Option<NonNull<N>> {
        // Publish the count decrement even when no pointer is parked yet.
        // The park CAS acquires this RMW if ordered after it. A failed CAS
        // alone would be only a read and would NOT supply this release edge.
        let observed = self.control.fetch_or(0, Ordering::AcqRel);
        if observed != generation << PHASE_BITS | PARKED {
            return None;
        }
        if self
            .control
            .compare_exchange(
                generation << PHASE_BITS | PARKED,
                generation << PHASE_BITS | CLAIMING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return None;
        }
        let node = NonNull::new(self.parked.swap(std::ptr::null_mut(), Ordering::AcqRel));
        // Complete ALL state writes before the ready pointer escapes. No
        // publisher touches it after inbox publication. New reuse advances tag.
        self.control
            .store(generation << PHASE_BITS | IDLE, Ordering::Release);
        node
    }
    pub(crate) fn is_parked(&self) -> bool {
        self.control.load(Ordering::Acquire) & PHASE_MASK == PARKED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Fixed stack metadata only. No tensor/native/service/thread operations.
    #[test]
    fn alias_release_and_park_both_orders_return_node_once() {
        for release_first in [false, true] {
            let pins = AliasPins::new();
            pins.clone_external();
            assert!(pins.claim_primary());
            let mut node = 7usize;
            let node = NonNull::from(&mut node);
            if release_first {
                assert!(pins.release_external().is_none());
                assert_eq!(unsafe { pins.park(0, node) }.unwrap(), Some(node));
            } else {
                assert!(unsafe { pins.park(0, node) }.unwrap().is_none());
                assert_eq!(pins.release_external(), Some(node));
            }
            assert!(!pins.is_parked());
            assert!(!pins.has_external());
        }
    }
    #[test]
    fn secondary_adoption_and_pending_clone_do_not_create_reusable_owner() {
        let pins = AliasPins::new();
        pins.clone_external();
        assert!(pins.claim_primary());
        assert!(!pins.claim_primary());
        let mut node = 9usize;
        let node = NonNull::from(&mut node);
        assert!(unsafe { pins.park(0, node) }.unwrap().is_none());
        pins.clone_external();
        assert!(pins.release_external().is_none());
        assert!(pins.is_parked());
        assert_eq!(pins.release_external(), Some(node));
        assert!(!pins.is_parked());
    }
    #[test]
    fn stale_final_alias_notifier_cannot_claim_reparked_node() {
        let pins = AliasPins::new();
        pins.clone_external();
        assert!(pins.claim_primary());
        let mut node = 11usize;
        let node = NonNull::from(&mut node);
        // Actual final-release steps split at the adversarial scheduling point:
        // the release publishes zero, then park wins ready-node ownership.
        let stale_notice = pins.begin_release().expect("last external alias");
        assert_eq!(unsafe { pins.park(0, node) }.unwrap(), Some(node));
        assert!(pins.begin_checkout());
        pins.clone_external();
        assert!(unsafe { pins.park(1, node) }.unwrap().is_none());
        // Resuming the prior release's actual production claim must do nothing.
        assert!(pins.finish_release(stale_notice).is_none());
        assert!(pins.is_parked());
        assert!(pins.has_external());
        assert_eq!(pins.release_external(), Some(node));
    }
    #[test]
    fn ordinary_fifo_path_and_generation_exhaustion_preserve_ownership() {
        let pins = AliasPins::<usize>::new();
        assert!(pins.claim_primary());
        assert!(pins.finish_without_alias(0));
        assert!(pins.begin_checkout());
        assert_eq!(pins.generation(), 1);
        assert!(!pins.finish_without_alias(0));
        assert!(pins.finish_without_alias(1));
        // Metadata-only boundary: no corresponding allocation or iteration.
        pins.control.store(usize::MAX, Ordering::Release); // last generation IDLE
        assert!(!pins.begin_checkout());
        assert_eq!(pins.control.load(Ordering::Acquire), usize::MAX);
    }
}
