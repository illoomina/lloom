//! Source-only proposed inline physical reuse occupancy, not a native lease.
//! Lives in the allocation-local PhysicalRoot's admitted Layout payload.
//! No HeaderRoot/pool/namespace/hold owner or Buffer-with-self-pin lives here.
//! The adapter must bind every ticket inseparably to the correct opaque root.
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const BITS: usize = 3;
const MASK: usize = (1 << BITS) - 1;
const FREE: usize = 0;
const PREPARING: usize = 1;
const LIVE: usize = 2;
const FREEZING: usize = 3;
const WAITING: usize = 4;
const CLEANING: usize = 5;
const ABORTING: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadyError {
    Busy,
    Failed,
    GenerationExhausted,
    WrongGeneration,
    Phase,
    UnrepresentableEpoch,
}
/// A generation, NOT allocation identity or an alias/terminality proof.
#[derive(Clone, Copy)]
pub(crate) struct UseGeneration(usize);
/// Metadata only. Actual rich A must inseparably retain the correct physical
/// root plus this receipt; no root identity is encoded in these integers.
pub(crate) struct PreparedUse {
    old: usize,
    next: usize,
}
pub(crate) struct PhysicalReady {
    control: AtomicUsize,
    cutoff: AtomicU64,
    never_submitted: AtomicBool,
    failed: AtomicBool,
}
pub(crate) struct ReadyReservation<'a> {
    state: &'a PhysicalReady,
    old: usize,
    next: usize,
    armed: bool,
}
impl PhysicalReady {
    pub(crate) fn new() -> Self {
        Self {
            control: AtomicUsize::new(FREE),
            cutoff: AtomicU64::new(0),
            never_submitted: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        }
    }
    /// Physical service final-custody predicate only. Actual alias/native
    /// closure and opaque root binding remain the adapter's obligations.
    pub(crate) fn is_free(&self) -> bool {
        !self.failed.load(Ordering::Acquire) && self.control.load(Ordering::Acquire) & MASK == FREE
    }
    /// Readonly quarantine observation, not FREE/reuse/native permission.
    pub(crate) fn check_failed(&self) -> Result<(), ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            Err(ReadyError::Failed)
        } else {
            Ok(())
        }
    }
    pub(crate) fn quarantine(&self) {
        self.failed.store(true, Ordering::Release);
    }
    /// Exclusive occupancy only. The physical backing remains owned while the
    /// caller performs unlocked admitted header growth; competitors see Busy.
    /// No success counters, native unpoison, epoch or generation change yet.
    pub(crate) fn reserve(&self) -> Result<ReadyReservation<'_>, ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        let old = self.control.load(Ordering::Acquire);
        if old & MASK != FREE {
            return Err(ReadyError::Busy);
        }
        let generation = old >> BITS;
        let next = generation
            .checked_add(1)
            .filter(|next| *next <= usize::MAX >> BITS)
            .ok_or(ReadyError::GenerationExhausted)?;
        self.control
            .compare_exchange(old, old | PREPARING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ReadyError::Busy)?;
        Ok(ReadyReservation {
            state: self,
            old,
            next,
            armed: true,
        })
    }
    /// After initialized rich C/A is independently rooted, commit immediately
    /// before primary-header publish-last; no fallible operation in between.
    pub(crate) fn commit_prepared(
        &self,
        receipt: &PreparedUse,
    ) -> Result<UseGeneration, ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        self.control
            .compare_exchange(
                receipt.old | PREPARING,
                receipt.next << BITS | LIVE,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| ReadyError::Phase)?;
        Ok(UseGeneration(receipt.next))
    }
    /// Actual rich header DETACHING authority proves its primary was never
    /// published while physical occupancy stayed PREPARING. No future/native
    /// borrow escaped, so no last-use cutoff is sampled. Still retain exclusive
    /// occupancy through arbitrary C cleanup, including its panic interval.
    pub(crate) fn claim_abandoned(&self, receipt: &PreparedUse) -> Result<bool, ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        match self.control.compare_exchange(
            receipt.old | PREPARING,
            receipt.old | ABORTING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(true),
            Err(actual) if actual == (receipt.old | ABORTING) => Ok(true),
            Err(actual)
                if actual == (receipt.next << BITS | LIVE)
                    || actual == (receipt.next << BITS | WAITING)
                    || actual == (receipt.next << BITS | CLEANING) =>
            {
                Ok(false)
            }
            Err(_) => Err(ReadyError::Phase),
        }
    }
    /// Sole rich A, after successful arbitrary C cleanup. Aborted initialized
    /// checkout restores original generation and cutoff; live cleanup preserves
    /// its completed cutoff and returns the committed generation's eligibility.
    pub(crate) fn finish_prepared_cleanup(&self, receipt: PreparedUse) -> Result<(), ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        let actual = self.control.load(Ordering::Acquire);
        let restored = if actual == (receipt.old | ABORTING) {
            receipt.old
        } else if actual == (receipt.next << BITS | CLEANING) {
            receipt.next << BITS
        } else {
            return Err(ReadyError::Phase);
        };
        self.control
            .compare_exchange(actual, restored, Ordering::Release, Ordering::Relaxed)
            .map_err(|_| ReadyError::Phase)?;
        Ok(())
    }
    /// Poll only AFTER actual primary header DETACHING authority excludes new
    /// primary borrowers, and the allocation-local future-use census is closed.
    /// external_closed must come from actual rich/view/native/IO holders, not
    /// old header Weak counts or an allocation's raw native retain count.
    /// This metadata method does NOT establish those adapter obligations.
    ///
    /// current is sampled after that closure; completed is the same clock's
    /// completed-count witness. A constructor/native caller must separately
    /// establish queue/clock identity and any untracked raw escape boundary.
    pub(crate) fn poll_closed_use(
        &self,
        generation: UseGeneration,
        external_closed: bool,
        current: u64,
        completed: u64,
    ) -> Result<bool, ReadyError> {
        self.poll_closed_cutoff(generation, external_closed, Some(current), completed)
    }
    /// Exact last submitted creator cutoff, encoded epoch+1; zero denotes no
    /// submitted creator use. Never samples the OPEN epoch or requires a future cut.
    /// # Safety
    /// Actual primary DETACHING excludes new issuers. Every native/IO/foreign/
    /// derived/raw-capability use retains the external pin through proven own
    /// terminality AND emission closure. Before pin release, each creator use
    /// observes its exact submitted epoch; cancelled/unknown are not submissions.
    /// submitted_count is the monotone historical maximum acquired AFTER census
    /// closure, not absence of a caller. Zero requires exhaustive no-creator-use
    /// proof under that same closed capability contract. No raw retain-count proof.
    pub(crate) unsafe fn poll_closed_exact_use(
        &self,
        generation: UseGeneration,
        external_closed: bool,
        submitted_count: u64,
        completed: u64,
    ) -> Result<bool, ReadyError> {
        let cutoff = submitted_count.checked_sub(1);
        self.poll_closed_cutoff(generation, external_closed, cutoff, completed)
    }
    fn poll_closed_cutoff(
        &self,
        generation: UseGeneration,
        external_closed: bool,
        cutoff: Option<u64>,
        completed: u64,
    ) -> Result<bool, ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        if !external_closed {
            return Ok(false);
        }
        let base = generation.0 << BITS;
        let observed = self.control.load(Ordering::Acquire);
        if observed >> BITS != generation.0 {
            return Err(ReadyError::WrongGeneration);
        }
        match observed & MASK {
            LIVE => {
                // MAX cannot have an epoch+1 completed-count representation.
                // Keep the owned LIVE backing charged; do not wrap or reuse.
                if cutoff == Some(u64::MAX) {
                    return Err(ReadyError::UnrepresentableEpoch);
                }
                self.control
                    .compare_exchange(
                        base | LIVE,
                        base | FREEZING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .map_err(|_| ReadyError::Phase)?;
                self.cutoff.store(cutoff.unwrap_or(0), Ordering::Relaxed);
                self.never_submitted
                    .store(cutoff.is_none(), Ordering::Relaxed);
                self.control.store(base | WAITING, Ordering::Release);
            }
            WAITING => {}
            CLEANING => return Ok(true),
            FREEZING => return Ok(false),
            _ => return Err(ReadyError::Phase),
        }
        // WAITING release publishes exactly one safe last-use cutoff. Later
        // polls do not restamp it; delayed publication cannot feed Retire FIFO.
        let cutoff = self.cutoff.load(Ordering::Relaxed);
        if !self.never_submitted.load(Ordering::Relaxed) && completed <= cutoff {
            return Ok(false);
        }
        self.control
            .compare_exchange(
                base | WAITING,
                base | CLEANING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| ReadyError::Phase)?;
        Ok(true)
    }
    /// Sole service cleanup diagnostic, not an eligibility or submission witness.
    /// # Safety
    /// Caller holds the matching rich DETACHING C/A authority; no concurrent
    /// producer may mutate this physical generation. ABORTING was never published.
    pub(crate) unsafe fn cleanup_cutoff(
        &self,
        generation: UseGeneration,
    ) -> Result<Option<u64>, ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        let control = self.control.load(Ordering::Acquire);
        if control == (generation.0 << BITS | CLEANING) {
            return Ok((!self.never_submitted.load(Ordering::Relaxed))
                .then(|| self.cutoff.load(Ordering::Relaxed)));
        }
        if control & MASK == ABORTING && (control >> BITS).checked_add(1) == Some(generation.0) {
            return Ok(None);
        }
        Err(ReadyError::Phase)
    }
    /// Called by the rich header's allocation-local A only AFTER arbitrary C
    /// cleanup returned successfully. Its root+generation binding and sole
    /// recycle authority are adapter obligations, not proved by this integer.
    /// C panic leaves A retained in HeaderRoot.active and keeps CLEANING: no
    /// physical eligibility is published and no sidecar/native retry occurs.
    pub(crate) fn publish_after_cleanup(
        &self,
        generation: UseGeneration,
    ) -> Result<(), ReadyError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        let base = generation.0 << BITS;
        self.control
            .compare_exchange(
                base | CLEANING,
                base | FREE,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| ReadyError::Phase)?;
        Ok(())
    }
}
impl ReadyReservation<'_> {
    /// Transfer original occupancy to an allocation-local rich A receipt,
    /// before payload initialization; actual preclaimed header adapter performs
    /// the transfer/root initialization without fallible/arbitrary operations.
    /// After this handoff, dropping this guard does NOT return physical FREE.
    /// Initialized abandonment queues rich cleanup and retains A on C panic.
    /// Metadata alone cannot bind the root or arrange that custody transfer.
    pub(crate) fn handoff(mut self) -> PreparedUse {
        self.armed = false;
        PreparedUse {
            old: self.old,
            next: self.next,
        }
    }

    /// For source-side transaction binding only; native/header ownership must
    /// remain unpublished until the caller finishes all fallible admission and
    /// revalidates pool failure/origin/eligibility under its original guard.
    pub(crate) fn next_generation(&self) -> UseGeneration {
        UseGeneration(self.next)
    }
    /// No fallible/arbitrary operation may occur between this commit and actual
    /// primary-header publish-last. Actual EmptyHeader/PreparedHeader now
    /// supplies reserve/initialize/publish ordering, but no typed physical
    /// receipt, native alias census or production caller is enabled here.
    pub(crate) fn commit(mut self) -> Result<UseGeneration, ReadyError> {
        if self.state.failed.load(Ordering::Acquire) {
            return Err(ReadyError::Failed);
        }
        self.state
            .control
            .compare_exchange(
                self.old | PREPARING,
                self.next << BITS | LIVE,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| ReadyError::Phase)?;
        self.armed = false;
        Ok(UseGeneration(self.next))
    }
    pub(crate) fn rollback(mut self) -> Result<(), ReadyError> {
        self.state
            .control
            .compare_exchange(
                self.old | PREPARING,
                self.old,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| ReadyError::Phase)?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for ReadyReservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Only restore the exact exclusively claimed old state. Never
            // overwrite another generation, rollback global counters, mutate
            // old cutoff, grow a Vec or destroy an owner on unwind.
            if self
                .state
                .control
                .compare_exchange(
                    self.old | PREPARING,
                    self.old,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                self.state.failed.store(true, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ready_rollback_restores_original_generation_and_cutoff() {
        let state = PhysicalReady::new();
        state.cutoff.store(17, Ordering::Relaxed);
        {
            let claim = state.reserve().unwrap();
            assert_eq!(claim.next_generation().0, 1);
            assert_eq!(state.control.load(Ordering::Acquire), PREPARING);
            assert!(matches!(state.reserve(), Err(ReadyError::Busy)));
        }
        assert_eq!(state.control.load(Ordering::Acquire), FREE);
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 17);
        state.reserve().unwrap().rollback().unwrap();
        assert_eq!(state.control.load(Ordering::Acquire), FREE);
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 17);
    }
    #[test]
    fn ready_freezes_once_and_requires_successful_cleanup_before_reuse() {
        let state = PhysicalReady::new();
        let use_generation = state.reserve().unwrap().commit().unwrap();
        assert!(!state.poll_closed_use(use_generation, false, 4, 99).unwrap());
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | LIVE);
        assert!(!state.poll_closed_use(use_generation, true, 4, 4).unwrap());
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 4);
        assert!(state.poll_closed_use(use_generation, true, 999, 5).unwrap());
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 4);
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | CLEANING);
        assert!(matches!(state.reserve(), Err(ReadyError::Busy)));
        state.publish_after_cleanup(use_generation).unwrap();
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | FREE);
        assert_eq!(state.reserve().unwrap().next_generation().0, 2);
    }
    #[test]
    fn ready_initialized_abandonment_retains_occupancy_until_anchor_cleanup() {
        let state = PhysicalReady::new();
        state.cutoff.store(23, Ordering::Relaxed);
        let receipt = state.reserve().unwrap().handoff();
        assert_eq!(state.control.load(Ordering::Acquire), PREPARING);
        assert!(state.claim_abandoned(&receipt).unwrap());
        assert!(state.claim_abandoned(&receipt).unwrap());
        assert_eq!(state.control.load(Ordering::Acquire), ABORTING);
        assert!(matches!(state.reserve(), Err(ReadyError::Busy)));
        state.finish_prepared_cleanup(receipt).unwrap();
        assert_eq!(state.control.load(Ordering::Acquire), FREE);
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 23);
        let receipt = state.reserve().unwrap().handoff();
        let use_generation = state.commit_prepared(&receipt).unwrap();
        assert!(!state.claim_abandoned(&receipt).unwrap());
        assert!(state.poll_closed_use(use_generation, true, 3, 4).unwrap());
        state.finish_prepared_cleanup(receipt).unwrap();
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | FREE);
        assert_eq!(state.cutoff.load(Ordering::Relaxed), 3);
    }
    #[test]
    fn ready_exhaustion_wrong_generation_and_epoch_overflow_retain_custody() {
        let state = PhysicalReady::new();
        let last_free = usize::MAX & !MASK;
        state.control.store(last_free, Ordering::Release);
        assert!(matches!(
            state.reserve(),
            Err(ReadyError::GenerationExhausted)
        ));
        assert_eq!(state.control.load(Ordering::Acquire), last_free);
        state.control.store(1 << BITS | LIVE, Ordering::Release);
        assert_eq!(
            state.poll_closed_use(UseGeneration(2), true, 3, 4),
            Err(ReadyError::WrongGeneration)
        );
        assert_eq!(
            state.poll_closed_use(UseGeneration(1), true, u64::MAX, u64::MAX),
            Err(ReadyError::UnrepresentableEpoch)
        );
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | LIVE);
        state.failed.store(true, Ordering::Release);
        assert_eq!(
            state.poll_closed_use(UseGeneration(1), true, 3, 4),
            Err(ReadyError::Failed)
        );
        assert!(matches!(state.reserve(), Err(ReadyError::Failed)));
        assert_eq!(state.control.load(Ordering::Acquire), 1 << BITS | LIVE);
    }
}
