//! Engine-owned deferred IO custody, shared by direct and deferred submission.
//!
//! This private adapter is not activated in native IoQueue yet. Tests execute
//! these transitions, not another lifecycle. Native owner/capacity/callback
//! enrollment and both scheduling arms require separate integration review.
use crate::{
    io_terminal::{Error, IoCell, IoTerminal},
    retirement_mailbox::Ticket,
};

/// A metadata capability into an independently rooted, admitted owner bank.
///
/// # Safety
/// `core`/`cell` must keep the same identity and address while this capability
/// exists. Dropping the capability MUST NOT destroy the last physical resource
/// owner: the bank retains CB, destinations, sources and callback custody until
/// actual returned handoff, callback completion and ordered cleanup permit it.
/// `commit_native` performs exactly one native commit; no hidden handoff before
/// it, and no cleanup from this method. It may deliver completion synchronously.
/// Progress/error reporting must not panic, free native resources or block.
/// No foreign exception recovery is promised by the Rust unwind guard.
pub(crate) unsafe trait IoSubmissionOwner {
    fn core(&self) -> &IoTerminal;
    fn cell(&self) -> &IoCell;
    fn commit_native(&self, ticket: Ticket);
    fn request_progress(&self);
    fn report_protocol_error(&self, error: Error);
}

pub(crate) struct Rejected<O> {
    pub owner: O,
    pub error: Error,
}

/// Non-Clone, movable engine custody. Moving into AfterCommit does not submit.
/// The native resources live in the owner's rooted bank, not in this lease.
#[must_use = "a dropped deferred IO is refused, never silently submitted"]
pub(crate) struct DeferredIo<O: IoSubmissionOwner> {
    owner: O,
    ticket: Ticket,
    prepared: bool,
}

impl<O: IoSubmissionOwner> DeferredIo<O> {
    /// The bank installed the accepted record before transferring this lease.
    /// Failure returns capability custody; it must not poison a foreign cell.
    ///
    /// # Safety
    /// For a ticket that passes validation, the caller owns the unique,
    /// not-previously-transferred submission authority for that exact ticket.
    /// Rejected stale/foreign/unaccepted probes transfer nothing. Validation
    /// does NOT establish uniqueness: Ticket is Copy. A production bank must
    /// create its non-Clone capability once while installing the record and
    /// keep that capability's constructors private. Cloning an Arc to a bank
    /// is not submission authority. After success no second lease may be
    /// constructed, including after refusal, commit or Drop. A rejected call
    /// returns the untransferred authority and may be recovered explicitly.
    pub unsafe fn from_prepared(owner: O, ticket: Ticket) -> Result<Self, Rejected<O>> {
        if let Err(error) = owner.core().validate_prepared_data(owner.cell(), ticket) {
            return Err(Rejected { owner, error });
        }
        Ok(Self { owner, ticket, prepared: true })
    }

    /// Called by both immediate issue and the deferred list's commit action.
    /// A concurrent refusal and this CAS have exactly one winner.
    pub fn commit(mut self) -> Result<(), Error> {
        if let Err(error) = self.owner.cell().begin_handoff(self.ticket) {
            // Keep Prepared ownership until Drop attempts explicit refusal.
            // If refusal already won, its terminal remains authoritative;
            // Drop's rejected second claim cannot overwrite it.
            return Err(error);
        }
        self.prepared = false;
        let mut call = NativeCall { owner: &self.owner, ticket: self.ticket, returned: false };
        self.owner.commit_native(self.ticket);
        self.owner.cell().handoff_returned(self.ticket)?;
        call.returned = true;
        self.owner.request_progress();
        Ok(())
    }

    /// Explicit rejection before any native call; failure retains bank custody.
    pub fn refuse(mut self) -> Result<(), Error> {
        let result = self.owner.core().refused_before_handoff(self.owner.cell(), self.ticket);
        self.prepared = false;
        self.owner.request_progress();
        result.map(|_| ())
    }
}

impl<O: IoSubmissionOwner> Drop for DeferredIo<O> {
    fn drop(&mut self) {
        if self.prepared {
            if let Err(error) = self.owner.core().refused_before_handoff(self.owner.cell(), self.ticket) {
                // Diagnostic only: do not turn a rejected stale/duplicate claim
                // into feedback or mutate another mailbox generation.
                self.owner.report_protocol_error(error);
            }
            self.owner.request_progress();
        }
    }
}

struct NativeCall<'a, O: IoSubmissionOwner> {
    owner: &'a O,
    ticket: Ticket,
    returned: bool,
}
impl<O: IoSubmissionOwner> Drop for NativeCall<'_, O> {
    fn drop(&mut self) {
        if !self.returned {
            if let Err(error) = self.owner.core().handoff_unknown(self.owner.cell(), self.ticket) {
                self.owner.report_protocol_error(error);
            }
            self.owner.request_progress();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        completion_frontier::RuntimeStorage,
        completion_wake::Wake,
        io_terminal::{Disposition, RawStatus},
        retirement_owners::{Failure, FailureSignal},
    };
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Barrier,
    };

    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
    }
    struct Bank {
        core: Arc<IoTerminal>,
        cell: IoCell,
        failure: Arc<FailureSignal>,
        resource: parking_lot::Mutex<Option<Resource>>,
        drops: Arc<AtomicUsize>,
        commits: AtomicUsize,
        wakes: AtomicUsize,
        errors: AtomicUsize,
        callback_inside_call: AtomicBool,
        panic_after_commit: AtomicBool,
        entered: Option<Arc<Barrier>>,
        release: Option<Arc<Barrier>>,
    }
    impl Bank {
        fn new() -> Arc<Self> { Self::with_barriers(None, None) }
        fn with_barriers(entered: Option<Arc<Barrier>>, release: Option<Arc<Barrier>>) -> Arc<Self> {
            let core = Arc::new(IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap());
            let failure = Arc::new(FailureSignal::default());
            let drops = Arc::new(AtomicUsize::new(0));
            Arc::new(Self {
                cell: IoCell::new(&core, failure.clone(), Arc::new(Wake::default())).unwrap(),
                core, failure,
                resource: parking_lot::Mutex::new(Some(Resource(drops.clone()))), drops,
                commits: AtomicUsize::new(0), wakes: AtomicUsize::new(0), errors: AtomicUsize::new(0),
                callback_inside_call: AtomicBool::new(false), panic_after_commit: AtomicBool::new(false),
                entered, release,
            })
        }
        fn accept(self: &Arc<Self>, sequence: u64) -> (DeferredIo<Owner>, Ticket) {
            let ticket = self.cell.prepare_data(sequence).unwrap();
            // SAFETY: this fixture creates one lease immediately after its
            // successful prepare; no other accept/lease constructor ran.
            let lease = unsafe { DeferredIo::from_prepared(Owner { bank: self.clone() }, ticket) }
                .unwrap_or_else(|_| panic!("valid prepared owner rejected"));
            (lease, ticket)
        }
        fn finish(&self) -> Disposition {
            let ready = self.core.next().unwrap().expect("one terminal");
            let terminal = ready.terminal();
            self.cell.cleanup(terminal).unwrap();
            // This fake bank models an independently rooted cleanup owner;
            // the lease/driver never owns or drops the payload directly.
            drop(self.resource.lock().take());
            ready.finish().unwrap();
            assert!(self.core.next().unwrap().is_none());
            terminal.disposition
        }
    }
    struct Owner { bank: Arc<Bank> }
    // SAFETY: test root retains the bank/resource; only explicit finish frees
    // the payload. Native call is this instrumented boundary, not a real GPU.
    unsafe impl IoSubmissionOwner for Owner {
        fn core(&self) -> &IoTerminal { &self.bank.core }
        fn cell(&self) -> &IoCell { &self.bank.cell }
        fn commit_native(&self, ticket: Ticket) {
            self.bank.commits.fetch_add(1, Ordering::SeqCst);
            assert_eq!(self.bank.drops.load(Ordering::SeqCst), 0);
            if let Some(barrier) = &self.bank.entered { barrier.wait(); }
            if let Some(barrier) = &self.bank.release { barrier.wait(); }
            if self.bank.callback_inside_call.load(Ordering::SeqCst) {
                self.bank.core.native_terminal(&self.bank.cell, ticket, RawStatus(3)).unwrap();
                assert_eq!(self.bank.drops.load(Ordering::SeqCst), 0, "callback cannot drop native owner");
                assert_eq!(self.bank.core.successful(), 0, "callback is not retirement");
            }
            if self.bank.panic_after_commit.load(Ordering::SeqCst) { panic!("native call unwind fixture"); }
        }
        fn request_progress(&self) { self.bank.wakes.fetch_add(1, Ordering::SeqCst); }
        fn report_protocol_error(&self, _error: Error) { self.bank.errors.fetch_add(1, Ordering::SeqCst); }
    }

    #[test]
    fn deferred_move_stays_prepared_and_drop_refuses_without_commit() {
        let bank = Bank::new();
        let (lease, _) = bank.accept(0);
        let held = vec![lease];
        assert_eq!(bank.commits.load(Ordering::SeqCst), 0);
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
        assert!(!bank.core.failed());
        drop(held);
        assert!(bank.core.failed(), "accepted deferred drop must not become an unused gap");
        assert_eq!(bank.commits.load(Ordering::SeqCst), 0, "deferred drop must never commit");
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0, "drop only publishes cleanup");
        assert_eq!(bank.finish(), Disposition::AcceptedNotSubmitted);
        assert_eq!(bank.core.successful(), 0);
        assert_eq!(bank.drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn direct_and_deferred_use_the_same_commit_boundary() {
        for deferred in [false, true] {
            let bank = Bank::new();
            let (lease, ticket) = bank.accept(0);
            let lease = if deferred { vec![lease].pop().unwrap() } else { lease };
            lease.commit().unwrap();
            assert_eq!(bank.commits.load(Ordering::SeqCst), 1);
            assert_eq!(bank.drops.load(Ordering::SeqCst), 0, "returned call is not completion");
            assert!(bank.core.native_terminal(&bank.cell, ticket, RawStatus(3)).unwrap());
            assert_eq!(bank.finish(), Disposition::Native(RawStatus(3)));
            assert_eq!(bank.core.successful(), 1);
        }
    }

    #[test]
    fn synchronous_callback_before_return_keeps_resources_rooted() {
        let bank = Bank::new();
        bank.callback_inside_call.store(true, Ordering::SeqCst);
        let (lease, _) = bank.accept(0);
        lease.commit().unwrap();
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
        assert_eq!(bank.finish(), Disposition::Native(RawStatus(3)));
        assert_eq!(bank.core.successful(), 1);
    }

    #[test]
    fn refusal_winner_cannot_be_rehomed_into_a_native_commit() {
        let bank = Bank::new();
        let (lease, ticket) = bank.accept(0);
        assert!(bank.core.refused_before_handoff(&bank.cell, ticket).unwrap());
        assert!(lease.commit().is_err());
        assert_eq!(bank.commits.load(Ordering::SeqCst), 0, "refusal winner must never enter native");
        assert_eq!(bank.finish(), Disposition::AcceptedNotSubmitted);
        assert_eq!(bank.core.successful(), 0);
    }

    #[test]
    fn native_winner_cannot_be_refused_or_freed_while_call_is_active() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let bank = Bank::with_barriers(Some(entered.clone()), Some(release.clone()));
        let (lease, ticket) = bank.accept(0);
        std::thread::scope(|scope| {
            let submit = scope.spawn(|| lease.commit());
            entered.wait();
            assert!(bank.core.refused_before_handoff(&bank.cell, ticket).is_err());
            assert!(!bank.core.failed(), "losing refusal cannot poison native winner");
            assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
            release.wait();
            submit.join().unwrap().unwrap();
        });
        assert!(bank.core.native_terminal(&bank.cell, ticket, RawStatus(3)).unwrap());
        assert_eq!(bank.finish(), Disposition::Native(RawStatus(3)));
    }

    #[test]
    fn concurrent_refusal_and_commit_never_both_win() {
        let (mut committed, mut refused) = (0, 0);
        for _ in 0..64 {
            let bank = Bank::new();
            let (lease, ticket) = bank.accept(0);
            let start = Barrier::new(2);
            let native = std::thread::scope(|scope| {
                let cancel = scope.spawn(|| {
                    start.wait();
                    bank.core.refused_before_handoff(&bank.cell, ticket)
                });
                start.wait();
                let submitted = lease.commit();
                let _cancel_result = cancel.join().unwrap();
                submitted.is_ok()
            });
            assert_eq!(bank.commits.load(Ordering::SeqCst), usize::from(native), "CAS winner controls actual commit");
            assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
            if native {
                committed += 1;
                assert!(!bank.core.failed());
                bank.core.native_terminal(&bank.cell, ticket, RawStatus(3)).unwrap();
                assert_eq!(bank.finish(), Disposition::Native(RawStatus(3)));
            } else {
                refused += 1;
                assert!(bank.core.failed());
                assert_eq!(bank.finish(), Disposition::AcceptedNotSubmitted);
            }
        }
        eprintln!("IO_DEFERRED_RACE cases=64 committed={committed} refused={refused} forced_orders=separate_tests");
    }

    #[test]
    fn explicit_refusal_retains_payload_until_ordered_cleanup() {
        let bank = Bank::new();
        let (lease, _) = bank.accept(0);
        lease.refuse().unwrap();
        assert!(bank.core.failed());
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
        assert_eq!(bank.commits.load(Ordering::SeqCst), 0);
        assert_eq!(bank.finish(), Disposition::AcceptedNotSubmitted);
    }

    #[test]
    fn stale_ticket_rejection_does_not_consume_new_prepared_record() {
        let bank = Bank::new();
        let (lease, old) = bank.accept(0);
        lease.commit().unwrap();
        bank.core.native_terminal(&bank.cell, old, RawStatus(3)).unwrap();
        bank.finish();
        // New generation gets a separately rooted payload, not one from lease.
        *bank.resource.lock() = Some(Resource(bank.drops.clone()));
        bank.drops.store(0, Ordering::SeqCst);
        let (next, ticket) = bank.accept(1);
        // SAFETY: old authority was retired; this deliberately stale
        // rejection probe has no live lease for old and cannot transfer it.
        let rejected = unsafe { DeferredIo::from_prepared(Owner { bank: bank.clone() }, old) };
        assert!(rejected.is_err());
        assert!(!bank.core.failed());
        next.commit().unwrap();
        bank.core.native_terminal(&bank.cell, ticket, RawStatus(3)).unwrap();
        bank.finish();
        assert_eq!(bank.core.successful(), 2);
    }

    #[test]
    fn foreign_core_rejection_leaves_original_ticket_usable() {
        struct Foreign { wrong: Arc<IoTerminal>, bank: Arc<Bank> }
        // SAFETY: immutable identities and independently rooted payload; this
        // deliberately mismatched pair must be rejected before native handoff.
        unsafe impl IoSubmissionOwner for Foreign {
            fn core(&self) -> &IoTerminal { &self.wrong }
            fn cell(&self) -> &IoCell { &self.bank.cell }
            fn commit_native(&self, _: Ticket) { panic!("foreign native call forbidden"); }
            fn request_progress(&self) {}
            fn report_protocol_error(&self, _: Error) {}
        }
        let bank = Bank::new();
        let wrong = Arc::new(IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap());
        let ticket = bank.cell.prepare_data(0).unwrap();
        // SAFETY: no lease has yet been created for this ticket; the
        // mismatched core probe must reject without transferring authority.
        let error = match unsafe { DeferredIo::from_prepared(Foreign { wrong: wrong.clone(), bank: bank.clone() }, ticket) } {
            Ok(_) => panic!("foreign core accepted"),
            Err(rejected) => { drop(rejected.owner); rejected.error }
        };
        assert!(matches!(error, Error::ForeignCore));
        assert!(!wrong.failed() && !bank.core.failed());
        // SAFETY: the rejected probe did not transfer the sole authority.
        let original = unsafe { DeferredIo::from_prepared(Owner { bank: bank.clone() }, ticket) }
            .unwrap_or_else(|_| panic!("foreign attempt consumed original ticket"));
        original.commit().unwrap();
        bank.core.native_terminal(&bank.cell, ticket, RawStatus(3)).unwrap();
        bank.finish();
        assert_eq!(bank.core.successful(), 1);
    }

    #[test]
    fn unaccepted_gap_cannot_become_a_deferred_native_submission() {
        let bank = Bank::new();
        let ticket = bank.cell.prepare_gap(0).unwrap();
        // SAFETY: sole provisional capability; no accepted lease exists.
        // This deliberate unaccepted probe must not transfer authority.
        let rejected = unsafe { DeferredIo::from_prepared(Owner { bank: bank.clone() }, ticket) };
        assert!(matches!(rejected, Err(Rejected { error: Error::UnacceptedNativeHandoff, .. })));
        assert!(!bank.core.failed());
        assert_eq!(bank.commits.load(Ordering::SeqCst), 0);
        bank.core.abandon(&bank.cell, ticket).unwrap();
        assert_eq!(bank.finish(), Disposition::NeverAccepted);
        assert_eq!(bank.core.successful(), 0);
    }

    #[test]
    fn native_unwind_is_unknown_not_accepted_not_submitted() {
        let bank = Bank::new();
        bank.panic_after_commit.store(true, Ordering::SeqCst);
        let (lease, ticket) = bank.accept(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lease.commit()));
        assert!(result.is_err());
        assert_eq!(bank.commits.load(Ordering::SeqCst), 1);
        assert_eq!(bank.failure.get(), Some(Failure::UnknownHandoff), "native unwind must preserve unknown handoff");
        assert!(bank.core.failed());
        assert!(bank.core.refused_before_handoff(&bank.cell, ticket).is_err());
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0, "unknown native custody stays rooted");
        assert_eq!(bank.core.successful(), 0);
    }

    #[test]
    fn callback_then_native_unwind_cannot_detach_or_publish_success() {
        let bank = Bank::new();
        bank.callback_inside_call.store(true, Ordering::SeqCst);
        bank.panic_after_commit.store(true, Ordering::SeqCst);
        let (lease, _) = bank.accept(0);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lease.commit())).is_err());
        let ready = bank.core.next().unwrap().expect("callback published terminal");
        assert!(bank.cell.cleanup(ready.terminal()).is_err(), "unknown return cannot authorize detach");
        assert_eq!(bank.drops.load(Ordering::SeqCst), 0);
        assert_eq!(bank.core.successful(), 0);
        // Keep ready unfinished: active retirement credit cannot be recycled.
        drop(ready);
    }
}
