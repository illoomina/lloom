//! Unapplied private child of io_terminal: native bank transition implementation.
//! Experimental source candidate: private bank/service transitions only.
//! Production enrollment and dependent-consumer wiring remain unimplemented.
use super::*;
use crate::retirement_owners::Failure;
use crate::retirement_service::ActiveRetirement;
use std::sync::atomic::AtomicUsize;

pub(super) struct Slot<C, A> {
    cell: IoCell<(C, A)>,
    // Independent metadata, never an unqualified native-owner borrow.
    native_key: AtomicUsize,
}

pub(super) struct Bank<C, A> {
    core: IoTerminal,
    slots: Vec<Slot<C, A>>,
    issued: AtomicU64,
    // Published only AFTER contiguous physical retirement, not on metadata pop.
    retired: AtomicU64,
    pending: AtomicBool,
    // Retained protocol categories, separate from data-generation health.
    diagnostics: AtomicU64,
    active: ActiveRetirement<C, (Terminal, A)>,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    metadata_charge: parking_lot::Mutex<Option<crate::memory_admission::PhysicalCharge>>,
}

pub(super) struct Prepared<C, A> {
    bank: BankHandle<C, A>,
    index: usize,
    ticket: Ticket,
    native_key: usize,
}

pub(super) enum PrepareError<C, A> {
    // Explicit NotIssued: no token, accepted cell or sequence escaped.
    NotIssued { error: Error, captures: C, anchors: A },
    // The installed record remains bank-owned; caller must not free its owners.
    RootedProtocolFailure,
    // Concurrent generation failure: installed owners were explicitly refused.
    RootedRefused,
}

#[derive(Debug)]
pub(super) enum CallbackError {
    NoPendingGeneration,
    WrongNativeIdentity,
    Terminal(Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparePoint { HeadSnapshot, BeforeClaim, Installed }

impl<C, A> Bank<C, A> {
    fn fail(&self) {
        self.core.failed.store(true, Ordering::Release);
        self.failure.publish(Failure::AdapterFailed, || self.wake.notify());
    }

    /// Private, already-admitted backing only. Concurrent callers use the SAME
    /// candidate head cell's existing mailbox CAS, not a new producer mutex.
    /// The private caller supplies real native identity for this exact C.
    fn try_prepare(handle: &BankHandle<C, A>, captures: C, anchors: A, native_key: usize)
        -> Result<Prepared<C, A>, PrepareError<C, A>>
    {
        Self::try_prepare_with(handle, captures, anchors, native_key, |_| {})
    }

    // Same production transitions; empty monomorphized callback in real callers.
    // Tests pause exact observation/claim boundaries without a second algorithm.
    fn try_prepare_with(handle: &BankHandle<C, A>, captures: C, anchors: A, native_key: usize,
                        mut at: impl FnMut(PreparePoint))
        -> Result<Prepared<C, A>, PrepareError<C, A>>
    {
        let this: &Self = handle;
        let sequence = this.issued.load(Ordering::Acquire);
        at(PreparePoint::HeadSnapshot);
        let retired = this.retired.load(Ordering::Acquire);
        let refuse = |error, captures, anchors| PrepareError::NotIssued { error, captures, anchors };
        if this.core.failed() || this.failure.get().is_some() {
            return Err(refuse(Error::Mailbox(MailboxError::Failed(Failure::AdapterFailed)), captures, anchors));
        }
        let Some(next) = sequence.checked_add(1) else {
            return Err(refuse(Error::Frontier(FrontierError::SequenceExhausted), captures, anchors));
        };
        let Some(distance) = sequence.checked_sub(retired) else {
            return Err(refuse(Error::Mailbox(MailboxError::Busy), captures, anchors));
        };
        if distance >= this.slots.len() as u64 {
            return Err(refuse(Error::Frontier(FrontierError::BeyondWindow), captures, anchors));
        }
        let index = (sequence % this.slots.len() as u64) as usize;
        let slot = &this.slots[index];
        at(PreparePoint::BeforeClaim);
        let ticket = match slot.cell.prepare_owned(sequence, true, (captures, anchors)) {
            Ok(ticket) => ticket,
            Err(OwnedRejected { error, payload: (captures, anchors) }) =>
                return Err(refuse(error, captures, anchors)),
        };
        // A producer can pause across an entire window reuse before preparing
        // its stale candidate. Failed issued CAS is therefore ordinary NotIssued,
        // not by itself queue corruption. This claim has no callback, token or
        // handoff authority outside this stack; return the intact owners.
        if this.issued.compare_exchange(sequence, next, Ordering::AcqRel, Ordering::Acquire).is_err() {
            match slot.cell.mailbox.cancel_unissued(ticket) {
                Ok(detached) => {
                    let (captures, anchors) = detached.owner.payload;
                    return Err(refuse(Error::Mailbox(MailboxError::Busy), captures, anchors));
                }
                Err(_) => {
                    this.fail();
                    // A genuine protocol mismatch still retains mailbox custody.
                    return Err(PrepareError::RootedProtocolFailure);
                }
            }
        }
        slot.native_key.store(native_key, Ordering::Release);
        at(PreparePoint::Installed);
        if this.core.failed() || this.failure.get().is_some() {
            if this.refuse(index, ticket).is_err() {
                this.fail();
                return Err(PrepareError::RootedProtocolFailure);
            }
            return Err(PrepareError::RootedRefused);
        }
        Ok(Prepared { bank: handle.clone(), index, ticket, native_key })
    }

    /// Called only by the cold per-cell native block with the actual valid CB
    /// argument. The raw argument is not obtained through a mailbox payload.
    fn callback(&self, index: usize, native_key: usize, status: RawStatus)
        -> Result<(), CallbackError>
    {
        let Some(slot) = self.slots.get(index) else {
            return Err(CallbackError::WrongNativeIdentity);
        };
        let ticket = slot.cell.mailbox.pending_ticket()
            .ok_or(CallbackError::NoPendingGeneration)?;
        if slot.native_key.load(Ordering::Acquire) != native_key {
            return Err(CallbackError::WrongNativeIdentity);
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.core.publish_native_feedback(&slot.cell, ticket, status)
                .map_err(CallbackError::Terminal)?;
            self.request_progress();
            Ok(())
        }));
        match outcome {
            Ok(result) => result,
            Err(payload) => {
                // Existing M3 boundary pattern: retain the original panic in
                // exact generation when possible; never Drop it on the native
                // callback when an already-detached/stale record rejects it.
                if let Err(payload) = slot.cell.mailbox.retain_boundary_panic(ticket, payload) {
                    let _ = Box::leak(payload);
                }
                self.core.failed.store(true, Ordering::Release);
                self.failure.publish(Failure::CallbackPanic, || self.wake.notify());
                Err(CallbackError::Terminal(Error::ObserverPanicked))
            }
        }
    }

    fn note_callback_error(&self, error: &CallbackError) {
        let flag = match error {
            CallbackError::NoPendingGeneration => 1,
            CallbackError::WrongNativeIdentity => 2,
            CallbackError::Terminal(_) => 4,
        };
        self.diagnostics.fetch_or(flag, Ordering::Release);
        self.wake.notify();
    }

    fn note_protocol_error(&self, error: &Error) {
        // Diagnostics survive even for expected same-ticket second refusal;
        // rejected foreign/stale attempts do not poison a new data generation.
        let flag = match error {
            Error::Mailbox(MailboxError::Stale) => 8,
            Error::Mailbox(MailboxError::WrongPhase) => 16,
            Error::ForeignCore => 32,
            _ => 64,
        };
        self.diagnostics.fetch_or(flag, Ordering::Release);
        self.wake.notify();
    }

    fn refuse(&self, index: usize, ticket: Ticket) -> Result<(), Error> {
        // Factor uses original phase CAS and sticky failure, but NOT publish's
        // order Mutex. Ingestion reads that same cancelled mailbox afterward.
        let result = self.core.record_refusal(&self.slots[index].cell, ticket);
        self.request_progress();
        result.map(|_| ())
    }

    fn request_progress(&self) {
        self.pending.fetch_or(true, Ordering::SeqCst);
        self.assist();
    }

    /// Holder assistance follows Commands; publication ordering is stronger:
    /// publisher RMWs chain multiple mailbox publications, and the clearing
    /// RMW acquires the consumed wave. Plain overwrite stores do not prove that
    /// every overwritten publisher's mailbox writes are visible to this holder.
    /// One pass covers at most the admitted window; repeated fresh waves are
    /// explicitly NOT a wait-free aggregate bound.
    fn assist(&self) { self.assist_with(|| {}); }

    // Test seam pauses only after the holder has observed pending and acquired
    // exclusion, immediately before consuming that wave. Production is empty.
    fn assist_with(&self, mut before_consume: impl FnMut()) {
        loop {
            std::sync::atomic::fence(Ordering::SeqCst);
            if !self.pending.load(Ordering::SeqCst) { return; }
            let Some(mut order) = self.core.order.try_lock() else { return; };
            before_consume();
            self.pending.swap(false, Ordering::SeqCst);
            let result = self.ingest_locked(&mut order);
            drop(order);
            // Future physical/service holders MUST call this assistance after
            // dropping their guard; otherwise a lost try-lock can strand work.
            if result.is_err() { self.fail(); return; }
            // Existing Wake may unpark: not an atomics-only/syscall-free claim.
            // Notification counts and E2E cost remain measurement obligations.
            self.wake.notify();
        }
    }

    fn ingest_locked(&self, order: &mut IoOrder) -> Result<(), Error> {
        let end = self.issued.load(Ordering::Acquire);
        while order.ingested_next < end {
            let sequence = order.ingested_next;
            let slot = &self.slots[(sequence % self.slots.len() as u64) as usize];
            let terminal = if let Some(ticket) = slot.cell.mailbox.feedback_ticket() {
                let mut terminal = None;
                slot.cell.mailbox.with_callbacks(ticket, |owner, raw| {
                    terminal = Some(Terminal {
                        sequence: owner.meta.sequence, ticket,
                        disposition: Disposition::Native(*raw),
                    });
                }, || self.wake.notify()).map_err(Error::Mailbox)?;
                terminal.expect("metadata callback returned")
            } else if let Some(ticket) = slot.cell.mailbox.cleanup_ticket() {
                // This sequence has not been ingested. In the private bank only
                // cancellation can reach CleanupReady before ingestion; native
                // cleanup is published below AFTER its frontier insertion.
                slot.cell.mailbox.with_cancelled_metadata(ticket, |owner| Terminal {
                    sequence: owner.meta.sequence, ticket,
                    disposition: if owner.meta.accepted {
                        Disposition::AcceptedNotSubmitted
                    } else { Disposition::NeverAccepted },
                }, || self.wake.notify()).map_err(Error::Mailbox)?
            } else { break; };
            if terminal.sequence != sequence {
                self.core.failed.store(true, Ordering::Release);
                return Err(Error::Frontier(FrontierError::UnexpectedSequence));
            }
            order.frontier.publish(sequence, terminal).map_err(|r| Error::Frontier(r.reason))?;
            order.ingested_next = sequence.checked_add(1)
                .ok_or(Error::Frontier(FrontierError::SequenceExhausted))?;
            // Physical slot-return/cleanup publication is done by the host
            // service, off this order guard. It is not a data-ready prerequisite.

        }
        self.core.validate_pending(order)
    }
    fn begin_retire(&self) -> bool { self.begin_retire_with(|| {}) }

    // One implementation; production closure is empty, controlled tests pause
    // this exact holder path without replacing the transition/service algorithm.
    fn begin_retire_with(&self, on_locked: impl FnOnce()) -> bool {
        if !self.active.is_empty() { return false; }
        let Some(mut order) = self.core.order.try_lock() else { return false; };
        on_locked();
        let terminal = if let Some(terminal) = order.retiring {
            Some(terminal)
        } else {
            if self.core.validate_pending(&mut order).is_err() {
                drop(order); self.fail(); self.assist(); return false;
            }
            match order.frontier.pop_ready_or_release_drainer() {
                Ok(Some((_, terminal))) => {
                    order.retiring = Some(terminal);
                    Some(terminal)
                }
                Ok(None) | Err(FrontierError::NoDrainer) => None,
                Err(_) => {
                    drop(order); self.fail(); self.assist(); return false;
                }
            }
        };
        drop(order);
        self.assist();
        let Some(terminal) = terminal else { return false; };
        let slot = &self.slots[(terminal.sequence % self.slots.len() as u64) as usize];
        if matches!(terminal.disposition, Disposition::Native(_))
            && slot.cell.mailbox.cleanup_ticket().is_none()
        {
            // Actual IO Complete/Error feedback was ingested and its metadata
            // callback borrow ended. SLOT_RETURNED is this adapter's physical
            // cleanup eligibility flag, NOT bank admission credit or CB Drop.
            if slot.cell.mailbox.slot_returned(terminal.ticket).is_err()
                || slot.cell.mailbox.publish_cleanup(terminal.ticket).is_err()
            {
                self.fail(); return false;
            }
        }
        let detached = match slot.cell.mailbox.try_detach_cleanup_with_policy(
            terminal.ticket, CleanupPolicy::ReturnedHandoff,
        ) {
            Ok(detached) => detached,
            // Complete can precede native call-return. UNKNOWN stays quarantined.
            // The bank-owned retiring terminal is retained across this return.
            Err(MailboxError::WrongPhase) => return false,
            Err(_) => { self.fail(); return false; }
        };
        let OwnedRecord { meta, payload: (captures, anchors) } = detached.owner;
        if meta.sequence != terminal.sequence {
            std::mem::forget((captures, anchors, detached.feedback));
            self.fail(); return false;
        }
        // Same trusted non-user-code detach->stage interval as existing Slab
        // CleanupInbox. The actual owners are rooted before arbitrary Drop.
        if let Err((_, captures, anchors)) = self.active.stage(captures, (terminal, anchors)) {
            std::mem::forget((captures, anchors, detached.feedback));
            self.fail(); return false;
        }
        // RawStatus is Copy metadata, not a callback/resource owner.
        true
    }

    fn drop_captures(&self) {
        self.active.drop_captures().expect("service owns staged IO captures");
    }

    fn finish_retire(&self) {
        let (terminal, anchors) = self.active.finish().expect("IO capture Drop returned");
        // A contains trusted native/accounting anchors, not arbitrary callbacks.
        drop(anchors);
        // Service metadata-only exclusion: never acquired blocking by producer
        // or native callback. No arbitrary destructor runs while held.
        let mut order = self.core.order.lock();
        if order.retiring.map(|t| t.sequence) != Some(terminal.sequence)
            || order.frontier.finish(terminal.sequence).is_err()
        {
            drop(order); self.fail(); self.assist(); return;
        }
        order.retiring = None;
        self.retired.store(terminal.sequence + 1, Ordering::Release);
        drop(order);
        self.assist();
    }

}

// Independent host service custody. This is not a second worker or registry.
// HostRetirementService::pass elects the existing exclusive passing CAS before
// entry.progress: it is the sole authority spanning begin->stage->Drop->finish.
// The bank does not add a second retirement mutex or concurrent service caller.
// C/A safety is constrained to the concrete native IO payload binding: C may
// run capture/native-CB destruction, A is trusted physical/cohort custody.
unsafe impl<C: Send + Sync + 'static, A: Send + Sync + 'static>
    crate::retirement_service::CleanupInbox for Bank<C, A>
{
    fn begin_retire(&self) -> bool { Bank::begin_retire(self) }
    fn drop_captures(&self) { Bank::drop_captures(self) }
    fn finish_retire(&self) { Bank::finish_retire(self) }
    fn can_unregister(&self) -> bool {
        self.issued.load(Ordering::Acquire) == self.retired.load(Ordering::Acquire)
            && self.active.is_empty()
            && self.slots.iter().all(|s| s.cell.mailbox.metadata_reusable())
    }
    fn worker_final_metadata_drop(&self) -> bool { true }
    fn take_final_metadata_charge(&self) -> Option<crate::memory_admission::PhysicalCharge> {
        self.metadata_charge.lock().take()
    }
    fn fail(&self, _reason: Failure) { Bank::fail(self) }
}

impl<C, A> crate::retirement_registry::MetadataWake for Bank<C, A> {
    fn metadata_wake(&self) -> &Arc<Wake> { &self.wake }
}

// Only counted HostMetadataHandle is permitted outside registered service root;
// a raw public Arc/Weak path would invalidate worker-final root census. Prepared
// and cold native blocks will use this handle, not export their Arc<Bank>.
type BankHandle<C, A> = crate::retirement_registry::HostMetadataHandle<Bank<C, A>>;

struct ColdParts<C, A> {
    core: IoTerminal,
    slots: Vec<Slot<C, A>>,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    // LAST: all backing bytes are freed before their charge on cold rejection.
    charge: crate::memory_admission::PhysicalCharge,
}

enum ColdError<C, A> {
    // Parts never became registered or externally visible.
    Parts(ColdParts<C, A>),
    // Preserve independently held metadata+charge until caller handles failure.
    Registration(
        BankHandle<C, A>,
        // AFTER handle: refund only after failed-registration bank allocation frees.
        crate::memory_admission::PhysicalCharge,
        crate::retirement_service::ServiceError,
    ),
}

impl<C: Send + Sync + 'static, A: Send + Sync + 'static> Bank<C, A> {
    /// Supplied parts have already been allocated under real cold admission.
    /// This validates their actual capacities/charge; it does not invent a byte
    /// ceiling, quota, producer bound or IO native-command-buffer capacity.
    /// Arc headers and allocator-internal rounding follow existing registry's
    /// explicitly excluded managed-layout convention; cold native blocks and
    /// cohort Vecs require their OWN accounted owners before publication.
    /// # Safety
    /// A is trusted native/cohort custody, with no arbitrary user destructor.
    /// C/A reflect the supplied native adapter's real owners; erased/raw fake
    /// anchors do not discharge this obligation. No Weak/raw bank acquisition.
    /// The private constructor caller supplies every cell with this exact
    /// failure Arc and the service wake Arc. Matching core_id alone is not proof
    /// of those identities. Only HostRetirementService drives CleanupInbox.
    unsafe fn from_admitted(parts: ColdParts<C, A>, service: &crate::retirement_service::HostRetirementService)
        -> Result<BankHandle<C, A>, ColdError<C, A>>
    {
        let valid = {
            let order = parts.core.order.lock();
            let capacity = u64::try_from(parts.slots.len()).ok();
            let geometry = capacity.is_some_and(|capacity| {
                capacity > 0 && order.frontier.next_sequence() == 0
                    && matches!(order.frontier.inspect_pending(0), Ok(PendingInspection::Hole))
                    && matches!(order.frontier.inspect_pending(capacity), Ok(PendingInspection::WindowEnd))
            });
            let slots_layout = std::alloc::Layout::array::<Slot<C, A>>(parts.slots.capacity()).ok();
            let managed = slots_layout.and_then(|layout| {
                layout.size().checked_add(std::mem::size_of::<Self>())?
                    .checked_add(order.frontier.allocation_layout().size())?
                    .checked_add(crate::retirement_service::HostRetirementService::requested_registration_bytes())
            });
            geometry && !parts.core.failed() && parts.core.successful() == 0
                && parts.failure.get().is_none()
                && order.ingested_next == 0 && order.retiring.is_none()
                && parts.slots.iter().all(|s| s.cell.core_id == parts.core.id
                    && s.cell.mailbox.metadata_reusable())
                && managed.is_some_and(|bytes| parts.charge.bytes() >= bytes)
        };
        if !valid { return Err(ColdError::Parts(parts)); }
        let ColdParts { core, slots, failure, wake, charge } = parts;
        let bank = Arc::new(Self {
            core, slots, issued: AtomicU64::new(0), retired: AtomicU64::new(0),
            pending: AtomicBool::new(false), diagnostics: AtomicU64::new(0), active: ActiveRetirement::new(),
            failure, wake, metadata_charge: parking_lot::Mutex::new(Some(charge)),
        });
        let inbox: Arc<dyn crate::retirement_service::CleanupInbox> = bank.clone();
        let result = service.register_owned(&inbox);
        drop(inbox);
        let handle = crate::retirement_registry::HostMetadataHandle::new(bank);
        match result {
            Ok(()) => Ok(handle),
            Err(error) => {
                // Nothing escaped and register_owned did not publish this root.
                // Keep actual inline charge beyond final failed-bank deallocation.
                let charge = handle.metadata_charge.lock().take().expect("private cold charge");
                Err(ColdError::Registration(handle, charge, error))
            },
        }
    }
}

#[cfg(feature = "metal")]
#[path = "native_callback_binding.rs"]
mod native;
#[cfg(test)]
#[path = "native_bank_tests.rs"]
mod tests;
