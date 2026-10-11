//! Recording exclusion shared by native wrappers, active encoder leases and
//! prepared commits. Associated M3 embeds this in reusable pinned metadata;
//! M4 embeds it in Submission. No per-associated-CB allocation is required.
//!
//! Adapters retain the owning generation for the full Ticket lifetime and wake
//! queue/service waiters AFTER releasing a phase. Tickets carry a serial, not
//! only a phase: a stale guard cannot abandon a later encoder in the same cell.
use std::sync::atomic::{AtomicU64, Ordering};

const RECORDING: u64 = 0;
const EDITING: u64 = 1;
const ENCODING: u64 = 2;
const ENDING: u64 = 3;
const PREPARED: u64 = 4;
const ENTERED: u64 = 5;
const SUBMITTED: u64 = 6;
const ABANDONED: u64 = 7;
const ENCODING_EDIT: u64 = 8;
const ABANDONED_EDIT: u64 = 9;
const CANCELLED_EDIT: u64 = 10;
const MASK: u64 = 15;
const STEP: u64 = MASK + 1;

/// Inspectable recording refusal; none of these states proves GPU completion.
#[derive(thiserror::Error, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleError {
    #[error("recording operation currently busy")]
    Busy,
    #[error("command buffer submission already entered")]
    Submitted,
    #[error("recording admission abandoned after exceptional exit")]
    Abandoned,
    #[error("recording generation exhausted")]
    Exhausted,
    #[error("recording phase owner is no longer active")]
    StaleOwner,
}

/// Not Clone/Copy: one native transaction or shared encoder lease owns this.
/// Its adapter retains the corresponding wrapper/generation pin independently.
pub(crate) struct Ticket {
    owner: usize,
    word: u64,
    active: bool,
}
pub(crate) struct CommandLifecycle(AtomicU64);
impl CommandLifecycle {
    pub(crate) const fn new() -> Self {
        Self(AtomicU64::new(RECORDING))
    }
    fn identity(&self) -> usize {
        self as *const Self as usize
    }
    fn enter(&self, phase: u64) -> Result<Ticket, LifecycleError> {
        let mut previous = self.0.load(Ordering::Acquire);
        loop {
            match previous & MASK {
                RECORDING => {}
                ENTERED | SUBMITTED => return Err(LifecycleError::Submitted),
                ABANDONED | ABANDONED_EDIT => return Err(LifecycleError::Abandoned),
                _ => return Err(LifecycleError::Busy),
            }
            let next = previous
                .checked_add(STEP)
                .ok_or(LifecycleError::Exhausted)?
                | phase;
            match self
                .0
                .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return Ok(Ticket {
                        owner: self.identity(),
                        word: next,
                        active: true,
                    })
                }
                Err(observed) => previous = observed,
            }
        }
    }
    fn transition(
        &self,
        ticket: &mut Ticket,
        from: u64,
        to: u64,
        terminal: bool,
    ) -> Result<(), LifecycleError> {
        if !ticket.active || ticket.owner != self.identity() || ticket.word & MASK != from {
            return Err(LifecycleError::StaleOwner);
        }
        let next = (ticket.word & !MASK) | to;
        self.0
            .compare_exchange(ticket.word, next, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| LifecycleError::StaleOwner)?;
        ticket.word = next;
        ticket.active = !terminal;
        Ok(())
    }
    /// Callback registration may borrow an open encoder's phase. The encoder
    /// keeps its original serial; end returns Busy while this borrow is live.
    pub(crate) fn begin_edit(&self) -> Result<Ticket, LifecycleError> {
        let previous = self.0.load(Ordering::Acquire);
        if previous & MASK != ENCODING {
            return self.enter(EDITING);
        }
        let next = (previous & !MASK) | ENCODING_EDIT;
        self.0
            .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| LifecycleError::Busy)?;
        Ok(Ticket {
            owner: self.identity(),
            word: next,
            active: true,
        })
    }
    /// Event boundaries require no open native encoder. Callback registration
    /// deliberately uses begin_edit instead, preserving its borrowed subphase.
    pub(crate) fn begin_boundary(&self) -> Result<Ticket, LifecycleError> {
        self.enter(EDITING)
    }
    pub(crate) fn finish_edit(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        if ticket.word & MASK != ENCODING_EDIT {
            return self.transition(ticket, EDITING, RECORDING, true);
        }
        if !ticket.active || ticket.owner != self.identity() {
            return Err(LifecycleError::StaleOwner);
        }
        let mut previous = ticket.word;
        loop {
            let to = match previous & MASK {
                ENCODING_EDIT => ENCODING,
                ABANDONED_EDIT => ABANDONED,
                CANCELLED_EDIT => RECORDING,
                _ => return Err(LifecycleError::StaleOwner),
            };
            if previous & !MASK != ticket.word & !MASK {
                return Err(LifecycleError::StaleOwner);
            }
            let next = (previous & !MASK) | to;
            match self
                .0
                .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    ticket.word = next;
                    ticket.active = false;
                    return Ok(());
                }
                Err(observed) => previous = observed,
            }
        }
    }
    pub(crate) fn begin_encoder(&self) -> Result<Ticket, LifecycleError> {
        self.enter(ENCODING)
    }
    /// Native encoder creation failed before a native object escaped.
    pub(crate) fn cancel_encoder_creation(
        &self,
        ticket: &mut Ticket,
    ) -> Result<(), LifecycleError> {
        if !ticket.active || ticket.owner != self.identity() || ticket.word & MASK != ENCODING {
            return Err(LifecycleError::StaleOwner);
        }
        let mut previous = ticket.word;
        loop {
            if previous & !MASK != ticket.word & !MASK {
                return Err(LifecycleError::StaleOwner);
            }
            let to = match previous & MASK {
                ENCODING => RECORDING,
                ENCODING_EDIT => CANCELLED_EDIT,
                _ => return Err(LifecycleError::StaleOwner),
            };
            let next = (previous & !MASK) | to;
            match self
                .0
                .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    ticket.word = next;
                    ticket.active = false;
                    return Ok(());
                }
                Err(observed) => previous = observed,
            }
        }
    }
    pub(crate) fn begin_end_encoder(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        if !ticket.active || ticket.owner != self.identity() || ticket.word & MASK != ENCODING {
            return Err(LifecycleError::StaleOwner);
        }
        let next = (ticket.word & !MASK) | ENDING;
        match self
            .0
            .compare_exchange(ticket.word, next, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                ticket.word = next;
                Ok(())
            }
            Err(observed) if observed == ((ticket.word & !MASK) | ENCODING_EDIT) => {
                Err(LifecycleError::Busy)
            }
            Err(_) => Err(LifecycleError::StaleOwner),
        }
    }
    /// Publish the local ended flag only after owning ENDING. Refusal keeps the
    /// original encoder ticket retryable; unwind closes it without native proof.
    pub(crate) fn end_encoder_with<R>(
        &self,
        ticket: &mut Ticket,
        ended: &mut bool,
        close: impl FnOnce() -> R,
    ) -> Result<R, LifecycleError> {
        self.begin_end_encoder(ticket)?;
        let mut ending = Ending {
            gate: self,
            ticket,
            returned: false,
        };
        *ended = true;
        let result = close();
        self.finish_end_encoder(ending.ticket)?;
        ending.returned = true;
        Ok(result)
    }
    /// Called AFTER native endEncoding returned; commit remains excluded before.
    pub(crate) fn finish_end_encoder(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        self.transition(ticket, ENDING, RECORDING, true)
    }
    pub(crate) fn prepare(&self) -> Result<Ticket, LifecycleError> {
        self.enter(PREPARED)
    }
    pub(crate) fn cancel_prepared(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        self.transition(ticket, PREPARED, RECORDING, true)
    }
    pub(crate) fn entered_native(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        self.transition(ticket, PREPARED, ENTERED, false)
    }
    pub(crate) fn handoff_returned(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        self.transition(ticket, ENTERED, SUBMITTED, true)
    }
    /// Exceptional exit closes only the exact phase/serial actually owned by
    /// this token. Refusal never overwrites a newer owner, even in the same phase.
    /// Admission closure is NOT native terminality or permission to free owners.
    pub(crate) fn abandon(&self, ticket: &mut Ticket) -> Result<(), LifecycleError> {
        let phase = ticket.word & MASK;
        if !ticket.active
            || ticket.owner != self.identity()
            || !matches!(
                phase,
                EDITING | ENCODING | ENDING | PREPARED | ENTERED | ENCODING_EDIT
            )
        {
            return Err(LifecycleError::StaleOwner);
        }
        let mut previous = ticket.word;
        loop {
            if previous & !MASK != ticket.word & !MASK {
                return Err(LifecycleError::StaleOwner);
            }
            let borrowed_parent = phase == ENCODING && previous & MASK == ENCODING_EDIT;
            let cancelled_borrow = phase == ENCODING_EDIT
                && matches!(previous & MASK, ABANDONED_EDIT | CANCELLED_EDIT);
            if previous & MASK != phase && !borrowed_parent && !cancelled_borrow {
                return Err(LifecycleError::StaleOwner);
            }
            // An active registration keeps its borrow while its encoder is
            // abandoned. Finishing that borrow closes instead of resurrecting.
            let to = if borrowed_parent {
                ABANDONED_EDIT
            } else {
                ABANDONED
            };
            let next = (previous & !MASK) | to;
            match self
                .0
                .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    ticket.word = next;
                    ticket.active = false;
                    return Ok(());
                }
                Err(observed) => previous = observed,
            }
        }
    }
    /// Registry alone resets after all generation pins/initial pinners are
    /// excluded. Adapters hold such a pin through every Ticket's final use.
    pub(crate) unsafe fn reset_generation(&self) {
        self.0.store(RECORDING, Ordering::Release);
    }
}

struct Ending<'a> {
    gate: &'a CommandLifecycle,
    ticket: &'a mut Ticket,
    returned: bool,
}
impl Drop for Ending<'_> {
    fn drop(&mut self) {
        if !self.returned {
            let _ = self.gate.abandon(self.ticket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_refusal_preserves_open_encoder_and_callback_registration() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        assert!(matches!(gate.begin_boundary(), Err(LifecycleError::Busy)));
        let mut registration = gate.begin_edit().unwrap();
        gate.finish_edit(&mut registration).unwrap();
        gate.begin_end_encoder(&mut encoder).unwrap();
        gate.finish_end_encoder(&mut encoder).unwrap();
        let mut boundary = gate.begin_boundary().unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.finish_edit(&mut boundary).unwrap();
        let mut commit = gate.prepare().unwrap();
        gate.entered_native(&mut commit).unwrap();
        gate.handoff_returned(&mut commit).unwrap();
        assert!(matches!(
            gate.begin_boundary(),
            Err(LifecycleError::Submitted)
        ));
    }

    #[test]
    fn prepared_excludes_edits_and_unused_reservation_can_retry() {
        let gate = CommandLifecycle::new();
        let mut prepared = gate.prepare().unwrap();
        assert!(matches!(gate.begin_edit(), Err(LifecycleError::Busy)));
        assert!(matches!(gate.begin_encoder(), Err(LifecycleError::Busy)));
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.cancel_prepared(&mut prepared).unwrap();
        let mut editing = gate.begin_edit().unwrap();
        gate.finish_edit(&mut editing).unwrap();
        let mut next = gate.prepare().unwrap();
        gate.entered_native(&mut next).unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Submitted)));
        gate.handoff_returned(&mut next).unwrap();
        assert!(matches!(
            gate.begin_encoder(),
            Err(LifecycleError::Submitted)
        ));
    }

    #[test]
    fn encoder_excludes_commit_through_native_end_return() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.begin_end_encoder(&mut encoder).unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.finish_end_encoder(&mut encoder).unwrap();
        let mut prepared = gate.prepare().unwrap();
        gate.cancel_prepared(&mut prepared).unwrap();
    }

    #[test]
    fn stale_phase_owner_cannot_abandon_successor() {
        let gate = CommandLifecycle::new();
        let mut previous = gate.begin_encoder().unwrap();
        // Deliberately preserve a stale token only inside this module. Public
        // adapters cannot clone a Ticket; this pins the defensive serial check.
        let mut stale = Ticket {
            owner: previous.owner,
            word: previous.word,
            active: true,
        };
        gate.begin_end_encoder(&mut previous).unwrap();
        gate.finish_end_encoder(&mut previous).unwrap();
        let mut successor = gate.begin_encoder().unwrap();
        assert_eq!(gate.abandon(&mut stale), Err(LifecycleError::StaleOwner));
        gate.begin_end_encoder(&mut successor).unwrap();
        gate.finish_end_encoder(&mut successor).unwrap();
        let mut prepared = gate.prepare().unwrap();
        gate.cancel_prepared(&mut prepared).unwrap();
    }

    #[test]
    fn unknown_entry_closes_admission_without_becoming_submitted() {
        let gate = CommandLifecycle::new();
        let mut prepared = gate.prepare().unwrap();
        gate.entered_native(&mut prepared).unwrap();
        gate.abandon(&mut prepared).unwrap();
        assert!(matches!(gate.begin_edit(), Err(LifecycleError::Abandoned)));
        assert!(matches!(gate.prepare(), Err(LifecycleError::Abandoned)));
        assert_eq!(
            gate.handoff_returned(&mut prepared),
            Err(LifecycleError::StaleOwner)
        );
    }
    #[test]
    fn registration_borrow_preserves_original_encoder_ticket_after_busy_end() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let original = encoder.word;
        let mut edit = gate.begin_edit().unwrap();
        assert_eq!(
            gate.begin_end_encoder(&mut encoder),
            Err(LifecycleError::Busy)
        );
        assert_eq!(encoder.word, original);
        assert!(encoder.active);
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.finish_edit(&mut edit).unwrap();
        gate.begin_end_encoder(&mut encoder).unwrap();
        gate.finish_end_encoder(&mut encoder).unwrap();
        let mut prepared = gate.prepare().unwrap();
        gate.cancel_prepared(&mut prepared).unwrap();
    }

    #[test]
    fn end_refusal_preserves_flag_and_unknown_end_exit_closes_it() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let mut ended = false;
        let mut edit = gate.begin_edit().unwrap();
        let refused = gate.end_encoder_with(&mut encoder, &mut ended, || -> () {
            panic!("Busy entered native close")
        });
        assert_eq!(refused, Err(LifecycleError::Busy));
        assert!(!ended, "refused end consumed encoder");
        gate.finish_edit(&mut edit).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gate.end_encoder_with(&mut encoder, &mut ended, || -> () {
                panic!("native-close fixture")
            })
        }));
        assert!(panic.is_err());
        assert!(ended, "unknown native close remained retryable");
        assert!(matches!(gate.prepare(), Err(LifecycleError::Abandoned)));
        assert_eq!(
            gate.end_encoder_with(&mut encoder, &mut ended, || ()),
            Err(LifecycleError::StaleOwner)
        );
    }

    #[test]
    fn successful_end_publishes_flag_and_releases_recording() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let mut ended = false;
        assert_eq!(gate.end_encoder_with(&mut encoder, &mut ended, || 7), Ok(7));
        assert!(ended);
        let mut prepared = gate.prepare().unwrap();
        gate.cancel_prepared(&mut prepared).unwrap();
    }

    #[test]
    fn abandoned_encoder_borrow_cannot_resurrect_recording() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let mut edit = gate.begin_edit().unwrap();
        gate.abandon(&mut encoder).unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Abandoned)));
        gate.finish_edit(&mut edit).unwrap();
        assert!(matches!(
            gate.begin_encoder(),
            Err(LifecycleError::Abandoned)
        ));
        assert_eq!(
            gate.begin_end_encoder(&mut encoder),
            Err(LifecycleError::StaleOwner)
        );
    }

    #[test]
    fn creation_refusal_keeps_live_registration_then_recovers_recording() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let mut edit = gate.begin_edit().unwrap();
        gate.cancel_encoder_creation(&mut encoder).unwrap();
        assert!(matches!(gate.prepare(), Err(LifecycleError::Busy)));
        gate.finish_edit(&mut edit).unwrap();
        assert_eq!(
            gate.begin_end_encoder(&mut encoder),
            Err(LifecycleError::StaleOwner)
        );
        let mut prepared = gate.prepare().unwrap();
        gate.cancel_prepared(&mut prepared).unwrap();
    }

    #[test]
    fn stale_borrow_cannot_overwrite_later_encoder_serial() {
        let gate = CommandLifecycle::new();
        let mut encoder = gate.begin_encoder().unwrap();
        let mut edit = gate.begin_edit().unwrap();
        let mut stale = Ticket {
            owner: edit.owner,
            word: edit.word,
            active: true,
        };
        gate.finish_edit(&mut edit).unwrap();
        gate.begin_end_encoder(&mut encoder).unwrap();
        gate.finish_end_encoder(&mut encoder).unwrap();
        let mut successor = gate.begin_encoder().unwrap();
        let mut next_edit = gate.begin_edit().unwrap();
        assert_eq!(
            gate.finish_edit(&mut stale),
            Err(LifecycleError::StaleOwner)
        );
        gate.finish_edit(&mut next_edit).unwrap();
        gate.begin_end_encoder(&mut successor).unwrap();
        gate.finish_end_encoder(&mut successor).unwrap();
    }
}
