//! Portable IO terminal accounting; not yet connected to native IoQueue.
//!
//! Capacity is supplied by the adapter's proved admission policy. This module
//! does not choose it, allocate native owners, change GpuWait, or validate bytes.
//! In particular, Metal's Complete status is not proof against a short read.
use crate::{
    completion_frontier::{FrontierError, RuntimeCompletionFrontier, RuntimeStorage},
    completion_wake::Wake,
    retirement_mailbox::{
        CleanupPolicy, FeedbackPublicationError, MailboxError, OwnerMailbox, Ticket,
    },
    retirement_owners::FailureSignal,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Preserve every native status, including a value unknown to this version.
/// The local Metal SDK MTLIOCommandBuffer.h / objc2-metal MTLIOStatus binding
/// defines Pending=0, Cancelled=1, Error=2, Complete=3 (NOT CB Completed=4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawStatus(pub usize);
impl RawStatus {
    pub fn is_complete(self) -> bool {
        self.0 == 3
    }
}

/// Core-local disposition is immutable for the whole actual mailbox generation.
/// A provisional capacity reservation is NOT yet a mailbox record. The adapter
/// commits the token contract before preparing Data; if never accepted, it may
/// instead prepare Gap. No successful reserve→accept promotion is hidden here.
/// No accepted token or native handoff may escape between the provisional
/// credit reservation and installing its accepted record. Native integration
/// must prove that boundary; this source-only core does not establish it.
#[derive(Clone, Copy, Debug)]
struct Record {
    sequence: u64,
    accepted: bool,
}

/// Cold-bound metadata cell. The private mailbox prevents replacement/promotion
/// between a guarded read and a terminal phase claim. Existing Ticket remains
/// the only identity/generation authority; core_id is queue membership only.
pub(crate) struct IoCell {
    core_id: u64,
    mailbox: OwnerMailbox<Record, RawStatus>,
}
impl IoCell {
    pub fn new(
        core: &IoTerminal,
        failure: Arc<FailureSignal>,
        wake: Arc<Wake>,
    ) -> Result<Self, MailboxError> {
        Ok(Self {
            core_id: core.id,
            mailbox: OwnerMailbox::new(failure, wake)?,
        })
    }
    pub fn prepare_data(&self, sequence: u64) -> Result<Ticket, Error> {
        self.prepare(sequence, true)
    }
    pub fn prepare_gap(&self, sequence: u64) -> Result<Ticket, Error> {
        self.prepare(sequence, false)
    }
    fn prepare(&self, sequence: u64, accepted: bool) -> Result<Ticket, Error> {
        if sequence == u64::MAX {
            return Err(Error::Frontier(FrontierError::SequenceExhausted));
        }
        self.mailbox
            .prepare(Record { sequence, accepted })
            .map_err(|r| Error::Mailbox(r.error))
    }
    fn prepared(&self, ticket: Ticket) -> Result<Record, Error> {
        self.mailbox
            .edit_prepared(ticket, (), |record, _| *record, || {})
            .map(|(record, _)| record)
            .map_err(|error| match error {
                crate::retirement_mailbox::PreparedEditError::Rejected(r) => {
                    Error::Mailbox(r.error)
                }
                crate::retirement_mailbox::PreparedEditError::Panicked { .. } => {
                    Error::ObserverPanicked
                }
            })
    }
    pub fn begin_handoff(&self, ticket: Ticket) -> Result<(), Error> {
        if !self.prepared(ticket)?.accepted {
            return Err(Error::UnacceptedNativeHandoff);
        }
        self.mailbox.begin_handoff(ticket).map_err(Error::Mailbox)
    }
    pub fn handoff_returned(&self, ticket: Ticket) -> Result<(), Error> {
        self.mailbox
            .handoff_returned(ticket)
            .map_err(Error::Mailbox)
    }
    /// Portable metadata-only cleanup. The native adapter must execute its real
    /// resource retirement outside the frontier lock, and qualify native return,
    /// before returning any physical or metadata admission credit.
    pub fn cleanup(&self, terminal: Terminal) -> Result<(), Error> {
        if matches!(terminal.disposition, Disposition::Native(_)) {
            self.mailbox
                .with_callbacks(terminal.ticket, |_, _| {}, || {})
                .map_err(Error::Mailbox)?;
            self.mailbox
                .slot_returned(terminal.ticket)
                .map_err(Error::Mailbox)?;
            self.mailbox
                .publish_cleanup(terminal.ticket)
                .map_err(Error::Mailbox)?;
        }
        self.mailbox
            .try_detach_cleanup_with_policy(terminal.ticket, CleanupPolicy::ReturnedHandoff)
            .map_err(Error::Mailbox)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Disposition {
    Native(RawStatus),
    NeverAccepted,
    /// Accepted record, but the Prepared-phase CAS proved native/deferred
    /// handoff had never begun. This is failure, not a gap or native feedback.
    AcceptedNotSubmitted,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Terminal {
    pub sequence: u64,
    pub ticket: Ticket,
    pub disposition: Disposition,
}
#[derive(Debug)]
pub(crate) enum Error {
    Mailbox(MailboxError),
    ObserverPanicked,
    Frontier(FrontierError),
    AcceptedCannotAbandon,
    UnacceptedCannotRefuse,
    UnacceptedNativeHandoff,
    ForeignCore,
    IdentityExhausted,
}

pub(crate) struct IoTerminal {
    id: u64,
    frontier: Mutex<RuntimeCompletionFrontier<Terminal>>,
    failed: AtomicBool,
    successful: AtomicU64,
}
impl IoTerminal {
    /// The caller admits the backing before this core/token can escape.
    pub fn new(storage: RuntimeStorage<Terminal>) -> Result<Self, Error> {
        static NEXT_CORE: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_CORE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::IdentityExhausted)?;
        Ok(Self {
            id,
            frontier: Mutex::new(
                RuntimeCompletionFrontier::with_storage(storage).map_err(Error::Frontier)?,
            ),
            failed: AtomicBool::new(false),
            successful: AtomicU64::new(0),
        })
    }
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
    /// One-based value; 0 initially. Retirement of failed or abandoned records
    /// alone never advances this. Failure never contributes data readiness.
    /// An earlier successful retirement can race failure publication; its data
    /// prefix is still valid, while sticky failure closes later admission.
    pub fn successful(&self) -> u64 {
        self.successful.load(Ordering::Acquire)
    }

    /// The existing mailbox validates identity, generation and Pending phase
    /// BEFORE the observer can latch failure or enqueue a terminal record.
    /// Failure publication precedes acquiring the reorder lock: an uncompleted
    /// earlier record cannot postpone guarded-admission refusal.
    pub fn native_terminal(
        &self,
        cell: &IoCell,
        ticket: Ticket,
        status: RawStatus,
    ) -> Result<bool, Error> {
        if cell.core_id != self.id {
            return Err(Error::ForeignCore);
        }
        let observed = Cell::new(None);
        cell.mailbox
            .publish_feedback_observed(
                ticket,
                status,
                |record, raw| {
                    if !raw.is_complete() || !record.accepted {
                        self.failed.store(true, Ordering::Release);
                    }
                    observed.set(Some(*record));
                },
                || {},
            )
            .map_err(|error| match error {
                FeedbackPublicationError::Rejected(rejected) => Error::Mailbox(rejected.error),
                FeedbackPublicationError::ObserverPanicked => {
                    self.failed.store(true, Ordering::Release);
                    Error::ObserverPanicked
                }
            })?;
        let record = observed.get().expect("mailbox observer ran");
        if !record.accepted {
            // Mailbox retains feedback/owner for explicit adapter recovery.
            return Err(Error::UnacceptedNativeHandoff);
        }
        self.publish(Terminal {
            sequence: record.sequence,
            ticket,
            disposition: Disposition::Native(status),
        })
    }

    /// Only an immutable reserved owner in Prepared can become a gap. The
    /// mailbox's phase CAS refuses a concurrent/already-started native handoff.
    /// Accepted failures must take the adapter's failure path, never this one.
    pub fn abandon(&self, cell: &IoCell, ticket: Ticket) -> Result<bool, Error> {
        if cell.core_id != self.id {
            return Err(Error::ForeignCore);
        }
        let record = cell.prepared(ticket)?;
        if record.accepted {
            return Err(Error::AcceptedCannotAbandon);
        }
        cell.mailbox
            .publish_cancelled(ticket)
            .map_err(Error::Mailbox)?;
        self.publish(Terminal {
            sequence: record.sequence,
            ticket,
            disposition: Disposition::NeverAccepted,
        })
    }

    /// Accepted metadata whose actual handoff has provably not begun.
    /// The existing mailbox CAS arbitrates against begin_handoff. No accepted
    /// token, native commit or deferred handoff may escape without that phase.
    /// A failed/stale CAS does not poison this core or consume another owner.
    /// Native/deferred work that already escaped cannot use this method.
    pub fn refused_before_handoff(&self, cell: &IoCell, ticket: Ticket) -> Result<bool, Error> {
        if cell.core_id != self.id {
            return Err(Error::ForeignCore);
        }
        let record = cell.prepared(ticket)?;
        if !record.accepted {
            return Err(Error::UnacceptedCannotRefuse);
        }
        cell.mailbox
            .publish_cancelled(ticket)
            .map_err(Error::Mailbox)?;
        self.failed.store(true, Ordering::Release);
        self.publish(Terminal {
            sequence: record.sequence,
            ticket,
            disposition: Disposition::AcceptedNotSubmitted,
        })
    }

    fn publish(&self, terminal: Terminal) -> Result<bool, Error> {
        self.frontier
            .lock()
            .publish(terminal.sequence, terminal)
            .map_err(|rejected| {
                // Malformed adapter sequence is visible; its actual mailbox still
                // retains owner/feedback. A protocol error is not a readiness event.
                self.failed.store(true, Ordering::Release);
                Error::Frontier(rejected.reason)
            })
    }

    /// Only the drainer elected by publication calls this. Actual resource
    /// cleanup/callback work occurs outside the frontier lock, before finish.
    pub fn next(&self) -> Result<Option<Ready<'_>>, Error> {
        self.frontier
            .lock()
            .pop_ready_or_release_drainer()
            .map(|record| {
                record.map(|(_, terminal)| Ready {
                    core: self,
                    terminal,
                })
            })
            .map_err(Error::Frontier)
    }
}

/// Non-Clone and bound to its originating core. Dropping without finish retains
/// the frontier's active slot; it cannot silently recycle admission credit.
pub(crate) struct Ready<'a> {
    core: &'a IoTerminal,
    terminal: Terminal,
}
impl Ready<'_> {
    pub fn terminal(&self) -> Terminal {
        self.terminal
    }
    pub fn finish(self) -> Result<(), Error> {
        self.core
            .frontier
            .lock()
            .finish(self.terminal.sequence)
            .map_err(Error::Frontier)?;
        if matches!(self.terminal.disposition, Disposition::Native(raw) if raw.is_complete())
            && !self.core.failed()
        {
            self.core
                .successful
                .store(self.terminal.sequence + 1, Ordering::Release);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{completion_wake::Wake, retirement_owners::FailureSignal};
    use std::sync::Arc;

    fn cell(core: &IoTerminal) -> IoCell {
        IoCell::new(
            core,
            Arc::new(FailureSignal::default()),
            Arc::new(Wake::default()),
        )
        .unwrap()
    }
    fn accepted(cell: &IoCell, sequence: u64) -> Ticket {
        let ticket = cell.prepare_data(sequence).unwrap();
        cell.begin_handoff(ticket).unwrap();
        cell.handoff_returned(ticket).unwrap();
        ticket
    }
    fn cleanup(cell: &IoCell, terminal: Terminal) {
        cell.cleanup(terminal).unwrap();
    }
    fn drain(core: &IoTerminal, cells: &[IoCell]) -> Vec<Terminal> {
        let mut retired = Vec::new();
        while let Some(ready) = core.next().unwrap() {
            let terminal = ready.terminal();
            cleanup(&cells[terminal.sequence as usize], terminal);
            ready.finish().unwrap();
            retired.push(terminal);
        }
        retired
    }

    #[test]
    fn every_raw_noncomplete_status_is_failure_and_is_preserved() {
        for raw in [0, 1, 2, 4, usize::MAX] {
            let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
            let cells = [cell(&core)];
            let ticket = accepted(&cells[0], 0);
            assert!(core
                .native_terminal(&cells[0], ticket, RawStatus(raw))
                .unwrap());
            assert!(
                core.failed(),
                "every non-Complete status must latch failure"
            );
            let retired = drain(&core, &cells);
            assert_eq!(retired[0].disposition, Disposition::Native(RawStatus(raw)));
            assert_eq!(core.successful(), 0);
        }
    }

    #[test]
    fn all_six_success_orders_publish_only_consecutive_data() {
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let core = IoTerminal::new(RuntimeStorage::try_new(3).unwrap()).unwrap();
            let cells = [cell(&core), cell(&core), cell(&core)];
            let tickets = std::array::from_fn::<_, 3, _>(|i| accepted(&cells[i], i as u64));
            let mut arrived = [false; 3];
            for i in order {
                arrived[i] = true;
                if core
                    .native_terminal(&cells[i], tickets[i], RawStatus(3))
                    .unwrap()
                {
                    drain(&core, &cells);
                }
                assert_eq!(
                    core.successful(),
                    arrived.iter().take_while(|&&v| v).count() as u64,
                    "successful prefix crossed missing record: {order:?}"
                );
            }
            assert!(!core.failed());
        }
    }

    #[test]
    fn failure_latches_behind_hole_and_retirement_still_drains() {
        let core = IoTerminal::new(RuntimeStorage::try_new(3).unwrap()).unwrap();
        let cells = [cell(&core), cell(&core), cell(&core)];
        let tickets = std::array::from_fn::<_, 3, _>(|i| accepted(&cells[i], i as u64));
        assert!(!core
            .native_terminal(&cells[2], tickets[2], RawStatus(3))
            .unwrap());
        assert!(!core
            .native_terminal(&cells[1], tickets[1], RawStatus(1))
            .unwrap());
        assert!(core.failed(), "failure cannot wait for sequence zero");
        assert_eq!(core.successful(), 0);
        assert!(core
            .native_terminal(&cells[0], tickets[0], RawStatus(3))
            .unwrap());
        assert_eq!(drain(&core, &cells).len(), 3);
        assert_eq!(core.successful(), 0, "retirement is not successful data");
    }

    #[test]
    fn unused_gap_retires_without_becoming_data_then_next_success_advances() {
        let core = IoTerminal::new(RuntimeStorage::try_new(2).unwrap()).unwrap();
        let cells = [cell(&core), cell(&core)];
        let gap = cells[0].prepare_gap(0).unwrap();
        assert!(core.abandon(&cells[0], gap).unwrap());
        drain(&core, &cells);
        assert_eq!(core.successful(), 0);
        let data = accepted(&cells[1], 1);
        assert!(core.native_terminal(&cells[1], data, RawStatus(3)).unwrap());
        drain(&core, &cells);
        assert_eq!(core.successful(), 2);
        assert!(!core.failed());
    }

    #[test]
    fn accepted_prepared_record_is_not_an_abandoned_gap() {
        let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let cells = [cell(&core)];
        let ticket = cells[0].prepare_data(0).unwrap();
        assert!(
            matches!(
                core.abandon(&cells[0], ticket),
                Err(Error::AcceptedCannotAbandon)
            ),
            "accepted reservation cannot be retired as an unused gap"
        );
        cells[0].begin_handoff(ticket).unwrap();
        cells[0].handoff_returned(ticket).unwrap();
        assert!(core
            .native_terminal(&cells[0], ticket, RawStatus(2))
            .unwrap());
        drain(&core, &cells);
        assert!(core.failed());
        assert_eq!(core.successful(), 0);
    }

    #[test]
    fn actual_mailbox_identity_and_generation_reject_old_failure() {
        let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let c = cell(&core);
        let wrong = cell(&core);
        let old = accepted(&c, 0);
        assert!(matches!(
            core.native_terminal(&wrong, old, RawStatus(2)),
            Err(Error::Mailbox(MailboxError::Stale))
        ));
        assert!(!core.failed());
        assert!(core.native_terminal(&c, old, RawStatus(3)).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&c, ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        let new = accepted(&c, 1);
        assert!(
            matches!(
                core.native_terminal(&c, old, RawStatus(2)),
                Err(Error::Mailbox(MailboxError::Stale))
            ),
            "stale native terminal reached new occupant"
        );
        assert!(!core.failed(), "old failure poisoned new occupant");
        assert!(core.native_terminal(&c, new, RawStatus(3)).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&c, ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        assert_eq!(core.successful(), 2);
    }

    #[test]
    fn foreign_core_refusal_preserves_original_pending_terminal() {
        let owner = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let foreign = IoTerminal::new(RuntimeStorage::try_new(2).unwrap()).unwrap();
        let cells = [cell(&owner)];
        let t = accepted(&cells[0], 0);
        assert!(
            matches!(
                foreign.native_terminal(&cells[0], t, RawStatus(2)),
                Err(Error::ForeignCore)
            ),
            "foreign core consumed a valid owner's terminal"
        );
        assert!(!foreign.failed());
        assert!(!owner.failed());
        assert_eq!(foreign.successful(), 0);
        assert!(
            owner.native_terminal(&cells[0], t, RawStatus(3)).unwrap(),
            "foreign refusal consumed Pending phase"
        );
        drain(&owner, &cells);
        assert_eq!(owner.successful(), 1);
        assert_eq!(foreign.successful(), 0);
    }

    #[test]
    fn prepared_gap_cannot_promote_or_handoff_but_next_generation_can_be_data() {
        let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let c = cell(&core);
        let gap = c.prepare_gap(0).unwrap();
        assert!(matches!(
            c.prepare_data(0),
            Err(Error::Mailbox(MailboxError::Busy))
        ));
        assert!(matches!(
            c.begin_handoff(gap),
            Err(Error::UnacceptedNativeHandoff)
        ));
        assert!(core.abandon(&c, gap).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&c, ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        assert_eq!(core.successful(), 0);
        let data = accepted(&c, 1);
        assert!(core.native_terminal(&c, data, RawStatus(3)).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&c, ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        assert_eq!(core.successful(), 2);
    }

    #[test]
    fn duplicate_native_terminal_cannot_change_original_status() {
        let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let cells = [cell(&core)];
        let t = accepted(&cells[0], 0);
        assert!(core.native_terminal(&cells[0], t, RawStatus(3)).unwrap());
        assert!(matches!(
            core.native_terminal(&cells[0], t, RawStatus(2)),
            Err(Error::Mailbox(MailboxError::WrongPhase))
        ));
        assert!(!core.failed());
        let retired = drain(&core, &cells);
        assert_eq!(retired[0].disposition, Disposition::Native(RawStatus(3)));
        assert_eq!(core.successful(), 1);
    }
    #[test]
    fn accepted_refusal_latches_behind_hole_and_keeps_retirement_credit() {
        let core = IoTerminal::new(RuntimeStorage::try_new(2).unwrap()).unwrap();
        let cells = [cell(&core), cell(&core)];
        let first = accepted(&cells[0], 0);
        let refused = cells[1].prepare_data(1).unwrap();
        assert!(!core.refused_before_handoff(&cells[1], refused).unwrap());
        assert!(
            core.failed(),
            "accepted refusal must immediately latch failure"
        );
        assert_eq!(core.successful(), 0);
        let rejected = core
            .frontier
            .lock()
            .publish(
                2,
                Terminal {
                    sequence: 2,
                    ticket: refused,
                    disposition: Disposition::AcceptedNotSubmitted,
                },
            )
            .unwrap_err();
        assert_eq!(
            rejected.reason,
            FrontierError::BeyondWindow,
            "out-of-order accepted refusal returned credit"
        );
        assert!(core
            .native_terminal(&cells[0], first, RawStatus(3))
            .unwrap());
        let ready = core.next().unwrap().unwrap();
        assert_eq!(ready.terminal().sequence, 0);
        cleanup(&cells[0], ready.terminal());
        // Dequeuing and owner cleanup alone cannot release sequence credit.
        assert_eq!(
            core.frontier
                .lock()
                .publish(2, rejected.value)
                .unwrap_err()
                .reason,
            FrontierError::BeyondWindow,
            "dequeue released refused-cohort credit"
        );
        ready.finish().unwrap();
        let ready = core.next().unwrap().unwrap();
        assert_eq!(
            ready.terminal().disposition,
            Disposition::AcceptedNotSubmitted,
            "accepted refusal disguised as gap or native feedback"
        );
        cleanup(&cells[1], ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        assert_eq!(
            core.successful(),
            0,
            "accepted refusal advanced successful data"
        );
        // Actual mailbox generation can recycle only after detached cleanup.
        let next = cells[0].prepare_gap(2).unwrap();
        assert!(core.abandon(&cells[0], next).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&cells[0], ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
    }

    #[test]
    fn refusal_and_handoff_each_win_their_explicit_order() {
        for refuse_first in [true, false] {
            let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
            let cells = [cell(&core)];
            let ticket = cells[0].prepare_data(0).unwrap();
            if refuse_first {
                assert!(core.refused_before_handoff(&cells[0], ticket).unwrap());
                assert!(
                    matches!(
                        cells[0].begin_handoff(ticket),
                        Err(Error::Mailbox(MailboxError::WrongPhase))
                    ),
                    "refusal did not exclude later native handoff"
                );
                assert!(matches!(
                    core.refused_before_handoff(&cells[0], ticket),
                    Err(Error::Mailbox(MailboxError::WrongPhase))
                ));
                assert!(core.failed());
            } else {
                cells[0].begin_handoff(ticket).unwrap();
                assert!(matches!(
                    core.refused_before_handoff(&cells[0], ticket),
                    Err(Error::Mailbox(MailboxError::WrongPhase))
                ));
                assert!(!core.failed(), "rejected refusal poisoned handed-off work");
                cells[0].handoff_returned(ticket).unwrap();
                assert!(core
                    .native_terminal(&cells[0], ticket, RawStatus(3))
                    .unwrap());
            }
            let retired = drain(&core, &cells);
            assert_eq!(
                retired[0].disposition,
                if refuse_first {
                    Disposition::AcceptedNotSubmitted
                } else {
                    Disposition::Native(RawStatus(3))
                }
            );
            assert_eq!(core.successful(), u64::from(!refuse_first));
        }
    }

    #[test]
    fn concurrent_refusal_and_handoff_have_one_owner() {
        let mut refused_count = 0;
        let mut handed_count = 0;
        for _ in 0..64 {
            let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
            let cells = [cell(&core)];
            let ticket = cells[0].prepare_data(0).unwrap();
            let start = std::sync::Barrier::new(2);
            let (refused, handed) = std::thread::scope(|scope| {
                let refusal = scope.spawn(|| {
                    start.wait();
                    core.refused_before_handoff(&cells[0], ticket)
                });
                let handoff = scope.spawn(|| {
                    start.wait();
                    cells[0].begin_handoff(ticket)
                });
                (refusal.join().unwrap(), handoff.join().unwrap())
            });
            assert_ne!(
                refused.is_ok(),
                handed.is_ok(),
                "refusal/handoff ownership not exclusive"
            );
            if refused.is_ok() {
                refused_count += 1;
                assert!(core.failed());
            } else {
                handed_count += 1;
                assert!(!core.failed(), "losing refusal poisoned native owner");
                cells[0].handoff_returned(ticket).unwrap();
                assert!(core
                    .native_terminal(&cells[0], ticket, RawStatus(3))
                    .unwrap());
            }
            let retired = drain(&core, &cells);
            assert_eq!(retired.len(), 1);
            assert_eq!(core.successful(), u64::from(handed.is_ok()));
        }
        assert_eq!(refused_count + handed_count, 64);
        println!("IO_REFUSAL_RACE cases=64 refused={refused_count} handed={handed_count} explicit_orders=separate_test");
    }

    #[test]
    fn refusal_rejects_gap_foreign_and_stale_without_poisoning() {
        let core = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let foreign = IoTerminal::new(RuntimeStorage::try_new(1).unwrap()).unwrap();
        let cells = [cell(&core)];
        let gap = cells[0].prepare_gap(0).unwrap();
        assert!(matches!(
            core.refused_before_handoff(&cells[0], gap),
            Err(Error::UnacceptedCannotRefuse)
        ));
        assert!(!core.failed());
        assert!(core.abandon(&cells[0], gap).unwrap());
        drain(&core, &cells);
        let data = cells[0].prepare_data(1).unwrap();
        assert!(matches!(
            foreign.refused_before_handoff(&cells[0], data),
            Err(Error::ForeignCore)
        ));
        assert!(!foreign.failed());
        assert!(matches!(
            core.refused_before_handoff(&cells[0], gap),
            Err(Error::Mailbox(MailboxError::Stale))
        ));
        assert!(
            !core.failed(),
            "stale/foreign refusal poisoned actual owner"
        );
        // The same valid ticket still hands off and completes after both refusals.
        cells[0].begin_handoff(data).unwrap();
        cells[0].handoff_returned(data).unwrap();
        assert!(core.native_terminal(&cells[0], data, RawStatus(3)).unwrap());
        let ready = core.next().unwrap().unwrap();
        cleanup(&cells[0], ready.terminal());
        ready.finish().unwrap();
        assert!(core.next().unwrap().is_none());
        assert_eq!(core.successful(), 2);
    }
}
