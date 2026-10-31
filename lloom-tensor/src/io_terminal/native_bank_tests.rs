// Uncompiled source controls for the ACTUAL Bank/OwnerMailbox/HostService path.
// No GPU/native invocation: counted generic captures/anchors stand in for native
// objects only; they do not certify Metal, A/B consumers or protected enrollment.
use super::*;
use crate::memory_admission::ModelDomain;
use crate::retirement_service::HostRetirementService;
use std::sync::{atomic::AtomicUsize, mpsc, Barrier};
use std::time::Duration;

struct Capture {
    drops: Arc<AtomicUsize>,
    stall: Option<(Arc<Barrier>, Arc<Barrier>)>,
}
impl Drop for Capture {
    fn drop(&mut self) {
        if let Some((entered, release)) = &self.stall {
            entered.wait(); release.wait();
        }
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
struct Anchor(Arc<AtomicUsize>);
impl Drop for Anchor {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}
type TestBank = Bank<Capture, Anchor>;
type Handle = BankHandle<Capture, Anchor>;

fn setup(capacity: usize) -> (Handle, Arc<HostRetirementService>, ModelDomain) {
    // Test-only prefunded cold envelope, not a proposed production capacity.
    let domain = ModelDomain::isolated_numeric_fixture(1 << 20);
    let pass = domain.reserve(1 << 20).unwrap();
    let mut ticket = pass.context().take(1 << 20).unwrap();
    let service = HostRetirementService::manual();
    let failure = Arc::new(FailureSignal::default());
    let wake = service.wake();
    let core = IoTerminal::new(RuntimeStorage::try_new(capacity).unwrap()).unwrap();
    let mut slots: Vec<Slot<Capture, Anchor>> = Vec::new();
    slots.try_reserve_exact(capacity).unwrap();
    for _ in 0..capacity {
        slots.push(Slot {
            cell: IoCell::new(&core, failure.clone(), wake.clone()).unwrap(),
            native_key: AtomicUsize::new(0),
        });
    }
    let actual = std::alloc::Layout::array::<Slot<Capture, Anchor>>(slots.capacity()).unwrap().size()
        + std::mem::size_of::<TestBank>()
        + core.order.lock().frontier.allocation_layout().size()
        + HostRetirementService::requested_registration_bytes();
    assert!(actual < 1 << 20, "bounded cold fixture admission");
    let charge = ticket.try_commit(actual).unwrap();
    // SAFETY: no native objects, counted Anchor Drop is trusted/atomic; actual
    // admitted backing and real charge are retained by registered service root.
    let bank = unsafe { TestBank::from_admitted(ColdParts {
        core, slots, failure, wake, charge,
    }, &service) }.unwrap_or_else(|_| panic!("cold bank admission"));
    (bank, service, domain)
}

fn prepare(bank: &Handle, key: usize, drops: &Arc<AtomicUsize>, anchors: &Arc<AtomicUsize>,
           stall: Option<(Arc<Barrier>, Arc<Barrier>)>) -> Prepared<Capture, Anchor> {
    TestBank::try_prepare(bank, Capture { drops: drops.clone(), stall }, Anchor(anchors.clone()), key)
        .unwrap_or_else(|_| panic!("fixture data preparation"))
}

#[test]
fn actual_bank_complete_before_return_preserves_native_custody() {
    let (bank, service, _domain) = setup(1);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let p = prepare(&bank, 101, &drops, &anchors, None);
    bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
    bank.callback(p.index, p.native_key, RawStatus(3)).unwrap();
    assert_eq!(bank.core.successful(), 1, "BANK_DATA_BEFORE_RETURN");
    assert!(!service.manual_pass(), "BANK_RETURN_REQUIRED");
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst), bank.retired.load(Ordering::Acquire)),
               (0, 0, 0), "BANK_NO_PREMATURE_CREDIT");
    bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    assert!(service.manual_pass());
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst), bank.retired.load(Ordering::Acquire)), (1, 1, 1));
}

#[test]
fn actual_bank_stalled_capture_does_not_stall_later_data() {
    let (bank, service, _domain) = setup(2);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let a = prepare(&bank, 201, &drops, &anchors, Some((entered.clone(), release.clone())));
    let b = prepare(&bank, 202, &drops, &anchors, None);
    for p in [&a, &b] {
        bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    }
    bank.callback(a.index, a.native_key, RawStatus(3)).unwrap();
    let worker = { let service = service.clone(); std::thread::spawn(move || service.manual_pass()) };
    entered.wait();
    bank.callback(b.index, b.native_key, RawStatus(3)).unwrap();
    let witness = (bank.core.successful(), bank.retired.load(Ordering::Acquire), anchors.load(Ordering::SeqCst));
    release.wait(); // release even when the subsequent witness assertion fails
    assert!(worker.join().unwrap());
    assert_eq!(witness, (2, 0, 0), "BANK_DATA_INDEPENDENT_OF_CAPTURE_DROP");
    assert!(service.manual_pass());
    assert_eq!((bank.retired.load(Ordering::Acquire), anchors.load(Ordering::SeqCst)), (2, 2));
}

#[test]
fn actual_bank_service_holder_assists_callback_after_unlock() {
    let (bank, service, _domain) = setup(1);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let p = prepare(&bank, 301, &drops, &anchors, None);
    bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
    bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    let (held_tx, held_rx) = mpsc::sync_channel(0);
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let holder = { let bank = bank.clone(); std::thread::spawn(move || bank.begin_retire_with(|| {
        held_tx.send(()).unwrap(); release_rx.recv().unwrap();
    })) };
    held_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let (done_tx, done_rx) = mpsc::sync_channel(0);
    let callback = { let bank = bank.clone(); let index=p.index; let key=p.native_key;
        std::thread::spawn(move || { done_tx.send(bank.callback(index,key,RawStatus(3))).unwrap(); }) };
    let returned_while_held = done_rx.recv_timeout(Duration::from_secs(3));
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    // In a blocking mutant the sender may now be available; reap the original
    // worker before reporting the named failure rather than detach a child.
    if returned_while_held.is_err() { let _ = done_rx.recv_timeout(Duration::from_secs(3)); }
    callback.join().unwrap();
    assert!(returned_while_held.is_ok(), "BANK_CALLBACK_MUST_NOT_WAIT_FOR_HOLDER");
    assert_eq!(bank.core.successful(), 1, "BANK_HOLDER_UNLOCK_ASSISTS");
    assert!(service.manual_pass());
}

#[test]
fn actual_bank_wrong_native_identity_does_not_consume_ticket() {
    let (bank, service, _domain) = setup(1);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let old = prepare(&bank, 401, &drops, &anchors, None);
    bank.slots[old.index].cell.begin_handoff(old.ticket).unwrap();
    bank.slots[old.index].cell.handoff_returned(old.ticket).unwrap();
    bank.callback(old.index, old.native_key, RawStatus(3)).unwrap();
    assert!(service.manual_pass());
    let new = prepare(&bank, 402, &drops, &anchors, None);
    assert_ne!(old.ticket, new.ticket);
    bank.slots[new.index].cell.begin_handoff(new.ticket).unwrap();
    bank.slots[new.index].cell.handoff_returned(new.ticket).unwrap();
    assert!(matches!(bank.callback(old.index, old.native_key, RawStatus(2)),
                     Err(CallbackError::WrongNativeIdentity)), "BANK_STALE_NATIVE_KEY");
    assert!(!bank.core.failed(), "BANK_FOREIGN_CALLBACK_MUST_NOT_POISON");
    assert_eq!(bank.core.successful(), 1);
    bank.callback(new.index, new.native_key, RawStatus(3)).unwrap();
    assert_eq!(bank.core.successful(), 2, "BANK_RIGHT_CALLBACK_REMAINS_PUBLISHABLE");
    assert!(service.manual_pass());
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (2, 2));
}

#[test]
fn actual_bank_failed_later_record_closes_admission_behind_pending_head() {
    let (bank, service, _domain) = setup(3);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let a = prepare(&bank, 501, &drops, &anchors, None);
    let b = prepare(&bank, 502, &drops, &anchors, None);
    for p in [&a, &b] {
        bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    }
    bank.callback(b.index, b.native_key, RawStatus(2)).unwrap();
    assert!(bank.core.failed(), "BANK_FAILURE_BEFORE_PREFIX");
    assert_eq!(bank.core.successful(), 0);
    let refusal = TestBank::try_prepare(&bank,
        Capture { drops: drops.clone(), stall: None }, Anchor(anchors.clone()), 503);
    let (rejected_capture, rejected_anchor) = match refusal {
        Err(PrepareError::NotIssued { captures, anchors, .. }) => (captures, anchors),
        _ => panic!("BANK_FAILED_ADMISSION_MUST_RETURN_UNISSUED_OWNERS"),
    };
    assert_eq!(bank.issued.load(Ordering::Acquire), 2, "BANK_FAILED_ADMISSION_NO_SEQUENCE");
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (0, 0));
    drop((rejected_capture, rejected_anchor));
    bank.callback(a.index, a.native_key, RawStatus(3)).unwrap();
    assert_eq!(bank.core.successful(), 0, "BANK_FAILURE_CANNOT_REVIVE_DATA");
    assert!(service.manual_pass());
    assert!(service.manual_pass());
    assert_eq!(bank.retired.load(Ordering::Acquire), 2, "BANK_FAILED_RECORDS_DRAIN_CREDIT");
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (3, 3));
}

#[test]
fn actual_bank_full_window_returns_owners_until_contiguous_retirement() {
    let (bank, service, _domain) = setup(1);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let p = prepare(&bank, 601, &drops, &anchors, None);
    bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
    bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    bank.callback(p.index, p.native_key, RawStatus(3)).unwrap();
    assert_eq!(bank.core.successful(), 1);
    let rejected = TestBank::try_prepare(&bank,
        Capture { drops: drops.clone(), stall: None }, Anchor(anchors.clone()), 602);
    let (captures, anchor) = match rejected {
        Err(PrepareError::NotIssued { error: Error::Frontier(FrontierError::BeyondWindow), captures, anchors }) =>
            (captures, anchors),
        _ => panic!("BANK_VALIDATED_DATA_IS_NOT_CREDIT"),
    };
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (0, 0));
    assert!(service.manual_pass());
    let next = TestBank::try_prepare(&bank, captures, anchor, 602)
        .unwrap_or_else(|_| panic!("BANK_RETURNED_CREDIT_REUSES_SAME_OWNERS"));
    bank.slots[next.index].cell.begin_handoff(next.ticket).unwrap();
    bank.slots[next.index].cell.handoff_returned(next.ticket).unwrap();
    bank.callback(next.index, next.native_key, RawStatus(3)).unwrap();
    assert!(service.manual_pass());
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (2, 2));
}

#[test]
fn actual_bank_stale_head_and_cas_loss_return_unissued_owners() {
    // Two real producers over one admitted cell. In HeadSnapshot the old
    // issued value is below subsequently loaded retired; BeforeClaim reaches
    // a stale issued CAS after claiming the now-reusable mailbox generation.
    for point in [PreparePoint::HeadSnapshot, PreparePoint::BeforeClaim] {
        let (bank, service, _domain) = setup(1);
        let drops = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let (paused_tx, paused_rx) = mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = mpsc::sync_channel(0);
        let stale = {
            let bank = bank.clone(); let drops = drops.clone(); let anchors = anchors.clone();
            std::thread::spawn(move || TestBank::try_prepare_with(&bank,
                Capture { drops, stall: None }, Anchor(anchors), 701, |at| {
                    if at == point { paused_tx.send(()).unwrap(); resume_rx.recv().unwrap(); }
                }))
        };
        paused_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        let fresh = prepare(&bank, 702, &drops, &anchors, None);
        bank.slots[fresh.index].cell.begin_handoff(fresh.ticket).unwrap();
        bank.slots[fresh.index].cell.handoff_returned(fresh.ticket).unwrap();
        bank.callback(fresh.index, fresh.native_key, RawStatus(3)).unwrap();
        assert!(service.manual_pass());
        resume_tx.send(()).unwrap();
        let result = stale.join().unwrap();
        let (captures, anchor) = match result {
            Err(PrepareError::NotIssued { error: Error::Mailbox(MailboxError::Busy), captures, anchors }) =>
                (captures, anchors),
            _ => panic!("BANK_STALE_HEAD_IS_UNISSUED_NOT_PROTOCOL_FAILURE"),
        };
        assert!(!bank.core.failed(), "BANK_STALE_HEAD_MUST_NOT_POISON");
        assert_eq!(bank.issued.load(Ordering::Acquire), 1, "BANK_STALE_HEAD_NO_GAP");
        assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (1, 1),
                   "BANK_STALE_HEAD_RETURN_OWNERS_INTACT");
        let next = TestBank::try_prepare(&bank, captures, anchor, 703)
            .unwrap_or_else(|_| panic!("BANK_STALE_HEAD_RECOVERY"));
        bank.slots[next.index].cell.begin_handoff(next.ticket).unwrap();
        bank.slots[next.index].cell.handoff_returned(next.ticket).unwrap();
        bank.callback(next.index, next.native_key, RawStatus(3)).unwrap();
        assert!(service.manual_pass());
        assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (2, 2));
    }
}

#[test]
fn actual_bank_failure_during_install_roots_refusal_without_handoff() {
    let (bank, service, _domain) = setup(2);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let first = prepare(&bank, 801, &drops, &anchors, None);
    bank.slots[first.index].cell.begin_handoff(first.ticket).unwrap();
    bank.slots[first.index].cell.handoff_returned(first.ticket).unwrap();
    let result = TestBank::try_prepare_with(&bank,
        Capture { drops: drops.clone(), stall: None }, Anchor(anchors.clone()), 802,
        |at| if at == PreparePoint::Installed {
            bank.callback(first.index, first.native_key, RawStatus(2)).unwrap();
        });
    assert!(matches!(result, Err(PrepareError::RootedRefused)), "BANK_INSTALL_FAILURE_REFUSED");
    assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)), (0, 0),
               "BANK_INSTALL_FAILURE_STAYS_ROOTED");
    assert_eq!(bank.core.successful(), 0);
    assert!(service.manual_pass());
    assert!(service.manual_pass());
    assert_eq!((bank.retired.load(Ordering::Acquire), drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst)),
               (2, 2, 2), "BANK_REFUSAL_DRAINS_WITHOUT_NATIVE_CALL");
}

#[test]
fn actual_bank_unknown_native_return_quarantines_even_after_complete() {
    for completed in [false, true] {
        let (bank, service, _domain) = setup(1);
        let drops = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let p = prepare(&bank, 901, &drops, &anchors, None);
        bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        if completed { bank.callback(p.index, p.native_key, RawStatus(3)).unwrap(); }
        bank.core.handoff_unknown(&bank.slots[p.index].cell, p.ticket).unwrap();
        bank.request_progress();
        assert!(bank.core.failed());
        assert!(!service.manual_pass(), "BANK_UNKNOWN_RETURN_IS_NOT_CLEANUP");
        assert_eq!(bank.core.successful(), u64::from(completed), "BANK_ONLY_ACTUAL_COMPLETE_CERTIFIES");
        assert_eq!((drops.load(Ordering::SeqCst), anchors.load(Ordering::SeqCst), bank.retired.load(Ordering::Acquire)),
                   (0, 0, 0), "BANK_UNKNOWN_RETURN_NO_RELEASE");
        // Test process retains the unknown native authority. No fabricated
        // return or teardown free is used merely to make the fixture quiescent.
        std::mem::forget((bank, service, _domain, p));
    }
}

#[test]
fn actual_bank_two_publishers_join_the_consumed_pending_wave() {
    let (bank, service, _domain) = setup(2);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let a = prepare(&bank, 1001, &drops, &anchors, None);
    let b = prepare(&bank, 1002, &drops, &anchors, None);
    for p in [&a, &b] {
        bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    }
    let order = bank.core.order.lock();
    let publishers: Vec<_> = [&a, &b].into_iter().map(|p| {
        let bank = bank.clone(); let index = p.index; let key = p.native_key;
        std::thread::spawn(move || bank.callback(index, key, RawStatus(3)))
    }).collect();
    for publisher in publishers { publisher.join().unwrap().unwrap(); }
    assert!(bank.pending.load(Ordering::Acquire));
    drop(order);
    bank.assist();
    assert_eq!(bank.core.successful(), 2, "BANK_TWO_PUBLISHER_WAVE");
    assert!(!bank.pending.load(Ordering::Acquire));
    assert_eq!(bank.retired.load(Ordering::Acquire), 0);
    assert!(service.manual_pass());
    assert!(service.manual_pass());
}

#[test]
fn actual_bank_new_publication_between_observation_and_wave_clear_progresses() {
    let (bank, service, _domain) = setup(2);
    let drops = Arc::new(AtomicUsize::new(0));
    let anchors = Arc::new(AtomicUsize::new(0));
    let a = prepare(&bank, 1101, &drops, &anchors, None);
    let b = prepare(&bank, 1102, &drops, &anchors, None);
    for p in [&a, &b] {
        bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
    }
    // Seed an earlier published wave while this boundary test holds the actual
    // exclusion. The service-holder production path is covered separately.
    let order = bank.core.order.lock();
    bank.callback(a.index, a.native_key, RawStatus(3)).unwrap();
    drop(order);
    let mut new_publications = 0;
    bank.assist_with(|| {
        if new_publications == 0 {
            new_publications += 1;
            // This publisher arrives after the holder's pending.load(true),
            // before its clearing RMW, and cannot acquire the holder's lock.
            bank.callback(b.index, b.native_key, RawStatus(3)).unwrap();
        }
    });
    assert_eq!(new_publications, 1);
    assert_eq!(bank.core.successful(), 2, "BANK_PUBLICATION_BEFORE_CLEAR_NOT_LOST");
    assert!(!bank.pending.load(Ordering::Acquire));
    assert!(service.manual_pass());
    assert!(service.manual_pass());
    // These deterministic schedules exercise implementation progress, not a
    // proof that a store-only mutant must fail on this hardware memory model.
}
