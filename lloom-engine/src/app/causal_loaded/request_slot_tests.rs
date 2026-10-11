//! Bounded scalar custody/recovery controls, not Qwen/MTP numerics or admission.
//! Reuses the existing V8 CPU owner fixture; at most two held positions, no
//! model weight/checkpoint/native execution or spawned worker.
use super::tests::{engine, Cache, Tiny};
use super::*;
use crate::app::request_slot::{AbortState, Lease, State};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Side {
    aborts: Arc<AtomicUsize>,
    unwind_aborts: Arc<AtomicUsize>,
    fail_once: bool,
    partial: usize,
}
impl AbortState for Side {
    fn abort(&mut self) {
        self.aborts.fetch_add(1, Ordering::Relaxed);
        if std::thread::panicking() {
            self.unwind_aborts.fetch_add(1, Ordering::Relaxed);
        }
        if std::mem::take(&mut self.fail_once) {
            panic!("intentional bounded deferred recovery panic");
        }
        self.partial = 0;
    }
}
fn slot(fail_once: bool) -> Mutex<Option<State<Cache, Side>>> {
    Mutex::new(Some(State {
        sequence: Some(Sequence::with_tokens(Cache { pos: 2 }, vec![2, 3])),
        rings: Some(RingPool::new()),
        extra: Side {
            aborts: Arc::new(AtomicUsize::new(0)),
            unwind_aborts: Arc::new(AtomicUsize::new(0)),
            fail_once,
            partial: 2,
        },
        needs_recovery: false,
    }))
}
fn start<'a>(lease: &mut Lease<'a, Tiny, Side>, owner: &'a CausalEngine<Tiny>) {
    lease.start_decode(&owner.model, &Device::Cpu, DecodeConfig::default());
}
#[test]
fn incomplete_and_held_checkout_preserve_marker_until_next_decode() -> Result<()> {
    let owner = engine()?;
    let slot = slot(false);
    let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
    let failed = (|| -> Result<()> {
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        start(&mut lease, &owner);
        lloom_tensor::bail!("intentional bounded decoder refusal")
    })();
    assert!(failed.is_err());
    assert_eq!(
        aborts.load(Ordering::Relaxed),
        0,
        "Drop ran deferred recovery"
    );
    assert!(
        slot.lock().as_ref().unwrap().needs_recovery,
        "incomplete request lost recovery marker"
    );
    drop(Lease::<Tiny, _>::checkout(&slot, true)?); // decision-only Held path
    assert!(
        slot.lock().as_ref().unwrap().needs_recovery,
        "Held checkout erased recovery marker"
    );
    let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
    start(&mut lease, &owner);
    assert_eq!(
        lease.session().cache_mut().pos,
        0,
        "deferred recovery retained partial cache"
    );
    assert_eq!(lease.session().pos(), 0);
    assert!(lease.session().tokens().is_empty());
    assert_eq!(
        lease.extra().partial,
        0,
        "deferred recovery retained partial side state"
    );
    assert_eq!(aborts.load(Ordering::Relaxed), 1);
    lease.mark_complete();
    drop(lease);
    assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    Ok(())
}
#[test]
fn callback_unwind_restores_custody_without_running_backend_recovery() -> Result<()> {
    let owner = engine()?;
    let slot = slot(false);
    let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
    let unwind_aborts = slot.lock().as_ref().unwrap().extra.unwind_aborts.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        start(&mut lease, &owner);
        panic!("intentional bounded checked-out callback unwind");
    }));
    assert!(result.is_err());
    assert_eq!(
        aborts.load(Ordering::Relaxed),
        0,
        "callback unwind entered backend recovery"
    );
    assert_eq!(unwind_aborts.load(Ordering::Relaxed), 0);
    assert!(
        slot.lock().as_ref().unwrap().needs_recovery,
        "callback unwind lost complete custody marker"
    );
    let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
    start(&mut lease, &owner);
    assert_eq!(aborts.load(Ordering::Relaxed), 1);
    assert_eq!(lease.extra().partial, 0);
    lease.mark_complete();
    Ok(())
}
#[test]
fn recovery_panic_restores_marker_and_can_retry_outside_unwind() -> Result<()> {
    let owner = engine()?;
    let slot = slot(true);
    slot.lock().as_mut().unwrap().needs_recovery = true;
    let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
    let unwind_aborts = slot.lock().as_ref().unwrap().extra.unwind_aborts.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        start(&mut lease, &owner);
        Ok(())
    }));
    assert!(result.is_err());
    assert_eq!(
        aborts.load(Ordering::Relaxed),
        1,
        "Drop retried panicking recovery"
    );
    assert_eq!(
        unwind_aborts.load(Ordering::Relaxed),
        0,
        "Drop retried recovery during unwind"
    );
    assert!(
        slot.lock().as_ref().unwrap().needs_recovery,
        "recovery panic lost retry marker"
    );
    let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
    start(&mut lease, &owner);
    assert_eq!(aborts.load(Ordering::Relaxed), 2);
    assert_eq!(lease.extra().partial, 0);
    assert_eq!(lease.session().cache_mut().pos, 0);
    lease.mark_complete();
    drop(lease);
    assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    Ok(())
}

#[test]
fn causal_recover_false_keeps_existing_partial_state_on_incomplete_exit() -> Result<()> {
    let owner = engine()?;
    let slot = slot(false);
    let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
    let mut lease = Lease::<Tiny, _>::checkout(&slot, false)?;
    start(&mut lease, &owner);
    drop(lease);
    assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    let mut lease = Lease::<Tiny, _>::checkout(&slot, false)?;
    start(&mut lease, &owner);
    assert_eq!(
        lease.session().cache_mut().pos,
        2,
        "causal opt-out reset existing cache"
    );
    assert_eq!(lease.session().pos(), 2);
    assert_eq!(lease.session().tokens(), &[2, 3]);
    assert_eq!(lease.extra().partial, 2);
    assert_eq!(aborts.load(Ordering::Relaxed), 0);
    lease.mark_complete();
    drop(lease);
    assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    Ok(())
}
