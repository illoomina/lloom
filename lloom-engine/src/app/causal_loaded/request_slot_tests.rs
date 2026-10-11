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

/// Actual shared recovery gate used by the LFM caller. Scalar fence callbacks
/// model refusal/terminal admission only, not native prefix-data completion.
#[cfg(test)]
mod recovery_fence_tests {
    use super::*;

    #[test]
    fn refused_prefix_keeps_held_custody_and_marker_until_successful_retry() -> Result<()> {
        let owner = engine()?;
        let slot = slot(false);
        slot.lock().as_mut().unwrap().needs_recovery = true;
        let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
        // One CPU-only ring, raw ctor's fixed 4092-slot geometry (no slack
        // expansion, draw, publish, native device or forward execution).
        let (token_identity, ring_identity) = {
            let mut guard = slot.lock();
            let held = guard.as_mut().unwrap();
            let rings = held.rings.as_mut().unwrap();
            assert!(rings.prepare(&[7], &Device::Cpu, lloom::generation::EosRead::Blocking)?);
            let ring = rings.ring(&[7], &Device::Cpu, lloom::generation::EosRead::Blocking)?;
            (
                held.sequence.as_ref().unwrap().tokens().as_ptr(),
                ring as *const _,
            )
        };
        let fences = AtomicUsize::new(0);
        let refused = (|| -> Result<()> {
            let lease = Lease::<Tiny, _>::checkout(&slot, true)?;
            lease.before_recovery(|| {
                fences.fetch_add(1, Ordering::Relaxed);
                assert!(
                    slot.lock().is_none(),
                    "fence released outer request custody"
                );
                assert!(Lease::<Tiny, _>::checkout(&slot, true).is_err());
                lloom_tensor::bail!("intentional bounded prefix refusal")
            })?;
            panic!("decoder started after refused prefix");
        })();
        assert_eq!(
            refused.unwrap_err().to_string(),
            "intentional bounded prefix refusal"
        );
        {
            let mut guard = slot.lock();
            let held = guard.as_mut().expect("refused prefix lost Held custody");
            assert!(
                held.needs_recovery,
                "refused prefix cleared recovery marker"
            );
            assert_eq!(held.sequence.as_ref().unwrap().tokens(), &[2, 3]);
            assert_eq!(held.sequence.as_ref().unwrap().cache().pos, 2);
            assert_eq!(held.extra.partial, 2);
            assert_eq!(
                held.sequence.as_ref().unwrap().tokens().as_ptr(),
                token_identity,
                "refused prefix replaced sequence storage"
            );
            assert!(
                Arc::ptr_eq(&held.extra.aborts, &aborts),
                "refused prefix replaced side-state identity"
            );
            let rings = held
                .rings
                .as_mut()
                .expect("refused prefix lost ring custody");
            assert!(
                !rings.prepare(&[7], &Device::Cpu, lloom::generation::EosRead::Blocking)?,
                "refused prefix replaced existing ring"
            );
            assert_eq!(
                rings.ring(&[7], &Device::Cpu, lloom::generation::EosRead::Blocking)? as *const _,
                ring_identity,
                "refused prefix replaced ring allocation"
            );
        }
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            0,
            "refused prefix ran reset"
        );
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        lease.before_recovery(|| -> Result<()> {
            fences.fetch_add(1, Ordering::Relaxed);
            assert_eq!(
                aborts.load(Ordering::Relaxed),
                0,
                "reset ran before prefix success"
            );
            assert_eq!(lease.sequence().cache().pos, 2);
            assert_eq!(lease.sequence().tokens().as_ptr(), token_identity);
            assert_eq!(lease.extra().partial, 2);
            Ok(())
        })?;
        assert!(
            lease.needs_recovery(),
            "successful fence prematurely cleared marker"
        );
        start(&mut lease, &owner);
        assert_eq!(lease.session().cache_mut().pos, 0);
        assert_eq!(lease.extra().partial, 0);
        assert_eq!(aborts.load(Ordering::Relaxed), 1);
        lease.mark_complete();
        drop(lease);
        assert!(!slot.lock().as_ref().unwrap().needs_recovery);
        assert_eq!(fences.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[test]
    fn prefix_unwind_restores_held_custody_without_reset_or_recursive_fence() -> Result<()> {
        let slot = slot(false);
        slot.lock().as_mut().unwrap().needs_recovery = true;
        let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
        let fences = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let lease = Lease::<Tiny, _>::checkout(&slot, true)?;
            lease.before_recovery(|| -> Result<()> {
                fences.fetch_add(1, Ordering::Relaxed);
                panic!("intentional bounded prefix unwind");
            })
        }));
        let payload = result.expect_err("prefix unwind was not propagated");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("intentional bounded prefix unwind")
        );
        assert_eq!(
            fences.load(Ordering::Relaxed),
            1,
            "Drop retried prefix fence"
        );
        assert_eq!(aborts.load(Ordering::Relaxed), 0, "prefix unwind ran reset");
        let lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        assert!(lease.needs_recovery());
        assert_eq!(lease.sequence().tokens(), &[2, 3]);
        assert_eq!(lease.extra().partial, 2);
        Ok(())
    }

    #[test]
    fn completed_request_reuse_and_recovery_opt_out_skip_prefix_fence() -> Result<()> {
        let owner = engine()?;
        let slot = slot(false);
        let fences = AtomicUsize::new(0);
        for _ in 0..2 {
            let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
            lease.before_recovery(|| -> Result<()> {
                fences.fetch_add(1, Ordering::Relaxed);
                lloom_tensor::bail!("unnecessary success-path fence")
            })?;
            start(&mut lease, &owner);
            lease.mark_complete();
        }
        assert!(!slot.lock().as_ref().unwrap().needs_recovery);
        slot.lock().as_mut().unwrap().needs_recovery = true;
        let lease = Lease::<Tiny, _>::checkout(&slot, false)?;
        lease.before_recovery(|| -> Result<()> {
            fences.fetch_add(1, Ordering::Relaxed);
            lloom_tensor::bail!("recovery opt-out called prefix fence")
        })?;
        drop(lease);
        assert!(slot.lock().as_ref().unwrap().needs_recovery);
        assert_eq!(fences.load(Ordering::Relaxed), 0);
        Ok(())
    }
}
