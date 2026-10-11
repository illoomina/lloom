//! Same-cache borrowed-model custody controls: scalar state only, max two held
//! positions, no forward, weights, native device, worker or admission proof.
use super::tests::{engine, Cache, Tiny};
use super::*;
use crate::app::request_slot::{AbortState, Lease, State};
use lloom::nn::{Model, VarBuilder};
use lloom_tensor::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct View<'a>(&'a Tiny);
impl Model for View<'_> {
    type Config = ();
    fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
        lloom_tensor::bail!("bounded borrowed view is never loaded")
    }
}
impl CausalLM for View<'_> {
    type Cache = Cache;
    fn new_cache(&self, _: &(), _: &Device) -> Result<Cache> {
        lloom_tensor::bail!("bounded borrowed view must reuse its owner's cache")
    }
    fn embed(&self, _: &Tensor) -> Result<Tensor> {
        lloom_tensor::bail!("bounded borrowed view must not execute a forward")
    }
    fn forward_embeds(&self, _: &Tensor, _: usize, _: &mut Cache) -> Result<Tensor> {
        lloom_tensor::bail!("bounded borrowed view must not execute a forward")
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
}
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
            panic!("intentional bounded borrowed recovery panic");
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
#[test]
fn borrowed_and_mixed_stages_keep_busy_and_require_final_completion() -> Result<()> {
    let owner = engine()?;
    for mixed in [false, true] {
        let slot = slot(false);
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        {
            let view = View(&owner.model);
            lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |s, extra| {
                assert!(
                    slot.lock().is_none(),
                    "borrowed stage released outer custody"
                );
                let reentry = Lease::<Tiny, Side>::checkout(&slot, true);
                assert!(matches!(reentry, Err(error) if error.to_string().contains("busy")));
                assert_eq!(s.pos(), 2);
                assert_eq!(s.cache_mut().pos, 2);
                extra.partial = 17;
                Ok::<_, Error>(())
            })?;
        }
        assert_eq!(lease.sequence().tokens(), &[2, 3]);
        assert_eq!(lease.extra().partial, 17);
        assert!(
            slot.lock().is_none(),
            "completed stage released request custody"
        );
        lease.mark_complete();
        if mixed {
            lease.start_decode(&owner.model, &Device::Cpu, DecodeConfig::default());
        } else {
            let view = View(&owner.model);
            lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |_, _| {
                Ok::<_, Error>(())
            })?;
        }
        drop(lease);
        let held = slot.lock();
        let held = held.as_ref().expect("stage exit lost complete state");
        assert!(
            held.needs_recovery,
            "prior completion certified a newly started stage"
        );
        assert!(held.rings.is_some(), "stage exit lost rings");
        assert_eq!(held.extra.partial, 17);
    }
    Ok(())
}
#[test]
fn failed_borrowed_stage_preserves_family_error_and_recovers_before_retry() -> Result<()> {
    let owner = engine()?;
    let slot = slot(false);
    let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
    let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
    let failed: anyhow::Result<()> = {
        let view = View(&owner.model);
        lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |_, extra| {
            extra.partial = 17;
            Err(anyhow::anyhow!("bounded family error").context("retained family context"))
        })
    };
    assert_eq!(failed.unwrap_err().to_string(), "retained family context");
    assert_eq!(lease.sequence().tokens(), &[2, 3]);
    assert_eq!(lease.extra().partial, 17);
    assert!(slot.lock().is_none());
    assert_eq!(aborts.load(Ordering::Relaxed), 0, "stage Drop ran recovery");
    drop(lease);
    drop(Lease::<Tiny, _>::checkout(&slot, true)?);
    assert!(
        slot.lock().as_ref().unwrap().needs_recovery,
        "Held erased recovery"
    );
    let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
    {
        let view = View(&owner.model);
        lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |s, extra| {
            assert_eq!(s.pos(), 0);
            assert!(s.tokens().is_empty());
            assert_eq!(s.cache_mut().pos, 0);
            assert_eq!(extra.partial, 0);
            Ok::<_, Error>(())
        })?;
    }
    assert_eq!(aborts.load(Ordering::Relaxed), 1);
    lease.mark_complete();
    drop(lease);
    assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    Ok(())
}
#[test]
fn borrowed_unwind_and_recovery_panic_restore_without_recursive_abort() -> Result<()> {
    let owner = engine()?;
    for recovery_panic in [false, true] {
        let slot = slot(recovery_panic);
        slot.lock().as_mut().unwrap().needs_recovery = recovery_panic;
        let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
        let unwind_aborts = slot.lock().as_ref().unwrap().extra.unwind_aborts.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
            let view = View(&owner.model);
            lease.with_model(
                &view,
                &Device::Cpu,
                DecodeConfig::default(),
                |_, _| -> Result<()> {
                    assert!(!recovery_panic, "callback ran after failed recovery");
                    panic!("intentional bounded borrowed callback panic");
                },
            )
        }));
        let payload = result.expect_err("bounded stage did not reach its expected panic");
        let actual = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
        let expected = if recovery_panic {
            "intentional bounded borrowed recovery panic"
        } else {
            "intentional bounded borrowed callback panic"
        };
        assert_eq!(
            actual,
            Some(expected),
            "wrong panic masqueraded as stage recovery"
        );
        assert_eq!(aborts.load(Ordering::Relaxed), usize::from(recovery_panic));
        assert_eq!(
            unwind_aborts.load(Ordering::Relaxed),
            0,
            "Drop retried abort while unwinding"
        );
        assert!(
            slot.lock().as_ref().unwrap().needs_recovery,
            "panic lost recovery marker"
        );
        let mut lease = Lease::<Tiny, _>::checkout(&slot, true)?;
        {
            let view = View(&owner.model);
            lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |s, extra| {
                assert_eq!(s.pos(), 0);
                assert_eq!(s.cache_mut().pos, 0);
                assert_eq!(extra.partial, 0);
                Ok::<_, Error>(())
            })?;
        }
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            1 + usize::from(recovery_panic)
        );
        lease.mark_complete();
        drop(lease);
        assert!(!slot.lock().as_ref().unwrap().needs_recovery);
    }
    Ok(())
}
#[test]
fn successful_borrowed_stages_and_opt_out_preserve_existing_state() -> Result<()> {
    let owner = engine()?;
    for recover in [false, true] {
        let slot = slot(false);
        let aborts = slot.lock().as_ref().unwrap().extra.aborts.clone();
        let mut lease = Lease::<Tiny, _>::checkout(&slot, recover)?;
        for _ in 0..2 {
            let view = View(&owner.model);
            lease.with_model(&view, &Device::Cpu, DecodeConfig::default(), |s, extra| {
                assert_eq!(s.pos(), 2);
                assert_eq!(s.tokens(), &[2, 3]);
                assert_eq!(s.cache_mut().pos, 2);
                assert_eq!(extra.partial, 2);
                Ok::<_, Error>(())
            })?;
        }
        if recover {
            lease.mark_complete();
        }
        drop(lease);
        let held = slot.lock();
        let held = held.as_ref().unwrap();
        assert!(
            !held.needs_recovery,
            "successful completion or opt-out changed recovery policy"
        );
        assert_eq!(held.sequence.as_ref().unwrap().pos(), 2);
        assert!(held.rings.is_some());
        assert_eq!(aborts.load(Ordering::Relaxed), 0);
    }
    Ok(())
}
