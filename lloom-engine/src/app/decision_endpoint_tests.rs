//! Actual noncausal tuple slot, inert custody only: no model/Loaded/native/socket.
use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
struct Owner {
    identity: Box<u32>,
    drops: Arc<AtomicUsize>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn actual_tuple_reentry_refusal_and_error_restore_whole_custody() {
    let drops = Arc::new(AtomicUsize::new(0));
    let slot = Slot::new((
        Owner {
            identity: Box::new(17),
            drops: drops.clone(),
        },
        Box::new(29u32),
    ));
    let identities = {
        let held = slot.checkout().unwrap();
        ((&*held.0.identity) as *const u32, (&*held.1) as *const u32)
    };
    let result: Result<()> = (|| {
        let mut held = slot.checkout()?;
        assert_eq!(*held.0.identity, 17);
        assert_eq!(*held.1, 29);
        assert!(
            slot.checkout().is_err(),
            "reentry obtained in-flight custody"
        );
        *held.1 = 31;
        anyhow::bail!("intentional tuple error")
    })();
    assert_eq!(result.unwrap_err().to_string(), "intentional tuple error");
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let held = slot.checkout().expect("Err lost full tuple");
    assert_eq!(
        ((&*held.0.identity) as *const u32, (&*held.1) as *const u32),
        identities
    );
    assert_eq!(*held.1, 31);
    drop(held);
    drop(slot);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[test]
fn actual_tuple_exact_unwind_restores_and_drops_only_with_owner() {
    let drops = Arc::new(AtomicUsize::new(0));
    let slot = Slot::new((
        Owner {
            identity: Box::new(37),
            drops: drops.clone(),
        },
        Box::new(41u32),
    ));
    let identities = {
        let held = slot.checkout().unwrap();
        ((&*held.0.identity) as *const u32, (&*held.1) as *const u32)
    };
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut held = slot.checkout().unwrap();
        *held.1 = 43;
        panic!("intentional endpoint callback unwind");
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"intentional endpoint callback unwind")
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let held = slot.checkout().expect("unwind lost full tuple");
    assert_eq!(
        ((&*held.0.identity) as *const u32, (&*held.1) as *const u32),
        identities
    );
    assert_eq!(*held.1, 43);
    drop(held);
    drop(slot);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
