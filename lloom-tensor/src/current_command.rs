//! A submitted command keeps its receipt until a replacement is available.
//! Replacement failure must not turn retry into a second submission.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) struct Current<C, R> {
    buffer: C,
    receipt: Option<R>,
}

impl<C, R> Current<C, R> {
    pub fn new(buffer: C) -> Self {
        Self {
            buffer,
            receipt: None,
        }
    }

    pub fn recording(&self) -> Option<&C> {
        self.receipt.is_none().then_some(&self.buffer)
    }

    pub fn submitted(&self) -> Option<(&C, &R)> {
        self.receipt.as_ref().map(|r| (&self.buffer, r))
    }

    pub fn submit_once<E>(&mut self, submit: impl FnOnce(&C) -> Result<R, E>) -> Result<&R, E> {
        if self.receipt.is_none() {
            self.receipt = Some(submit(&self.buffer)?);
        }
        // Both the existing and newly submitted paths retain their receipt.
        Ok(self.receipt.as_ref().expect("submitted receipt"))
    }

    pub fn submit_marked<E>(
        &mut self,
        pending: &AtomicBool,
        submit: impl FnOnce(&C) -> Result<R, E>,
    ) -> Result<&R, E> {
        let receipt = self.submit_once(submit)?;
        pending.store(true, Ordering::Release);
        Ok(receipt)
    }

    /// Failure keeps both the owned submission and the producer's counters.
    pub fn recover_recording<E>(
        &mut self,
        create: impl FnOnce() -> Result<C, E>,
        in_flight: &mut Vec<C>,
        dispatch_count: &AtomicUsize,
        pending: &AtomicBool,
        reset_to: usize,
    ) -> Result<Option<R>, E> {
        let Some((old, receipt)) = self.replace_submitted(create)? else {
            return Ok(None);
        };
        in_flight.push(old);
        dispatch_count.store(reset_to, Ordering::Release);
        pending.store(false, Ordering::Release);
        Ok(Some(receipt))
    }

    pub fn replace_submitted<E>(
        &mut self,
        create: impl FnOnce() -> Result<C, E>,
    ) -> Result<Option<(C, R)>, E> {
        if self.receipt.is_none() {
            return Ok(None);
        }
        let next = create()?;
        let old = std::mem::replace(&mut self.buffer, next);
        // The factory cannot access this state, so the receipt remains present.
        let receipt = self.receipt.take().expect("submitted receipt");
        Ok(Some((old, receipt)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[derive(Debug)]
    struct Buffer {
        id: u64,
        drops: Rc<Cell<u64>>,
    }
    impl Drop for Buffer {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn replacement_failures_preserve_original_receipt_and_owner_until_one_recovery() {
        let drops = Rc::new(Cell::new(0));
        let mut current = Current::new(Buffer {
            id: 4,
            drops: drops.clone(),
        });
        let epochs = Cell::new(0);
        let handlers = Cell::new(0);
        let signals = Cell::new(0);
        let submit = |cb: &Buffer| -> Result<(u64, u64), ()> {
            assert_eq!(cb.id, 4);
            epochs.set(epochs.get() + 1);
            handlers.set(handlers.get() + 1);
            signals.set(signals.get() + 1);
            Ok((17, 23))
        };
        assert_eq!(*current.submit_once(submit).unwrap(), (17, 23));
        for _ in 0..3 {
            assert!(current
                .replace_submitted(|| Err::<Buffer, _>("allocation"))
                .is_err());
            assert_eq!(*current.submit_once(submit).unwrap(), (17, 23));
            let (cb, receipt) = current.submitted().unwrap();
            assert_eq!((cb.id, *receipt), (4, (17, 23)));
            assert!(current.recording().is_none());
            assert_eq!(
                (epochs.get(), handlers.get(), signals.get(), drops.get()),
                (1, 1, 1, 0)
            );
        }
        let (old, receipt) = current
            .replace_submitted(|| {
                Ok::<_, ()>(Buffer {
                    id: 5,
                    drops: drops.clone(),
                })
            })
            .unwrap()
            .unwrap();
        assert_eq!((old.id, receipt), (4, (17, 23)));
        assert_eq!(current.recording().unwrap().id, 5);
        assert!(current
            .replace_submitted::<()>(|| panic!("recording needs no replacement"))
            .is_ok_and(|v: Option<(Buffer, (u64, u64))>| v.is_none()));
        assert_eq!(drops.get(), 0);
        drop(old);
        assert_eq!(drops.get(), 1);
        current
            .submit_once(|cb| {
                assert_eq!(cb.id, 5);
                epochs.set(epochs.get() + 1);
                Ok::<_, ()>((18, 24))
            })
            .unwrap();
        assert_eq!(epochs.get(), 2);
    }

    #[test]
    fn retained_submission_stays_accessible_without_replacement_or_new_epoch() {
        let mut current = Current::new(7);
        current
            .submit_once(|_| Ok::<_, ()>((11, None::<u64>)))
            .unwrap();
        assert!(current
            .replace_submitted(|| Err::<u64, _>("allocation"))
            .is_err());
        let (cb, receipt) = current.submitted().unwrap();
        assert_eq!((*cb, *receipt), (7, (11, None)));
        assert!(current.recording().is_none());
        assert_eq!(
            *current
                .submit_once(|_| panic!("must not resubmit to wait"))
                .unwrap_or_else(|_: ()| unreachable!()),
            (11, None)
        );
    }

    #[test]
    fn refusal_before_submission_preserves_recording_and_consumes_no_receipt() {
        let mut current = Current::<u64, u64>::new(3);
        assert_eq!(current.submit_once(|_| Err("refused")), Err("refused"));
        assert_eq!(current.recording(), Some(&3));
        assert!(current.submitted().is_none());
        assert!(current
            .replace_submitted::<()>(|| panic!("unsubmitted factory"))
            .is_ok_and(|v: Option<(u64, u64)>| v.is_none()));
        assert_eq!(current.submit_once(|_| Ok::<_, ()>(29)), Ok(&29));
    }
    #[test]
    fn zero_dispatch_submission_keeps_pending_hint_and_original_signal_until_replacement() {
        let pending = AtomicBool::new(false);
        let count = AtomicUsize::new(0);
        let mut in_flight = Vec::new();
        let mut current = Current::new(4);
        current
            .submit_marked(&pending, |_| Ok::<_, ()>(Some(23)))
            .unwrap();
        assert!(pending.load(Ordering::Acquire));
        for _ in 0..3 {
            assert!(current
                .recover_recording(
                    || Err::<u64, _>("allocation"),
                    &mut in_flight,
                    &count,
                    &pending,
                    0
                )
                .is_err());
            assert!(pending.load(Ordering::Acquire));
            assert_eq!(count.load(Ordering::Acquire), 0);
            assert!(in_flight.is_empty());
        }
        assert_eq!(
            current
                .recover_recording(|| Ok::<_, ()>(5), &mut in_flight, &count, &pending, 0)
                .unwrap(),
            Some(Some(23))
        );
        assert_eq!(in_flight, vec![4]);
        assert!(!pending.load(Ordering::Acquire));
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert!(current
            .recover_recording(
                || panic!("already recording"),
                &mut in_flight,
                &count,
                &pending,
                1
            )
            .unwrap_or_else(|_: ()| unreachable!())
            .is_none());
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert_eq!(in_flight, vec![4]);
    }

    #[test]
    fn replacement_failure_keeps_failed_request_count_then_recovery_resets_it() {
        for reset_to in [0, 1] {
            let count = AtomicUsize::new(51);
            let pending = AtomicBool::new(false);
            let mut in_flight = Vec::new();
            let mut current = Current::new(1);
            current
                .submit_marked(&pending, |_| Ok::<_, ()>(19))
                .unwrap();
            assert!(current
                .recover_recording(
                    || Err::<u64, _>("allocation"),
                    &mut in_flight,
                    &count,
                    &pending,
                    reset_to
                )
                .is_err());
            assert_eq!(count.load(Ordering::Acquire), 51);
            assert!(pending.load(Ordering::Acquire));
            assert_eq!(
                current
                    .recover_recording(
                        || Ok::<_, ()>(2),
                        &mut in_flight,
                        &count,
                        &pending,
                        reset_to
                    )
                    .unwrap(),
                Some(19)
            );
            assert_eq!(count.load(Ordering::Acquire), reset_to);
            assert!(!pending.load(Ordering::Acquire));
            assert_eq!(in_flight, vec![1]);
        }
    }
    #[test]
    fn failed_node_cut_stays_consumed_and_retry_counts_only_its_new_request() {
        use crate::command_policy::CutPolicy;
        for policy in [
            CutPolicy::DispatchCounter,
            CutPolicy::NodeIndex { nodes_per_cb: 2 },
            CutPolicy::NodeAligned,
        ] {
            let count = AtomicUsize::new(50);
            let node_cut = AtomicBool::new(true);
            let pending = AtomicBool::new(false);
            let mut in_flight = Vec::new();
            let mut current = Current::new(1);
            let before = count.fetch_add(1, Ordering::Relaxed);
            assert!(policy.take_cut(before, 50, &node_cut));
            current
                .submit_marked(&pending, |_| Ok::<_, ()>(31))
                .unwrap();
            assert!(current
                .recover_recording(
                    || Err::<u64, _>("allocation"),
                    &mut in_flight,
                    &count,
                    &pending,
                    1
                )
                .is_err());
            assert_eq!(count.load(Ordering::Acquire), 51);
            if policy != CutPolicy::DispatchCounter {
                assert!(!node_cut.load(Ordering::Relaxed));
            }
            assert_eq!(
                current
                    .recover_recording(|| Ok::<_, ()>(2), &mut in_flight, &count, &pending, 0)
                    .unwrap(),
                Some(31)
            );
            let before = count.fetch_add(1, Ordering::Relaxed);
            assert_eq!(before, 0);
            assert!(!policy.take_cut(before, 50, &node_cut));
            assert_eq!(count.load(Ordering::Acquire), 1);
            assert_eq!(in_flight, vec![1]);
            if policy != CutPolicy::DispatchCounter {
                node_cut.store(true, Ordering::Relaxed);
                assert!(policy.take_cut(1, 50, &node_cut));
                assert!(!policy.take_cut(1, 50, &node_cut));
            }
        }
    }
}
