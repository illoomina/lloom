//! Fixed-capacity collection of out-of-order completions. Dequeuing a record
//! is not retiring it: its callbacks and resource handoff must finish first.
//! The caller serializes access, and executes callbacks outside that lock.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrontierError {
    ZeroCapacity,
    SequenceExhausted,
    BeforeFrontier,
    BeyondWindow,
    Duplicate,
    NoDrainer,
    RetirementPending,
    WrongRetirement,
}

/// Rejection returns the original resource owner without releasing it.
#[derive(Debug)]
pub(crate) struct Rejected<T> {
    pub reason: FrontierError,
    pub value: T,
}

pub(crate) struct CompletionFrontier<T, const N: usize> {
    next: u64,
    retired: u64,
    pending: [Option<(u64, T)>; N],
    active: Option<u64>,
    draining: bool,
}

impl<T, const N: usize> CompletionFrontier<T, N> {
    pub fn new() -> Result<Self, FrontierError> {
        if N == 0 {
            return Err(FrontierError::ZeroCapacity);
        }
        Ok(Self {
            next: 0,
            retired: 0,
            pending: std::array::from_fn(|_| None),
            active: None,
            draining: false,
        })
    }

    /// Returns true only to the newly elected drainer. The window is bounded
    /// by fully retired work, not the next record whose callback may start.
    pub fn publish(&mut self, sequence: u64, value: T) -> Result<bool, Rejected<T>> {
        let reject = |reason, value| Err(Rejected { reason, value });
        if sequence.checked_add(1).is_none() {
            return reject(FrontierError::SequenceExhausted, value);
        }
        let Some(distance) = sequence.checked_sub(self.retired) else {
            return reject(FrontierError::BeforeFrontier, value);
        };
        if distance >= N as u64 {
            return reject(FrontierError::BeyondWindow, value);
        }
        let index = (sequence % N as u64) as usize;
        if sequence < self.next || self.pending[index].is_some() {
            return reject(FrontierError::Duplicate, value);
        }
        self.pending[index] = Some((sequence, value));
        let head = (self.next % N as u64) as usize;
        if !self.draining && self.pending[head].is_some() {
            self.draining = true;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Only the elected drainer calls this. An empty head relinquishes the
    /// election in the same serialized operation that observes the hole.
    pub fn pop_ready_or_release_drainer(&mut self) -> Result<Option<(u64, T)>, FrontierError> {
        if !self.draining {
            return Err(FrontierError::NoDrainer);
        }
        if self.active.is_some() {
            return Err(FrontierError::RetirementPending);
        }
        let index = (self.next % N as u64) as usize;
        if self.pending[index].is_none() {
            self.draining = false;
            return Ok(None);
        }
        let following = self
            .next
            .checked_add(1)
            .ok_or(FrontierError::SequenceExhausted)?;
        let record = self.pending[index].take().expect("checked occupied head");
        self.active = Some(self.next);
        self.next = following;
        Ok(Some(record))
    }

    /// Publish after the dequeued record's callbacks finish. The native slot
    /// can then be returned before asking for the next ready record.
    pub fn finish(&mut self, sequence: u64) -> Result<(), FrontierError> {
        if self.active != Some(sequence) {
            return Err(FrontierError::WrongRetirement);
        }
        let following = sequence
            .checked_add(1)
            .ok_or(FrontierError::SequenceExhausted)?;
        self.retired = following;
        self.active = None;
        Ok(())
    }

    #[cfg(test)]
    pub fn retired_prefix(&self) -> u64 {
        self.retired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    fn drain<const N: usize>(f: &mut CompletionFrontier<u64, N>, out: &mut Vec<u64>) {
        while let Some((sequence, value)) = f.pop_ready_or_release_drainer().unwrap() {
            assert_eq!(sequence, out.len() as u64);
            assert_eq!(value, sequence);
            assert_eq!(f.retired_prefix(), sequence, "dequeue is not retirement");
            assert_eq!(
                f.pop_ready_or_release_drainer(),
                Err(FrontierError::RetirementPending)
            );
            out.push(value);
            f.finish(sequence).unwrap();
            assert_eq!(f.retired_prefix(), sequence + 1);
        }
    }

    #[test]
    fn all_eight_completion_permutations_retire_only_the_ready_prefix() {
        fn visit(order: &mut [u64; 8], at: usize, cases: &mut usize) {
            if at != order.len() {
                for i in at..order.len() {
                    order.swap(at, i);
                    visit(order, at + 1, cases);
                    order.swap(at, i);
                }
                return;
            }
            let mut f = CompletionFrontier::<u64, 8>::new().unwrap();
            let mut out = Vec::new();
            let mut arrived = [false; 8];
            for &sequence in order.iter() {
                arrived[sequence as usize] = true;
                if f.publish(sequence, sequence).unwrap() {
                    drain(&mut f, &mut out);
                }
                let expected = arrived.iter().take_while(|&&v| v).count();
                assert_eq!(out.len(), expected, "delivery order {order:?}");
                assert_eq!(f.retired_prefix(), expected as u64);
            }
            assert_eq!(out, (0..8).collect::<Vec<_>>());
            *cases += 1;
        }
        let mut cases = 0;
        visit(&mut [0, 1, 2, 3, 4, 5, 6, 7], 0, &mut cases);
        assert_eq!(cases, 40320);
        println!("FRONTIER_PERMUTATIONS cases={cases} prefix_oracle=true");
    }

    #[test]
    fn one_slot_wraps_only_after_finish_and_relinquishes_each_drainer() {
        let mut f = CompletionFrontier::<u64, 1>::new().unwrap();
        for sequence in 0..32 {
            assert!(f.publish(sequence, sequence).unwrap());
            assert_eq!(
                f.pop_ready_or_release_drainer().unwrap(),
                Some((sequence, sequence))
            );
            assert_eq!(f.retired_prefix(), sequence);
            assert_eq!(
                f.publish(sequence + 1, sequence + 1).unwrap_err().reason,
                FrontierError::BeyondWindow
            );
            f.finish(sequence).unwrap();
            assert_eq!(f.pop_ready_or_release_drainer().unwrap(), None);
            assert_eq!(
                f.pop_ready_or_release_drainer(),
                Err(FrontierError::NoDrainer)
            );
        }
        assert_eq!(f.retired_prefix(), 32);
    }

    #[test]
    fn callback_in_progress_keeps_the_window_until_slot_return() {
        let mut f = CompletionFrontier::<u64, 2>::new().unwrap();
        assert!(f.publish(0, 0).unwrap());
        assert!(!f.publish(1, 1).unwrap());
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), Some((0, 0)));
        assert_eq!(f.next, 1);
        assert_eq!(f.retired_prefix(), 0);
        let rejected = f.publish(2, 2).unwrap_err();
        assert_eq!(rejected.reason, FrontierError::BeyondWindow);
        assert_eq!(rejected.value, 2);
        assert_eq!(f.finish(1), Err(FrontierError::WrongRetirement));
        assert_eq!(f.retired_prefix(), 0);
        f.finish(0).unwrap();
        // Returning slot0 now admits sequence2 before the drainer takes head1.
        assert!(!f.publish(2, rejected.value).unwrap());
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), Some((1, 1)));
        assert_eq!(
            f.publish(3, 3).unwrap_err().reason,
            FrontierError::BeyondWindow
        );
        f.finish(1).unwrap();
        assert!(!f.publish(3, 3).unwrap());
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), Some((2, 2)));
        f.finish(2).unwrap();
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), Some((3, 3)));
        f.finish(3).unwrap();
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), None);
        assert_eq!(f.retired_prefix(), 4);
    }

    #[test]
    fn publication_during_callback_and_after_hole_elects_exactly_one_drainer() {
        let mut f = CompletionFrontier::<u64, 3>::new().unwrap();
        assert!(f.publish(0, 0).unwrap());
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), Some((0, 0)));
        assert!(!f.publish(2, 2).unwrap());
        assert!(!f.publish(1, 1).unwrap());
        f.finish(0).unwrap();
        let mut out = vec![0];
        drain(&mut f, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
        assert_eq!(
            f.pop_ready_or_release_drainer(),
            Err(FrontierError::NoDrainer)
        );
        assert!(!f.publish(4, 4).unwrap());
        assert!(f.publish(3, 3).unwrap());
        drain(&mut f, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4]);
    }

    #[derive(Debug)]
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn every_rejection_returns_resource_ownership_without_early_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = || Owner(drops.clone());
        let mut f = CompletionFrontier::<Owner, 2>::new().unwrap();
        assert!(f.publish(0, owner()).unwrap());
        for (seq, expected) in [
            (0, FrontierError::Duplicate),
            (2, FrontierError::BeyondWindow),
            (u64::MAX, FrontierError::SequenceExhausted),
        ] {
            let before = drops.load(Ordering::Relaxed);
            let rejected = f.publish(seq, owner()).unwrap_err();
            assert_eq!(rejected.reason, expected);
            assert_eq!(drops.load(Ordering::Relaxed), before);
            drop(rejected.value);
            assert_eq!(drops.load(Ordering::Relaxed), before + 1);
        }
        let (_, active) = f.pop_ready_or_release_drainer().unwrap().unwrap();
        let rejected = f.publish(0, owner()).unwrap_err();
        assert_eq!(rejected.reason, FrontierError::Duplicate);
        f.finish(0).unwrap();
        let stale = f.publish(0, owner()).unwrap_err();
        assert_eq!(stale.reason, FrontierError::BeforeFrontier);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
        drop((rejected.value, stale.value, active));
        assert_eq!(drops.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn capacity_and_sequence_exhaustion_are_explicit() {
        assert!(matches!(
            CompletionFrontier::<u64, 0>::new(),
            Err(FrontierError::ZeroCapacity)
        ));
        let mut f = CompletionFrontier::<u64, 2>::new().unwrap();
        f.next = u64::MAX - 1;
        f.retired = f.next;
        assert!(f.publish(u64::MAX - 1, 9).unwrap());
        assert_eq!(
            f.pop_ready_or_release_drainer().unwrap(),
            Some((u64::MAX - 1, 9))
        );
        f.finish(u64::MAX - 1).unwrap();
        assert_eq!(f.retired_prefix(), u64::MAX);
        assert_eq!(f.pop_ready_or_release_drainer().unwrap(), None);
        let rejected = f.publish(u64::MAX, 11).unwrap_err();
        assert_eq!(rejected.reason, FrontierError::SequenceExhausted);
        assert_eq!(rejected.value, 11);
        assert_eq!(f.finish(u64::MAX), Err(FrontierError::WrongRetirement));
    }

    #[test]
    fn terminal_errors_do_not_leave_holes_or_bypass_earlier_records() {
        for failed in 0..8 {
            let mut f = CompletionFrontier::<Result<u64, u64>, 8>::new().unwrap();
            for sequence in (1..8).rev() {
                let value = if sequence == failed {
                    Err(sequence)
                } else {
                    Ok(sequence)
                };
                assert!(!f.publish(sequence, value).unwrap());
            }
            assert_eq!(f.retired_prefix(), 0);
            assert!(f
                .publish(0, if failed == 0 { Err(0) } else { Ok(0) })
                .unwrap());
            for sequence in 0..8 {
                let (got, value) = f.pop_ready_or_release_drainer().unwrap().unwrap();
                assert_eq!(got, sequence);
                assert_eq!(value.is_err(), sequence == failed);
                f.finish(sequence).unwrap();
            }
            assert_eq!(f.pop_ready_or_release_drainer().unwrap(), None);
            assert_eq!(f.retired_prefix(), 8);
        }
    }

    #[test]
    fn concurrent_publishers_share_one_nonblocking_drainer() {
        for _ in 0..32 {
            let f = Arc::new(Mutex::new(CompletionFrontier::<u64, 8>::new().unwrap()));
            let output = Arc::new(Mutex::new(Vec::new()));
            let mut workers = Vec::new();
            for sequence in (0..8).rev() {
                let f = f.clone();
                let output = output.clone();
                workers.push(std::thread::spawn(move || {
                    let elected = f.lock().publish(sequence, sequence).unwrap();
                    if elected {
                        loop {
                            let record = f.lock().pop_ready_or_release_drainer().unwrap();
                            let Some((sequence, value)) = record else {
                                break;
                            };
                            std::thread::yield_now();
                            output.lock().push(value);
                            f.lock().finish(sequence).unwrap();
                        }
                    }
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
            assert_eq!(*output.lock(), (0..8).collect::<Vec<_>>());
            assert_eq!(f.lock().retired_prefix(), 8);
        }
    }
}
