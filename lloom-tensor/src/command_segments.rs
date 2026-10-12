//! Ordered native segments behind one logical submission. Native callbacks may
//! arrive in any order; a logical feedback is emitted only after every segment.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

/// The same queue transaction submits each recording and its following event.
/// This function does not acquire a queue lock: its caller owns that transaction.
pub(crate) fn dispatch<E>(
    segments: usize,
    events: &[E],
    position: impl Fn(&E) -> usize,
    mut segment: impl FnMut(usize),
    mut event: impl FnMut(&E),
) {
    let mut next = 0;
    for index in 0..segments {
        segment(index);
        while next < events.len() && position(&events[next]) == index {
            event(&events[next]);
            next += 1;
        }
    }
    assert_eq!(next, events.len(), "event outside recording segments");
}

pub(crate) struct FeedbackJoin<T> {
    values: Vec<OnceLock<T>>,
    remaining: AtomicUsize,
}
impl<T> FeedbackJoin<T> {
    pub(crate) fn try_new(count: usize) -> Result<Self, &'static str> {
        if count == 0 {
            return Err("empty feedback join");
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| "segment feedback allocation refused")?;
        values.resize_with(count, OnceLock::new);
        Ok(Self {
            values,
            remaining: AtomicUsize::new(count),
        })
    }
    /// Only the final distinct publisher receives the initialized values. A
    /// duplicate never decrements the count; a missing callback never completes.
    pub(crate) fn publish(
        &self,
        index: usize,
        value: T,
    ) -> Result<Option<&[OnceLock<T>]>, &'static str> {
        let cell = self
            .values
            .get(index)
            .ok_or("invalid segment feedback index")?;
        cell.set(value).map_err(|_| "duplicate segment feedback")?;
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            Ok(Some(&self.values))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    #[test]
    fn event_positions_and_other_recording_are_preserved() {
        let log = RefCell::new(Vec::new());
        let events = [(0, "signal:E"), (1, "wait:F"), (2, "terminal:G")];
        // Recording A's boundaries has no queue side effect. B commits first.
        assert!(log.borrow().is_empty());
        dispatch::<(usize, &str)>(
            1,
            &[],
            |e| e.0,
            |_| log.borrow_mut().push("B"),
            |_| unreachable!(),
        );
        dispatch(
            3,
            &events,
            |e| e.0,
            |n| log.borrow_mut().push(["A0", "A1", "A2"][n]),
            |e| log.borrow_mut().push(e.1),
        );
        assert_eq!(
            *log.borrow(),
            ["B", "A0", "signal:E", "A1", "wait:F", "A2", "terminal:G"]
        );
    }
    #[test]
    fn all_feedback_orders_join_once_and_reject_duplicates() {
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let join = FeedbackJoin::try_new(3).unwrap();
            for (step, index) in order.into_iter().enumerate() {
                let result = join.publish(index, index + 10).unwrap();
                assert_eq!(
                    result.is_some(),
                    step == 2,
                    "last-index callback is not all callbacks"
                );
                assert!(join.publish(index, 999).is_err());
                if let Some(values) = result {
                    assert_eq!(
                        values.iter().map(|v| *v.get().unwrap()).collect::<Vec<_>>(),
                        [10, 11, 12]
                    );
                }
            }
        }
    }
    #[test]
    fn missing_invalid_and_empty_feedback_never_complete() {
        assert!(FeedbackJoin::<usize>::try_new(0).is_err());
        let join = FeedbackJoin::try_new(2).unwrap();
        assert!(join.publish(2, 2).is_err());
        assert!(join.publish(1, 1).unwrap().is_none());
        assert!(join.publish(1, 1).is_err());
        assert!(join.publish(0, 0).unwrap().is_some());
    }
}
