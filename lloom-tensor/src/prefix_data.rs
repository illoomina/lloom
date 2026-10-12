//! Recovery-only prefix sequencing. Readiness of an owned submission is not a
//! receipt for earlier GPU work; callbacks and physical teardown are distinct.

pub(crate) fn wait_prefix<E>(
    admit: impl FnOnce() -> Result<(), E>,
    prefix: impl FnOnce() -> Result<(), E>,
    own: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    admit()?;
    prefix()?;
    own()
}

/// A true receipt observation or successful native event wait is the only
/// completion path. A timed-out wait only permits another failure/reentry check.
/// Caller supplies a bounded native wait and retains the event/generation for
/// the entire call. Own completion must not occur in the readiness predicate.
pub(crate) fn wait_front<E>(
    mut ready: impl FnMut() -> bool,
    mut before_wait: impl FnMut() -> Result<(), E>,
    mut wait: impl FnMut() -> bool,
) -> Result<(), E> {
    loop {
        if ready() {
            return Ok(());
        }
        before_wait()?;
        if wait() {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn own_completed_does_not_skip_old_generation_front_receipt() {
        let event = Cell::new(2u64);
        let generation = 3;
        let own_completed = true;
        let callbacks_finished = false;
        let waits = Cell::new(0);
        let checks = Cell::new(0);
        let own_reads = Cell::new(0);
        let result = wait_prefix(
            || Ok::<(), &'static str>(()),
            || {
                wait_front(
                    || event.get() >= generation,
                    || {
                        checks.set(checks.get() + 1);
                        Ok(())
                    },
                    || {
                        waits.set(waits.get() + 1);
                        if waits.get() == 3 {
                            event.set(generation);
                            true
                        } else {
                            false
                        }
                    },
                )
            },
            || {
                assert!(
                    event.get() >= generation,
                    "prefix receipt omitted before own data"
                );
                assert!(own_completed);
                assert!(!callbacks_finished, "control accidentally joined callbacks");
                own_reads.set(own_reads.get() + 1);
                Ok(())
            },
        );
        assert_eq!(result, Ok(()));
        assert_eq!(
            waits.get(),
            3,
            "old generation or timeout accepted as current prefix"
        );
        assert_eq!(checks.get(), 3);
        assert_eq!(own_reads.get(), 1);
    }

    #[test]
    fn unknown_handoff_refuses_even_when_front_and_own_are_ready() {
        use crate::command_lifecycle::{CommandLifecycle, LifecycleError};
        let gate = CommandLifecycle::new();
        let mut ticket = gate.prepare().unwrap();
        gate.entered_native(&mut ticket).unwrap();
        gate.abandon(&mut ticket).unwrap();
        let visited = Cell::new(0);
        let result = wait_prefix(
            || gate.check_wait_submission(),
            || {
                visited.set(visited.get() + 1);
                Ok(())
            },
            || {
                visited.set(visited.get() + 1);
                Ok(())
            },
        );
        assert_eq!(
            result,
            Err(LifecycleError::Abandoned),
            "unknown handoff bypassed admission with terminal readiness"
        );
        assert_eq!(
            visited.get(),
            0,
            "unknown handoff bypassed admission with terminal readiness"
        );
        assert!(matches!(gate.prepare(), Err(LifecycleError::Abandoned)));
        // Positive control uses the same admission helper after real RETURNED.
        let ready = CommandLifecycle::new();
        let mut ticket = ready.prepare().unwrap();
        ready.entered_native(&mut ticket).unwrap();
        ready.handoff_returned(&mut ticket).unwrap();
        assert_eq!(
            wait_prefix(|| ready.check_wait_submission(), || Ok(()), || Ok(())),
            Ok(())
        );
    }

    #[test]
    fn pending_prefix_refuses_callback_self_wait_but_ready_receipt_does_not_park() {
        crate::completion_scope::with_callback(11, || {
            let waits = Cell::new(0);
            let pending = wait_front(
                || false,
                || crate::completion_scope::before_park(11),
                || {
                    waits.set(waits.get() + 1);
                    assert!(
                        waits.get() <= 2,
                        "prefix wait lost bounded refusal progress"
                    );
                    false
                },
            );
            assert_eq!(pending, Err(crate::completion_scope::ReentrantWait));
            assert_eq!(waits.get(), 0, "callback entered native prefix park");
            let ready = wait_front(
                || true,
                || crate::completion_scope::before_park(11),
                || {
                    waits.set(waits.get() + 1);
                    assert!(
                        waits.get() <= 2,
                        "prefix wait lost bounded refusal progress"
                    );
                    false
                },
            );
            assert_eq!(ready, Ok(()));
            assert_eq!(waits.get(), 0);
        });
    }

    #[test]
    fn timed_out_prefix_observes_failure_before_any_own_data_or_reset() {
        let waits = Cell::new(0);
        let checks = Cell::new(0);
        let writes = Cell::new(0);
        let result = wait_prefix(
            || Ok::<(), &'static str>(()),
            || {
                wait_front(
                    || false,
                    || {
                        checks.set(checks.get() + 1);
                        if checks.get() == 2 {
                            Err("unknown queue handoff")
                        } else {
                            Ok(())
                        }
                    },
                    || {
                        waits.set(waits.get() + 1);
                        assert!(waits.get() <= 2, "prefix timeout lost failure observation");
                        false
                    },
                )
            },
            || {
                writes.set(writes.get() + 1);
                Ok(())
            },
        );
        assert_eq!(
            result,
            Err("unknown queue handoff"),
            "timed-out front accepted as prefix completion"
        );
        assert_eq!(waits.get(), 1);
        assert_eq!(checks.get(), 2);
        assert_eq!(writes.get(), 0, "timeout/failure authorized mapped reuse");
    }
}
