//! Allocation-free dynamic callback scope. A recoverable wait refusal tests
//! every enclosing queue, not only the innermost callback.
use std::cell::Cell;

struct Frame {
    queue: usize,
    previous: *const Frame,
}
thread_local! { static HEAD: Cell<*const Frame> = const { Cell::new(std::ptr::null()) }; }
struct Restore<'a> {
    head: &'a Cell<*const Frame>,
    frame: &'a Frame,
}
impl Drop for Restore<'_> {
    fn drop(&mut self) {
        self.head.set(self.frame.previous);
    }
}

pub(crate) fn with_callback<R>(queue: usize, call: impl FnOnce() -> R) -> R {
    HEAD.with(|head| {
        let frame = Frame {
            queue,
            previous: head.get(),
        };
        let _restore = Restore {
            head,
            frame: &frame,
        };
        head.set(&frame);
        call()
    })
}

pub(crate) fn contains(queue: usize) -> bool {
    HEAD.with(|head| {
        let mut cursor = head.get();
        while !cursor.is_null() {
            // SAFETY: HEAD is thread-local. Each installed Frame is borrowed by
            // its stack guard until that guard restores previous, including Rust
            // unwind. No pointer escapes the dynamic scope or crosses threads.
            let frame = unsafe { &*cursor };
            if frame.queue == queue {
                return true;
            }
            cursor = frame.previous;
        }
        false
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReentrantWait;
/// Called only after the ordinary readiness attempt, immediately before a park.
/// Available capacity/completed work remains usable even inside callbacks.
pub(crate) fn before_park(queue: usize) -> Result<(), ReentrantWait> {
    if contains(queue) {
        Err(ReentrantWait)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn nested_queues_reject_outer_and_inner_blocking_reentry() {
        assert_eq!(before_park(11), Ok(()));
        with_callback(11, || {
            assert_eq!(before_park(11), Err(ReentrantWait));
            with_callback(22, || {
                assert_eq!(before_park(11), Err(ReentrantWait));
                assert_eq!(before_park(22), Err(ReentrantWait));
                assert_eq!(before_park(33), Ok(()));
            });
            assert_eq!(before_park(22), Ok(()));
            assert_eq!(before_park(11), Err(ReentrantWait));
        });
        assert_eq!(before_park(11), Ok(()));
    }

    #[test]
    fn unwinding_nested_callback_restores_outer_scope() {
        with_callback(11, || {
            let result = catch_unwind(AssertUnwindSafe(|| with_callback(22, || panic!("control"))));
            assert!(result.is_err());
            assert_eq!(before_park(22), Ok(()));
            assert_eq!(before_park(11), Err(ReentrantWait));
        });
        assert_eq!(before_park(11), Ok(()));
    }
}
