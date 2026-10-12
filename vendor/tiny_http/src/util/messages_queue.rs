use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

enum Control<T> {
    Elem(T),
    Unblock,
}

pub struct MessagesQueue<T>
where
    T: Send,
{
    queue: Mutex<VecDeque<Control<T>>>,
    condvar: Condvar,
    // Written only under `queue`'s lock: after close nothing enters and
    // every pop returns None.
    closed: AtomicBool,
    #[cfg(test)]
    timeout_observer: parking_lot::Mutex<Option<std::sync::mpsc::SyncSender<Duration>>>,
}

impl<T> MessagesQueue<T>
where
    T: Send,
{
    pub fn with_capacity(capacity: usize) -> Arc<MessagesQueue<T>> {
        Arc::new(MessagesQueue {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            condvar: Condvar::new(),
            closed: AtomicBool::new(false),
            #[cfg(test)]
            timeout_observer: parking_lot::Mutex::new(None),
        })
    }

    /// Pushes an element to the queue. After `close` the value is dropped.
    pub fn push(&self, value: T) {
        let _ = self.try_push(value);
    }

    /// Pushes an element, or hands it back if the queue is closed.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let mut queue = self.queue.lock().unwrap();
        if self.closed.load(Ordering::Relaxed) {
            return Err(value);
        }
        queue.push_back(Control::Elem(value));
        self.condvar.notify_one();
        Ok(())
    }

    /// Refuses every later push, wakes every waiting pop, and returns what
    /// was queued so the caller decides when it is dropped.
    pub fn close_and_take(&self) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();
        self.closed.store(true, Ordering::Relaxed);
        let taken = queue
            .drain(..)
            .filter_map(|control| match control {
                Control::Elem(value) => Some(value),
                Control::Unblock => None,
            })
            .collect();
        self.condvar.notify_all();
        taken
    }

    /// Unblock one thread stuck in pop loop.
    pub fn unblock(&self) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(Control::Unblock);
        self.condvar.notify_one();
    }

    /// Pops an element. Blocks until one is available.
    /// Returns None in case unblock() was issued.
    pub fn pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();

        loop {
            if self.closed.load(Ordering::Relaxed) {
                return None;
            }
            match queue.pop_front() {
                Some(Control::Elem(value)) => return Some(value),
                Some(Control::Unblock) => return None,
                None => (),
            }

            queue = self.condvar.wait(queue).unwrap();
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Tries to pop an element without blocking.
    pub fn try_pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        match queue.pop_front() {
            Some(Control::Elem(value)) => Some(value),
            Some(Control::Unblock) | None => None,
        }
    }

    fn wait_timeout<'a>(
        &self,
        queue: std::sync::MutexGuard<'a, VecDeque<Control<T>>>,
        duration: Duration,
    ) -> std::sync::LockResult<(
        std::sync::MutexGuard<'a, VecDeque<Control<T>>>,
        std::sync::WaitTimeoutResult,
    )> {
        #[cfg(test)]
        let observer = self.timeout_observer.lock().clone();
        #[cfg(test)]
        if let Some(observer) = observer {
            // The first two observations fit even before the notifier runs.
            // Later spurious waits never block while the actual queue is held.
            let _ = observer.try_send(duration);
        }
        self.condvar.wait_timeout(queue, duration)
    }

    /// Tries to pop an element without blocking
    /// more than the specified timeout duration
    /// or unblock() was issued
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        let mut duration = timeout;
        loop {
            if self.closed.load(Ordering::Relaxed) {
                return None;
            }
            match queue.pop_front() {
                Some(Control::Elem(value)) => return Some(value),
                Some(Control::Unblock) => return None,
                None => (),
            }
            let now = Instant::now();
            let (_queue, result) = self.wait_timeout(queue, duration).unwrap();
            queue = _queue;
            let sleep_time = now.elapsed();
            duration = if duration > sleep_time {
                duration - sleep_time
            } else {
                Duration::from_millis(0)
            };
            if result.timed_out()
                || (duration.as_secs() == 0 && duration.subsec_nanos() < 1_000_000)
            {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod timeout_controls {
    use super::{Control, MessagesQueue};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn actual_empty_wake_waits_only_remaining_duration() {
        let queue = MessagesQueue::<u8>::with_capacity(1);
        let timeout = Duration::from_millis(500);
        let (wait_sender, wait_receiver) = mpsc::sync_channel(2);
        *queue.timeout_observer.lock() = Some(wait_sender);
        let (value, (first, second)) = thread::scope(|scope| {
            let queue_for_wake = &queue;
            let notifier = scope.spawn(move || {
                let first = wait_receiver
                    .recv_timeout(Duration::from_secs(1))
                    .expect("actual first Condvar wait");
                {
                    // Holding the queue lock excludes notification before the
                    // waiting thread has released it to its actual Condvar.
                    let held = queue_for_wake.queue.lock().unwrap();
                    assert!(held.is_empty());
                    thread::sleep(Duration::from_millis(50));
                    queue_for_wake.condvar.notify_one();
                    drop(held);
                }
                let second = wait_receiver
                    .recv_timeout(Duration::from_secs(1))
                    .expect("actual remaining Condvar wait");
                {
                    let mut held = queue_for_wake.queue.lock().unwrap();
                    held.push_back(Control::Elem(7));
                    queue_for_wake.condvar.notify_one();
                }
                (first, second)
            });
            let value = queue.pop_timeout(timeout);
            (value, notifier.join().expect("owned notifier terminated"))
        });
        assert_eq!(
            value,
            Some(7),
            "actual queue must deliver the retained value"
        );
        assert_eq!(first, timeout);
        assert!(
            !second.is_zero() && second < first,
            "empty wake must not restart the full requested timeout"
        );
        println!(
            "ACTUAL_TIMEOUT_WAIT_NS first={} second={}",
            first.as_nanos(),
            second.as_nanos()
        );
    }

    #[test]
    fn actual_timeout_preserves_fifo_unblock_and_zero_budget() {
        let queue = MessagesQueue::<u8>::with_capacity(3);
        queue.push(11);
        queue.unblock();
        queue.push(22);
        assert_eq!(queue.pop_timeout(Duration::ZERO), Some(11));
        assert_eq!(queue.pop_timeout(Duration::ZERO), None);
        assert_eq!(queue.pop_timeout(Duration::ZERO), Some(22));
        assert_eq!(queue.pop_timeout(Duration::ZERO), None);
        assert_eq!(queue.try_pop(), None);
    }
}
