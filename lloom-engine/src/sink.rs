//! Text handed off the decode thread without a syscall, a wait, or a
//! per-token allocation (`docs/invariants.md` §15.2 #10, #10a, #10b).
//!
//! The decode thread produces a fragment per drained batch. Writing it to a
//! terminal or a socket there is a `write(2)` (and a lock) on the thread the
//! GPU waits on, and queueing it through a channel to a parked writer is a
//! wake (`ulock_wake`) per fragment — a syscall either way. So the producer
//! appends to a shared buffer under `try_lock`: uncontended, that is one
//! atomic, and when the consumer happens to hold the lock the fragment goes to
//! a producer-local spill that the next push (or [`FragmentSink::finish`])
//! carries over. Nothing on the producer side waits, wakes anyone, or
//! allocates once the two buffers have grown to the largest batch.
//!
//! **Why the consumer polls (§15.2 #10a):** the only announcement a producer
//! could make is the wake this module exists to avoid. The consumer takes
//! what accumulated every [`DRAIN_PERIOD`], which is below a terminal's frame
//! and a client's perception; a run's text is fully drained at `finish`.
//!
//! **Its scheduling class (§15.2 #10b):** the consumer is not on the GPU's
//! critical path — nothing the device runs waits on it — so it runs at the
//! class `std` spawns it with (`default`), and its wait is a sleep.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;

/// How often a consumer takes the accumulated text.
pub const DRAIN_PERIOD: Duration = Duration::from_millis(2);

/// The buffer the producer appends to and a consumer swaps out.
#[derive(Debug, Default)]
pub struct DeltaBuffer {
    buf: Mutex<String>,
    dirty: AtomicBool,
}

impl DeltaBuffer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Append `frag` (and `spill`, first) without waiting: under `try_lock`,
    /// or into `spill` when a consumer holds the buffer this instant.
    pub fn push(&self, spill: &mut String, frag: &str) {
        let mut g = match self.buf.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
            Err(TryLockError::WouldBlock) => {
                spill.push_str(frag);
                return;
            }
        };
        if !spill.is_empty() {
            g.push_str(spill);
            spill.clear();
        }
        g.push_str(frag);
        drop(g);
        self.dirty.store(true, Ordering::Release);
    }

    /// Carry `spill` over, waiting for the lock if it must: the run's end,
    /// off the per-token path.
    pub fn flush(&self, spill: &mut String) {
        if spill.is_empty() {
            return;
        }
        self.buf
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_str(spill);
        spill.clear();
        self.dirty.store(true, Ordering::Release);
    }

    /// Swap what accumulated into `out` (cleared first; both strings keep
    /// their capacity). Returns whether anything was taken.
    pub fn take_into(&self, out: &mut String) -> bool {
        out.clear();
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return false;
        }
        std::mem::swap(&mut *self.buf.lock().unwrap_or_else(|e| e.into_inner()), out);
        !out.is_empty()
    }
}

/// A [`DeltaBuffer`] with its own consumer thread: `drain` sees the text in
/// order, a batch at a time, and returns `false` to stop taking it (the
/// consumer went away; [`Self::stopped`] reports it).
pub struct FragmentSink<S: Send + 'static> {
    buf: Arc<DeltaBuffer>,
    spill: String,
    done: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<S>>,
}

impl<S: Send + 'static> FragmentSink<S> {
    pub fn start<F>(mut state: S, mut drain: F) -> Self
    where
        F: FnMut(&mut S, &str) -> bool + Send + 'static,
    {
        let buf = DeltaBuffer::new();
        let done = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (b, d, s) = (Arc::clone(&buf), Arc::clone(&done), Arc::clone(&stopped));
        let thread = std::thread::Builder::new()
            .name("fragment-sink".into())
            .spawn(move || {
                let mut local = String::with_capacity(4096);
                loop {
                    // `done` first: the producer raised it after its last
                    // push, so a take after seeing it gets everything.
                    let finished = d.load(Ordering::Acquire);
                    if b.take_into(&mut local) {
                        if !s.load(Ordering::Relaxed) && !drain(&mut state, &local) {
                            s.store(true, Ordering::Release);
                        }
                    } else if finished {
                        return state;
                    } else {
                        std::thread::sleep(DRAIN_PERIOD);
                    }
                }
            })
            .expect("spawn the fragment sink");
        Self {
            buf,
            spill: String::with_capacity(256),
            done,
            stopped,
            thread: Some(thread),
        }
    }

    /// Hand `frag` over (the decode thread's call: no wait, no syscall).
    pub fn push(&mut self, frag: &str) {
        self.buf.push(&mut self.spill, frag);
    }

    /// Whether the consumer stopped taking text.
    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// Drain everything pushed and return the consumer's state.
    pub fn finish(mut self) -> S {
        self.buf.flush(&mut self.spill);
        self.done.store(true, Ordering::Release);
        self.thread
            .take()
            .expect("finished once")
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p))
    }
}

impl<S: Send + 'static> Drop for FragmentSink<S> {
    fn drop(&mut self) {
        if let Some(t) = self.thread.take() {
            self.buf.flush(&mut self.spill);
            self.done.store(true, Ordering::Release);
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fragment_arrives_in_order() {
        let mut sink = FragmentSink::start(String::new(), |acc: &mut String, s: &str| {
            acc.push_str(s);
            true
        });
        let mut want = String::new();
        for i in 0..5000 {
            let f = format!("{i},");
            want.push_str(&f);
            sink.push(&f);
            if i % 997 == 0 {
                std::thread::sleep(Duration::from_millis(3));
            }
        }
        assert_eq!(sink.finish(), want);
    }

    #[test]
    fn a_busy_buffer_spills_and_the_spill_is_carried_over() {
        let b = DeltaBuffer::new();
        let mut spill = String::new();
        let held = b.buf.lock().unwrap();
        b.push(&mut spill, "ab");
        b.push(&mut spill, "c");
        assert_eq!(spill, "abc", "the producer never waits on a held buffer");
        drop(held);
        b.push(&mut spill, "d");
        assert!(spill.is_empty());
        let mut out = String::new();
        assert!(b.take_into(&mut out));
        assert_eq!(out, "abcd");
        assert!(!b.take_into(&mut out), "nothing new");
    }

    #[test]
    fn a_consumer_that_stops_is_reported_and_the_run_still_finishes() {
        let mut sink = FragmentSink::start(0usize, |n: &mut usize, _s: &str| {
            *n += 1;
            false
        });
        sink.push("x");
        while !sink.stopped() {
            std::thread::sleep(Duration::from_millis(1));
        }
        sink.push("y");
        assert_eq!(sink.finish(), 1, "nothing is drained after the stop");
    }
}
