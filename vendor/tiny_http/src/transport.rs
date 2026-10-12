//! Opt-in plain-HTTP origin cancellation and bounded unread-body disposal.
//!
//! These controls do not bound application response producers, header storage,
//! connection/task cardinality, or join the server's dependency threads.
use crate::connection::Connection;
use parking_lot::Mutex;
use std::io::{self, ErrorKind};
use std::net::Shutdown;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Explicit opt-in limits. Legacy constructors remain callable without these
/// time/work limits; all EqualReader destructors use a fixed-size workspace.
#[derive(Clone, Copy, Debug)]
pub struct TransportPolicy {
    /// Maximum duration of one socket I/O or one sequential ownership wait.
    /// A long-running earlier application request can therefore cause a later
    /// pipelined request to time out. This is not an application execution limit.
    pub io_timeout: Duration,
    /// Absolute time budget starting at unread-body disposal (including the
    /// automatic error response on Request Drop).
    pub disposal_timeout: Duration,
    /// Accepts at most this many unread disposal bytes. An unknown-length body
    /// may consume one extra probe byte to detect excess, then closes origin.
    /// EqualReader Drop knows the exact remainder and refuses excess directly.
    pub max_discard_bytes: usize,
}
impl TransportPolicy {
    pub fn validate(self) -> io::Result<Self> {
        if self.io_timeout.is_zero()
            || self.disposal_timeout.is_zero()
            || self.max_discard_bytes == 0
            || self.max_discard_bytes == usize::MAX
            || Instant::now().checked_add(self.io_timeout).is_none()
            || Instant::now().checked_add(self.disposal_timeout).is_none()
        {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "invalid transport limits",
            ));
        }
        Ok(self)
    }
}

struct Disposal {
    users: usize,
    deadline: Option<Instant>,
}
struct Origin {
    /// A clone of THIS accepted connection, never a peer-address lookup.
    socket: Connection,
    policy: TransportPolicy,
    cancelled: AtomicBool,
    disposal: Mutex<Disposal>,
}

/// Cloning this handle retains only the exact accepted origin, not a model.
/// Cancel closes both directions and makes controlled buffered/sequential reads
/// fail even if unread bytes remain locally buffered. It does not join threads.
#[derive(Clone)]
pub struct ConnectionCancellation {
    origin: Arc<Origin>,
}
impl ConnectionCancellation {
    pub(crate) fn new(socket: &Connection, policy: TransportPolicy) -> io::Result<Self> {
        let policy = policy.validate()?;
        let socket = socket.try_clone()?;
        socket.set_read_timeout(Some(policy.io_timeout))?;
        socket.set_write_timeout(Some(policy.io_timeout))?;
        Ok(Self {
            origin: Arc::new(Origin {
                socket,
                policy,
                cancelled: AtomicBool::new(false),
                disposal: Mutex::new(Disposal {
                    users: 0,
                    deadline: None,
                }),
            }),
        })
    }
    pub fn cancel(&self) -> io::Result<()> {
        self.origin.cancelled.store(true, Ordering::Release);
        self.origin.socket.shutdown(Shutdown::Both)
    }
    pub(crate) fn policy(&self) -> TransportPolicy {
        self.origin.policy
    }
    pub(crate) fn check(&self) -> io::Result<()> {
        if self.origin.cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                ErrorKind::ConnectionAborted,
                "origin cancelled",
            ));
        }
        let expired = self
            .origin
            .disposal
            .lock()
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline);
        if expired {
            return Err(self.timeout());
        }
        Ok(())
    }
    fn timeout(&self) -> io::Error {
        let _ = self.cancel();
        io::Error::new(ErrorKind::TimedOut, "transport budget exhausted")
    }
    pub(crate) fn remaining(&self, deadline: Instant) -> io::Result<Duration> {
        self.check()?;
        let active = self.origin.disposal.lock().deadline;
        let deadline = active.map_or(deadline, |active| active.min(deadline));
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(self.timeout());
        }
        Ok(remaining)
    }
    pub(crate) fn prepare_io(&self, read: bool) -> io::Result<()> {
        let timeout = self.remaining(Instant::now() + self.origin.policy.io_timeout)?;
        if read {
            self.origin.socket.set_read_timeout(Some(timeout))
        } else {
            self.origin.socket.set_write_timeout(Some(timeout))
        }
    }
    pub(crate) fn after_io<T>(&self, result: io::Result<T>) -> io::Result<T> {
        match result {
            Err(error) if error.kind() == ErrorKind::Interrupted => Err(error),
            Err(error) => {
                let _ = self.cancel();
                Err(error)
            }
            Ok(value) => {
                self.check()?;
                Ok(value)
            }
        }
    }
    pub(crate) fn begin_disposal(&self) -> DisposalGuard {
        let deadline = Instant::now() + self.origin.policy.disposal_timeout;
        let mut state = self.origin.disposal.lock();
        // Concurrent/nested disposal never extends an already active budget.
        state.deadline = Some(state.deadline.map_or(deadline, |old| old.min(deadline)));
        state.users += 1;
        drop(state);
        DisposalGuard {
            control: self.clone(),
        }
    }
    pub(crate) fn receive<T>(&self, receiver: &Receiver<T>) -> io::Result<T> {
        let deadline = Instant::now() + self.origin.policy.io_timeout;
        loop {
            let wait = self.remaining(deadline)?.min(Duration::from_millis(10));
            match receiver.recv_timeout(wait) {
                Ok(value) => {
                    self.check()?;
                    return Ok(value);
                }
                Err(RecvTimeoutError::Timeout) => (),
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = self.cancel();
                    return Err(io::Error::new(
                        ErrorKind::ConnectionAborted,
                        "ownership handoff closed",
                    ));
                }
            }
        }
    }
    pub(crate) fn lock_writer<'a, T>(
        &self,
        mutex: &'a std::sync::Mutex<T>,
    ) -> io::Result<std::sync::MutexGuard<'a, T>> {
        let deadline = Instant::now() + self.origin.policy.io_timeout;
        loop {
            self.remaining(deadline)?;
            match mutex.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(std::sync::TryLockError::WouldBlock) => (),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    // Do not resume writes through a poisoned legacy writer.
                    // Controlled mode refuses/cancels; legacy keeps unwrap.
                    let _ = self.cancel();
                    return Err(io::Error::new(
                        ErrorKind::ConnectionAborted,
                        "response writer poisoned",
                    ));
                }
            }
            std::thread::sleep(self.remaining(deadline)?.min(Duration::from_millis(1)));
        }
    }
}
pub(crate) struct DisposalGuard {
    control: ConnectionCancellation,
}
impl Drop for DisposalGuard {
    fn drop(&mut self) {
        let mut state = self.control.origin.disposal.lock();
        state.users -= 1;
        if state.users == 0 {
            state.deadline = None;
        }
    }
}
