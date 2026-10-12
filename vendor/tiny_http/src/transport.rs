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
    #[cfg(test)]
    clock: Mutex<Option<Instant>>,
}

/// Cloning this handle retains only the exact accepted origin, not a model.
/// Cancel closes both directions and makes controlled buffered/sequential reads
/// fail even if unread bytes remain locally buffered. It does not join threads.
#[derive(Clone)]
pub struct ConnectionCancellation {
    origin: Arc<Origin>,
}
impl ConnectionCancellation {
    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = *self.origin.clock.lock() {
            return now;
        }
        Instant::now()
    }

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
                #[cfg(test)]
                clock: Mutex::new(None),
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
            .is_some_and(|deadline| self.now() >= deadline);
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
        let remaining = deadline.saturating_duration_since(self.now());
        if remaining.is_zero() {
            return Err(self.timeout());
        }
        Ok(remaining)
    }
    pub(crate) fn prepare_io(&self, read: bool) -> io::Result<()> {
        let timeout = self.remaining(self.now() + self.origin.policy.io_timeout)?;
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
        let deadline = self.now() + self.origin.policy.disposal_timeout;
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
        let deadline = self.now() + self.origin.policy.io_timeout;
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
        let deadline = self.now() + self.origin.policy.io_timeout;
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

#[cfg(test)]
mod origin_controls {
    use super::*;
    use crate::connection::ConnectionProbe;
    use crate::util::{EqualReader, RefinedTcpStream, SequentialReaderBuilder};
    use crate::{HTTPVersion, Header, Method};
    use std::io::{Cursor, Read, Write};
    use std::sync::atomic::AtomicUsize;

    fn fixture(cap: usize) -> (ConnectionCancellation, Arc<ConnectionProbe>, Instant) {
        let probe = Arc::new(ConnectionProbe::default());
        let control = ConnectionCancellation::new(
            &Connection::Probe(probe.clone()),
            TransportPolicy {
                io_timeout: Duration::from_millis(100),
                disposal_timeout: Duration::from_millis(50),
                max_discard_bytes: cap,
            },
        )
        .unwrap();
        let base = Instant::now();
        *control.origin.clock.lock() = Some(base);
        (control, probe, base)
    }
    fn advance(control: &ConnectionCancellation, time: Instant) {
        *control.origin.clock.lock() = Some(time);
    }
    struct CountReader {
        input: Cursor<Vec<u8>>,
        calls: Arc<AtomicUsize>,
        bytes: Arc<AtomicUsize>,
    }
    impl Read for CountReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let count = self.input.read(buf)?;
            self.bytes.fetch_add(count, Ordering::Relaxed);
            Ok(count)
        }
    }
    fn counted(bytes: Vec<u8>) -> (CountReader, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::new(AtomicUsize::new(0));
        (
            CountReader {
                input: Cursor::new(bytes),
                calls: calls.clone(),
                bytes: count.clone(),
            },
            calls,
            count,
        )
    }
    #[test]
    fn cancel_refusal_still_gates_same_origin_buffered_and_native_reads() {
        let (control, probe, _) = fixture(4);
        assert_eq!(
            probe.state.lock().read_timeout,
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            probe.state.lock().write_timeout,
            Some(Duration::from_millis(100))
        );
        match &control.origin.socket {
            Connection::Probe(retained) => assert!(Arc::ptr_eq(retained, &probe)),
            _ => panic!("fixture must retain the original probe identity"),
        }
        probe.state.lock().input = Cursor::new(b"DATA".to_vec());
        probe.state.lock().refuse_shutdown = true;
        let (mut reader, writer) =
            RefinedTcpStream::new_controlled(Connection::Probe(probe.clone()), control.clone());
        assert_eq!(control.cancel().unwrap_err().kind(), ErrorKind::Other);
        assert_eq!(
            control.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        let mut bytes = [0; 4];
        assert_eq!(
            reader.read(&mut bytes).unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert_eq!(
            probe.state.lock().reads,
            0,
            "cancel refusal must not restore read authority"
        );
        assert_eq!(probe.state.lock().shutdowns, vec![Shutdown::Both]);
        drop(reader);
        drop(writer);
        let (buffered_control, buffered_probe, _) = fixture(4);
        buffered_probe.state.lock().input = Cursor::new(b"DATA".to_vec());
        let (reader, writer) = RefinedTcpStream::new_controlled(
            Connection::Probe(buffered_probe.clone()),
            buffered_control.clone(),
        );
        let mut builder = SequentialReaderBuilder::with_control(
            std::io::BufReader::with_capacity(4, reader),
            Some(buffered_control.clone()),
        );
        let mut sequence = builder.next().unwrap();
        let mut first = [0];
        sequence.read_exact(&mut first).unwrap();
        assert_eq!(first, [b'D']);
        assert_eq!(
            buffered_probe.state.lock().input.position(),
            4,
            "actual BufReader prefetched all four fixture bytes before cancellation"
        );
        assert_eq!(buffered_probe.state.lock().reads, 1);
        buffered_control.cancel().unwrap();
        assert_eq!(
            sequence.read(&mut first).unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert_eq!(
            buffered_probe.state.lock().reads,
            1,
            "buffered cancellation must not reach the underlying source again"
        );
        drop(sequence);
        drop(writer);
    }
    #[test]
    fn nested_disposal_keeps_earliest_deadline_and_cancellation_after_guard_drop() {
        let (control, probe, base) = fixture(4);
        let first = control.begin_disposal();
        advance(&control, base + Duration::from_millis(20));
        let second = control.begin_disposal();
        assert_eq!(
            control.origin.disposal.lock().deadline,
            Some(base + Duration::from_millis(50))
        );
        assert_eq!(
            control
                .remaining(base + Duration::from_millis(100))
                .unwrap(),
            Duration::from_millis(30)
        );
        drop(first);
        assert_eq!(control.origin.disposal.lock().users, 1);
        assert_eq!(
            control.origin.disposal.lock().deadline,
            Some(base + Duration::from_millis(50))
        );
        advance(&control, base + Duration::from_millis(50));
        assert_eq!(control.check().unwrap_err().kind(), ErrorKind::TimedOut);
        assert_eq!(probe.state.lock().shutdowns, vec![Shutdown::Both]);
        drop(second);
        assert_eq!(control.origin.disposal.lock().deadline, None);
        assert_eq!(
            control.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        let (fresh, _, base) = fixture(4);
        drop(fresh.begin_disposal());
        advance(&fresh, base + Duration::from_secs(1));
        assert!(
            fresh.check().is_ok(),
            "completed guard must not leave an active stale deadline"
        );
    }
    #[test]
    fn actual_handoff_accepts_ready_value_but_refuses_cancel_and_disconnect() {
        let (control, _, _) = fixture(4);
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(42).unwrap();
        assert_eq!(control.receive(&rx).unwrap(), 42);
        tx.send(77).unwrap();
        control.cancel().unwrap();
        assert_eq!(
            control.receive(&rx).unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            77,
            "cancelled receive must not consume later ownership"
        );
        let (fresh, _, _) = fixture(4);
        let (tx, rx) = std::sync::mpsc::channel::<u8>();
        drop(tx);
        assert_eq!(
            fresh.receive(&rx).unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert_eq!(
            fresh.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
    }
    #[test]
    fn actual_sequential_body_drop_preserves_valid_pipeline_and_revokes_excess_handoff() {
        let (control, _, _) = fixture(4);
        let mut builder = SequentialReaderBuilder::with_control(
            Cursor::new(b"abcNEXT".to_vec()),
            Some(control.clone()),
        );
        let first = builder.next().unwrap();
        let mut next = builder.next().unwrap();
        let (mut body, signal) = EqualReader::new_controlled(first, 3, Some(control.clone()));
        let mut byte = [0];
        body.read_exact(&mut byte).unwrap();
        assert_eq!(byte, [b'a']);
        drop(body);
        assert!(signal.try_recv().unwrap().is_ok());
        let mut tail = [0; 4];
        next.read_exact(&mut tail).unwrap();
        assert_eq!(&tail, b"NEXT");
        let (control, _, _) = fixture(2);
        let (source, calls, bytes) = counted(b"abcNEXT".to_vec());
        let mut builder = SequentialReaderBuilder::with_control(source, Some(control.clone()));
        let first = builder.next().unwrap();
        let mut next = builder.next().unwrap();
        let (body, signal) = EqualReader::new_controlled(first, 3, Some(control.clone()));
        drop(body);
        assert_eq!(
            signal.try_recv().unwrap().unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "known excess must refuse before draining"
        );
        assert_eq!(bytes.load(Ordering::Relaxed), 0);
        next.assert_closed_handoff();
        assert_eq!(
            next.read(&mut [0; 4]).unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "cancelled next header cannot reuse undrained source"
        );
    }
    #[test]
    fn actual_request_unknown_body_uses_one_probe_byte_then_cancels() {
        let (control, _, _) = fixture(3);
        let (source, calls, bytes) = counted(b"abcdNEXT".to_vec());
        let mut request = crate::request::new_request_with_control(
            false,
            Method::Post,
            "/".into(),
            HTTPVersion(1, 1),
            vec![Header::from_bytes("Connection", "upgrade").unwrap()],
            None,
            source,
            io::sink(),
            Some(control.clone()),
        )
        .unwrap();
        assert_eq!(
            request.discard_body().unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            bytes.load(Ordering::Relaxed),
            4,
            "unknown length cap permits exactly one excess probe byte"
        );
        drop(request);
        assert_eq!(
            bytes.load(Ordering::Relaxed),
            4,
            "request Drop must not restart canceled disposal"
        );
        assert_eq!(
            control.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
    }
    struct LateWriter {
        control: ConnectionCancellation,
        time: Instant,
        calls: Arc<AtomicUsize>,
    }
    impl Write for LateWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            advance(&self.control, self.time);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn actual_request_drop_budget_starts_before_error_response_and_field_disposal() {
        let (control, probe, base) = fixture(2048);
        let (source, reads, _) = counted(vec![b'x'; 1025]);
        let writes = Arc::new(AtomicUsize::new(0));
        let writer = LateWriter {
            control: control.clone(),
            time: base + Duration::from_millis(50),
            calls: writes.clone(),
        };
        let request = crate::request::new_request_with_control(
            false,
            Method::Post,
            "/".into(),
            HTTPVersion(1, 1),
            vec![Header::from_bytes("Content-Length", "1025").unwrap()],
            None,
            source,
            writer,
            Some(control.clone()),
        )
        .unwrap();
        drop(request);
        assert!(
            writes.load(Ordering::Relaxed) > 0,
            "actual automatic error response must reach writer"
        );
        assert_eq!(
            reads.load(Ordering::Relaxed),
            0,
            "expired outer disposal must prevent hidden field drain"
        );
        assert_eq!(
            control.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert!(!probe.state.lock().shutdowns.is_empty());
        assert_eq!(control.origin.disposal.lock().users, 0);
        assert_eq!(control.origin.disposal.lock().deadline, None);
    }
}

// Proposed ignored controls only. No socket execution is authorized by this file.
#[cfg(test)]
mod controlled_loopback {
    use super::*;
    use crate::client::ClientConnection;
    use crate::connection::Connection;
    use crate::util::RefinedTcpStream;
    use crate::Response;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    fn pair(cap: usize) -> (ClientConnection, TcpStream, ConnectionCancellation) {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect_timeout(&address, Duration::from_secs(1)).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        client
            .set_write_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let until = Instant::now() + Duration::from_secs(1);
        let accepted = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    assert!(Instant::now() < until, "owned localhost acceptance bound");
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("owned localhost accept failed: {}", error),
            }
        };
        // The bounded accept loop needs a nonblocking listener, but controlled
        // origin I/O uses blocking sockets with finite read/write timeouts.
        let blocking = accepted.set_nonblocking(false);
        eprintln!("actual accepted socket set_nonblocking(false) receipt={blocking:?}");
        blocking.unwrap();
        let connection = Connection::from(accepted);
        let control = ConnectionCancellation::new(
            &connection,
            TransportPolicy {
                io_timeout: Duration::from_millis(200),
                disposal_timeout: Duration::from_millis(100),
                max_discard_bytes: cap,
            },
        )
        .unwrap();
        let (read, write) = RefinedTcpStream::new_controlled(connection, control.clone());
        (
            ClientConnection::new(write, read, Some(control.clone())),
            client,
            control,
        )
    }
    // Failure observation only: one bounded peer read, preserving the original expectation.
    fn observed_next(
        origin: &mut ClientConnection,
        peer: &mut TcpStream,
        control: &ConnectionCancellation,
        expectation: &str,
    ) -> crate::Request {
        let result = origin.next();
        if result.is_none() {
            let mut response = [0u8; 8193];
            let receipt = peer.read(&mut response);
            eprintln!(
                "actual failed origin expectation={expectation}; gate={:?}; peer read={receipt:?}",
                control.check()
            );
            if let Ok(count) = receipt {
                eprintln!("bounded peer bytes={:?}", &response[..count]);
            }
        }
        result.expect(expectation)
    }
    #[test]
    #[ignore = "socket fixture requires separate bounded execution review"]
    fn actual_origin_small_body_retains_two_request_pipeline() {
        let (mut origin, mut peer, control) = pair(4096);
        peer.write_all(b"POST /one HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\n\r\nabcGET /two HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").unwrap();
        let mut first = observed_next(&mut origin, &mut peer, &control, "first actual request");
        assert_eq!(first.url(), "/one");
        let mut body = [0; 3];
        first.as_reader().read_exact(&mut body).unwrap();
        assert_eq!(&body, b"abc");
        first.respond(Response::from_string("one")).unwrap();
        let second = observed_next(&mut origin, &mut peer, &control, "second actual request");
        assert_eq!(second.url(), "/two");
        second.respond(Response::from_string("two")).unwrap();
        assert!(origin.next().is_none());
        drop(origin);
        let mut wire = Vec::new();
        Read::by_ref(&mut peer)
            .take(8193)
            .read_to_end(&mut wire)
            .unwrap();
        assert!(wire.len() <= 8192);
        let wire = String::from_utf8(wire).unwrap();
        assert_eq!(wire.matches("HTTP/1.1 200 OK").count(), 2);
        assert!(wire.contains("\r\n\r\noneHTTP/1.1 200 OK"));
        assert!(wire.ends_with("\r\n\r\ntwo"));
        assert!(
            control.check().is_ok(),
            "successful pipeline must not cancel the retained origin gate"
        );
    }
    #[test]
    #[ignore = "socket fixture requires separate bounded execution review"]
    fn actual_origin_incomplete_body_drop_cancels_with_absolute_disposal() {
        let (mut origin, mut peer, control) = pair(16);
        peer.write_all(b"POST /held HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2048\r\nExpect: 100-continue\r\n\r\n").unwrap();
        let request = observed_next(
            &mut origin,
            &mut peer,
            &control,
            "large unprebuffered request",
        );
        assert_eq!(request.url(), "/held");
        let started = Instant::now();
        drop(request);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "coarse retained bound, not latency claim"
        );
        assert_eq!(
            control.check().unwrap_err().kind(),
            ErrorKind::ConnectionAborted
        );
        assert!(origin.next().is_none());
        drop(origin);
        let mut wire = Vec::new();
        Read::by_ref(&mut peer)
            .take(8193)
            .read_to_end(&mut wire)
            .unwrap();
        assert!(wire.len() <= 8192);
        assert!(String::from_utf8(wire)
            .unwrap()
            .starts_with("HTTP/1.1 500 Internal Server Error"));
    }
    #[test]
    #[ignore = "socket fixture requires separate bounded execution review"]
    fn actual_origin_cancel_refuses_following_header_without_prefetch_claim() {
        let (mut origin, mut peer, control) = pair(4096);
        peer.write_all(b"GET /first HTTP/1.1\r\nHost: localhost\r\n\r\nGET /never HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").unwrap();
        let request = observed_next(&mut origin, &mut peer, &control, "first actual request");
        assert_eq!(request.url(), "/first");
        let request_control = request
            .cancellation_handle()
            .expect("actual accepted origin control");
        assert!(Arc::ptr_eq(&request_control.origin, &control.origin));
        request_control.cancel().unwrap();
        drop(request);
        assert!(
            origin.next().is_none(),
            "canceled origin cannot produce another request"
        );
        drop(origin);
    }
}
