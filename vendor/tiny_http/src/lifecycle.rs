//! Bounded `Server` shutdown, and the retirement service that owns whatever a
//! dropped `Server` leaves running.
//!
//! Retiring a server closes intake, shuts the read side of every connection so
//! idle and parsing workers return, and waits up to the budget for the accept
//! thread, the pool's threads and every connection a request or an escaped
//! stream still holds. At expiry the remaining connections are shut in both
//! directions and named in the report; threads still running then belong to
//! the retirement service, which joins them. Nothing is detached.

use crate::connection::{Connection, ListenAddr};
use crate::util::{MessagesQueue, TaskPool};
use crate::Message;
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::collections::VecDeque;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The budget a `Server` dropped without [`crate::Server::shutdown`] gets.
pub const DROP_RETIREMENT_BUDGET: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(2);
const WAKE_RETRY: Duration = Duration::from_millis(50);
const WAKE_CONNECT: Duration = Duration::from_millis(100);
const KEPT_REPORTS: usize = 32;

/// One accepted connection, as the server sees it from outside its threads.
/// The socket is a duplicate of the accepted one: shutting it down reaches
/// every reader, writer, request and escaped stream of that connection.
pub(crate) struct LiveConnection {
    id: u64,
    peer: Option<SocketAddr>,
    socket: Connection,
    read_closed: AtomicBool,
}

impl LiveConnection {
    fn close_read(&self) {
        if !self.read_closed.swap(true, Ordering::AcqRel) {
            let _ = self.socket.shutdown(Shutdown::Read);
        }
    }

    // Each direction separately: on Darwin, Both after an earlier Read
    // shutdown fails with ENOTCONN and leaves the write side open.
    fn cut(&self) -> CutConnection {
        let _ = self.socket.shutdown(Shutdown::Write);
        self.close_read();
        CutConnection {
            id: self.id,
            peer: self.peer,
        }
    }
}

/// Every connection the server accepted that something still uses. Entries
/// are weak: a connection leaves when its last half, request or stream drops.
#[derive(Default)]
pub(crate) struct Connections {
    state: Mutex<ConnectionsState>,
}

#[derive(Default)]
struct ConnectionsState {
    next_id: u64,
    live: Vec<Weak<LiveConnection>>,
}

impl Connections {
    pub(crate) fn register(
        &self,
        socket: &Connection,
        peer: Option<SocketAddr>,
    ) -> io::Result<Arc<LiveConnection>> {
        let socket = socket.try_clone()?;
        let mut state = self.state.lock();
        state.live.retain(|entry| entry.strong_count() > 0);
        state.next_id += 1;
        let live = Arc::new(LiveConnection {
            id: state.next_id,
            peer,
            socket,
            read_closed: AtomicBool::new(false),
        });
        state.live.push(Arc::downgrade(&live));
        Ok(live)
    }

    fn live(&self) -> Vec<Arc<LiveConnection>> {
        self.state
            .lock()
            .live
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }

    pub(crate) fn live_count(&self) -> usize {
        self.state
            .lock()
            .live
            .iter()
            .filter(|entry| entry.strong_count() > 0)
            .count()
    }
}

/// A connection still in use when the shutdown budget expired. Its socket was
/// shut in both directions; whoever holds its request or stream now gets
/// errors instead of the peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutConnection {
    /// The server's sequence number for the connection, from 1.
    pub id: u64,
    /// The peer, when the transport has one (TCP).
    pub peer: Option<SocketAddr>,
}

#[derive(Debug)]
struct Ticket {
    remaining: Mutex<usize>,
    joined: Condvar,
}

impl Ticket {
    fn joined_one(&self) {
        let mut remaining = self.remaining.lock();
        *remaining -= 1;
        if *remaining == 0 {
            self.joined.notify_all();
        }
    }

    fn wait(&self) {
        let mut remaining = self.remaining.lock();
        while *remaining > 0 {
            self.joined.wait(&mut remaining);
        }
    }
}

/// What a shutdown did. `completed` means everything retired inside the
/// budget; otherwise `cut` names the connections closed at expiry and
/// `threads_retiring` counts threads the retirement service now joins.
#[derive(Debug)]
pub struct ShutdownReport {
    /// Every connection released and every thread joined within the budget.
    pub completed: bool,
    /// The budget expired before the server had retired.
    pub timed_out: bool,
    /// Set when this server had already been shut down; nothing else was done.
    pub already_shut_down: bool,
    /// Requests accepted but never received, answered by their drop.
    pub queued_requests_dropped: usize,
    /// Accept and worker threads joined by this call.
    pub threads_joined: usize,
    /// Threads still running at return, owned by the retirement service.
    pub threads_retiring: usize,
    /// Connections shut at budget expiry.
    pub cut: Vec<CutConnection>,
    retiring: Retiring,
}

#[derive(Debug, Default)]
enum Retiring {
    #[default]
    Nothing,
    Service(Arc<Ticket>),
    // The service could not start; the report holds the threads itself.
    Inline(Vec<Straggler>),
}

impl ShutdownReport {
    fn already() -> Self {
        Self {
            completed: false,
            timed_out: false,
            already_shut_down: true,
            queued_requests_dropped: 0,
            threads_joined: 0,
            threads_retiring: 0,
            cut: Vec::new(),
            retiring: Retiring::Nothing,
        }
    }

    /// Blocks until every thread in `threads_retiring` has been joined.
    pub fn wait(mut self) -> Self {
        match std::mem::take(&mut self.retiring) {
            Retiring::Nothing => (),
            Retiring::Service(ticket) => ticket.wait(),
            Retiring::Inline(stragglers) => {
                for straggler in stragglers {
                    straggler.join_blocking();
                }
            }
        }
        self.threads_joined += self.threads_retiring;
        self.threads_retiring = 0;
        self
    }
}

#[derive(Debug)]
pub(crate) struct Straggler {
    handle: JoinHandle<()>,
    // Set for the accept thread: it may need another self-connect to return,
    // and a unix listener's path is removed once it has.
    accept: Option<ListenAddr>,
    last_wake: Instant,
    ticket: Option<Arc<Ticket>>,
}

impl Straggler {
    fn finish(self) {
        let _ = self.handle.join();
        if let Some(addr) = &self.accept {
            remove_unix_path(addr);
        }
        if let Some(ticket) = &self.ticket {
            ticket.joined_one();
        }
    }

    fn join_blocking(self) {
        while !self.handle.is_finished() {
            if let Some(addr) = &self.accept {
                wake_accept(addr);
            }
            std::thread::sleep(WAKE_RETRY);
        }
        self.finish();
    }
}

/// Everything a running `Server` owns besides its queue's readers.
pub(crate) struct Lifecycle {
    pub(crate) close: Arc<AtomicBool>,
    pub(crate) messages: Arc<MessagesQueue<Message>>,
    pub(crate) listening_addr: ListenAddr,
    pub(crate) accept: JoinHandle<()>,
    pub(crate) pool: Arc<TaskPool>,
    pub(crate) connections: Arc<Connections>,
}

impl Lifecycle {
    /// Retires the server by `deadline`, returning the threads that outlived it.
    fn retire(self, deadline: Instant) -> (ShutdownReport, Vec<Straggler>) {
        let Lifecycle {
            close,
            messages,
            listening_addr,
            accept,
            pool,
            connections,
        } = self;
        close.store(true, Ordering::SeqCst);
        pool.close();
        let mut queued = Some(messages.close_and_take());
        let mut queued_requests_dropped = 0;
        wake_accept(&listening_addr);
        let mut last_wake = Instant::now();
        let timed_out = loop {
            // Repeated: a connection accepted while closing registers late.
            for connection in connections.live() {
                connection.close_read();
            }
            // After the read sides are shut, so a request's drop cannot wait
            // on a body the peer never sends.
            if let Some(queued) = queued.take() {
                queued_requests_dropped = queued
                    .iter()
                    .filter(|message| matches!(message, Message::NewRequest(_)))
                    .count();
                drop(queued);
            }
            if accept.is_finished() && pool.all_finished() && connections.live_count() == 0 {
                break false;
            }
            let now = Instant::now();
            if now >= deadline {
                break true;
            }
            if !accept.is_finished() && now.duration_since(last_wake) >= WAKE_RETRY {
                wake_accept(&listening_addr);
                last_wake = Instant::now();
            }
            std::thread::sleep(POLL.min(deadline - now));
        };
        let cut = if timed_out {
            connections.live().iter().map(|c| c.cut()).collect()
        } else {
            Vec::new()
        };
        let mut threads_joined = 0;
        let mut outliving = Vec::new();
        let threads = pool
            .take_handles()
            .into_iter()
            .map(|handle| (handle, None))
            .chain(std::iter::once((accept, Some(listening_addr))));
        for (handle, accept) in threads {
            if handle.is_finished() {
                let _ = handle.join();
                if let Some(addr) = &accept {
                    remove_unix_path(addr);
                }
                threads_joined += 1;
            } else {
                outliving.push((handle, accept));
            }
        }
        let threads_retiring = outliving.len();
        let ticket = (threads_retiring > 0).then(|| {
            Arc::new(Ticket {
                remaining: Mutex::new(threads_retiring),
                joined: Condvar::new(),
            })
        });
        let stragglers = outliving
            .into_iter()
            .map(|(handle, accept)| Straggler {
                handle,
                accept,
                last_wake: Instant::now(),
                ticket: ticket.clone(),
            })
            .collect();
        let report = ShutdownReport {
            completed: !timed_out && threads_retiring == 0,
            timed_out,
            already_shut_down: false,
            queued_requests_dropped,
            threads_joined,
            threads_retiring,
            cut,
            retiring: ticket.map_or(Retiring::Nothing, Retiring::Service),
        };
        (report, stragglers)
    }

    /// Non-blocking last resort when the service cannot run: close, cut every
    /// connection now, and let the threads unwind on their own.
    fn abandon(self) {
        self.close.store(true, Ordering::SeqCst);
        self.pool.close();
        for connection in self.connections.live() {
            connection.cut();
        }
        drop(self.messages.close_and_take());
        log::error!("tiny_http: retirement service unavailable; server threads were not joined");
    }

    fn cut_now(&self) {
        self.close.store(true, Ordering::SeqCst);
        for connection in self.connections.live() {
            connection.cut();
        }
    }
}

/// Bounded shutdown on the calling thread: retire by `deadline`, then give the
/// threads that outlived it to the service.
pub(crate) fn shutdown(lifecycle: Option<Lifecycle>, deadline: Instant) -> ShutdownReport {
    let Some(lifecycle) = lifecycle else {
        return ShutdownReport::already();
    };
    let (mut report, stragglers) = lifecycle.retire(deadline);
    if !stragglers.is_empty() {
        if let Err(stragglers) = hand_over(stragglers) {
            report.retiring = Retiring::Inline(stragglers);
        }
    }
    report
}

/// The dropping thread's whole share of a drop: queue the retirement. A unix
/// listener is woken and its path removed here, as before, so a process that
/// exits right after the drop leaves no socket file behind; both are local
/// operations that do not wait on a peer.
pub(crate) fn retire_in_background(lifecycle: Lifecycle) {
    #[cfg(unix)]
    if matches!(lifecycle.listening_addr, ListenAddr::Unix(_)) {
        wake_accept(&lifecycle.listening_addr);
        remove_unix_path(&lifecycle.listening_addr);
    }
    let deadline = Instant::now() + DROP_RETIREMENT_BUDGET;
    let mut state = SERVICE.state.lock();
    match ensure_worker(&mut state) {
        Ok(()) => {
            state.jobs.push_back((lifecycle, deadline));
            SERVICE.changed.notify_all();
        }
        Err(()) => {
            drop(state);
            lifecycle.abandon();
        }
    }
}

fn hand_over(stragglers: Vec<Straggler>) -> Result<(), Vec<Straggler>> {
    let mut state = SERVICE.state.lock();
    if ensure_worker(&mut state).is_err() {
        return Err(stragglers);
    }
    state.stragglers.extend(stragglers);
    SERVICE.changed.notify_all();
    Ok(())
}

/// What [`join_retirement`] found.
#[derive(Debug)]
pub struct RetirementReport {
    /// The service had nothing left and its thread was joined.
    pub completed: bool,
    /// Dropped servers not yet retired at the deadline; their connections
    /// were cut.
    pub pending_servers: usize,
    /// Threads still running at the deadline, still owned by the service.
    pub threads_retiring: usize,
    /// Reports of servers retired after a drop, oldest first, since the last
    /// call (at most the last 32).
    pub dropped_server_reports: Vec<ShutdownReport>,
}

/// Joins the retirement service, waiting until `deadline` for the work it
/// holds. Call it before the process exits; the service restarts if a server
/// is dropped afterwards. At the deadline, connections of servers it has not
/// reached yet are cut, and whatever is still running stays owned.
pub fn join_retirement(deadline: Instant) -> RetirementReport {
    let mut state = SERVICE.state.lock();
    state.stopping = true;
    SERVICE.changed.notify_all();
    let completed = loop {
        if state.worker.as_ref().is_none_or(JoinHandle::is_finished) {
            if let Some(worker) = state.worker.take() {
                MutexGuard::unlocked(&mut state, || {
                    let _ = worker.join();
                });
            }
            break state.jobs.is_empty() && state.stragglers.is_empty();
        }
        let now = Instant::now();
        if now >= deadline {
            for (lifecycle, _) in &state.jobs {
                lifecycle.cut_now();
            }
            break false;
        }
        SERVICE
            .changed
            .wait_for(&mut state, (deadline - now).min(Duration::from_millis(10)));
    };
    RetirementReport {
        completed,
        pending_servers: state.jobs.len(),
        threads_retiring: state.stragglers.len(),
        dropped_server_reports: state.reports.drain(..).collect(),
    }
}

struct Service {
    state: Mutex<ServiceState>,
    changed: Condvar,
}

struct ServiceState {
    jobs: VecDeque<(Lifecycle, Instant)>,
    stragglers: Vec<Straggler>,
    worker: Option<JoinHandle<()>>,
    stopping: bool,
    reports: VecDeque<ShutdownReport>,
}

static SERVICE: Service = Service {
    state: Mutex::new(ServiceState {
        jobs: VecDeque::new(),
        stragglers: Vec::new(),
        worker: None,
        stopping: false,
        reports: VecDeque::new(),
    }),
    changed: Condvar::new(),
};

fn ensure_worker(state: &mut MutexGuard<'_, ServiceState>) -> Result<(), ()> {
    if let Some(worker) = state.worker.take() {
        if !worker.is_finished() {
            state.worker = Some(worker);
            return Ok(());
        }
        let _ = worker.join();
    }
    state.stopping = false;
    let worker = std::thread::Builder::new()
        .name("tiny-http-retire".into())
        .spawn(service_loop)
        .map_err(|error| log::error!("tiny_http: cannot start retirement service: {error}"))?;
    state.worker = Some(worker);
    Ok(())
}

fn service_loop() {
    let mut state = SERVICE.state.lock();
    loop {
        if let Some((lifecycle, deadline)) = state.jobs.pop_front() {
            let retired = MutexGuard::unlocked(&mut state, || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    lifecycle.retire(deadline)
                }))
            });
            match retired {
                Ok((mut report, stragglers)) => {
                    if !report.completed {
                        log::warn!(
                            "tiny_http: dropped server retired with {} connection(s) cut, {} thread(s) still running",
                            report.cut.len(),
                            report.threads_retiring
                        );
                    }
                    // The service holds these threads itself; the stored
                    // report must not keep their ticket alive.
                    report.retiring = Retiring::Nothing;
                    state.stragglers.extend(stragglers);
                    if state.reports.len() == KEPT_REPORTS {
                        state.reports.pop_front();
                    }
                    state.reports.push_back(report);
                }
                Err(_) => log::error!("tiny_http: a server's retirement panicked"),
            }
            SERVICE.changed.notify_all();
            continue;
        }
        let mut index = 0;
        let mut wake = Vec::new();
        while index < state.stragglers.len() {
            if state.stragglers[index].handle.is_finished() {
                let straggler = state.stragglers.swap_remove(index);
                MutexGuard::unlocked(&mut state, || straggler.finish());
                SERVICE.changed.notify_all();
                continue;
            }
            let straggler = &mut state.stragglers[index];
            if let Some(addr) = &straggler.accept {
                if straggler.last_wake.elapsed() >= WAKE_RETRY {
                    straggler.last_wake = Instant::now();
                    wake.push(addr.clone());
                }
            }
            index += 1;
        }
        if !wake.is_empty() {
            MutexGuard::unlocked(&mut state, || wake.iter().for_each(wake_accept));
            continue;
        }
        if state.stragglers.is_empty() {
            if state.stopping {
                SERVICE.changed.notify_all();
                return;
            }
            SERVICE.changed.wait(&mut state);
        } else {
            SERVICE
                .changed
                .wait_for(&mut state, Duration::from_millis(10));
        }
    }
}

/// Unblocks an accept thread parked in `accept()` by connecting to it.
fn wake_accept(addr: &ListenAddr) {
    match addr {
        ListenAddr::IP(addr) => {
            let mut target = *addr;
            if target.ip().is_unspecified() {
                target.set_ip(match target.ip() {
                    IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
                    IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
                });
            }
            if let Ok(stream) = TcpStream::connect_timeout(&target, WAKE_CONNECT) {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        #[cfg(unix)]
        ListenAddr::Unix(addr) => {
            if let Some(path) = addr.as_pathname() {
                if let Ok(stream) = std::os::unix::net::UnixStream::connect(path) {
                    let _ = stream.shutdown(Shutdown::Both);
                }
            }
        }
    }
}

fn remove_unix_path(addr: &ListenAddr) {
    #[cfg(unix)]
    if let ListenAddr::Unix(addr) = addr {
        if let Some(path) = addr.as_pathname() {
            let _ = std::fs::remove_file(path);
        }
    }
    #[cfg(not(unix))]
    let _ = addr;
}

#[cfg(test)]
mod lifecycle_controls {
    //! Localhost sockets with finite read timeouts; every wait is bounded.
    use super::*;
    use crate::{Response, Server};
    use std::io::{Read, Write};
    use std::sync::mpsc;

    const CLIENT_WAIT: Duration = Duration::from_secs(3);

    fn server() -> (Server, SocketAddr) {
        let server = Server::http("127.0.0.1:0").unwrap();
        let addr = server.server_addr().to_ip().unwrap();
        (server, addr)
    }

    fn client(addr: SocketAddr, request: &str) -> TcpStream {
        let mut stream = TcpStream::connect_timeout(&addr, CLIENT_WAIT).unwrap();
        stream.set_read_timeout(Some(CLIENT_WAIT)).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        stream
    }

    /// Everything the server sent until it closed the connection. A read
    /// timeout means the connection was left open: that fails the caller.
    fn until_closed(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => bytes.extend_from_slice(&buf[..n]),
                Err(error) if error.kind() == io::ErrorKind::ConnectionReset => break,
                Err(error) => panic!("connection left open after shutdown: {}", error),
            }
        }
        String::from_utf8(bytes).unwrap()
    }

    fn recv(server: &Server) -> crate::Request {
        server
            .recv_timeout(CLIENT_WAIT)
            .unwrap()
            .expect("request within the client wait")
    }

    #[test]
    fn idle_keep_alive_connection_retires_inside_the_budget() {
        let (server, addr) = server();
        let mut stream = client(addr, "GET /one HTTP/1.1\r\nHost: l\r\n\r\n");
        recv(&server).respond(Response::from_string("one")).unwrap();
        // The worker is now parked reading the next request of a live
        // keep-alive connection; only the read-side close can release it.
        let report = server.shutdown(Instant::now() + Duration::from_secs(10));
        assert!(report.completed, "{:?}", report);
        assert!(!report.timed_out && report.cut.is_empty(), "{:?}", report);
        assert_eq!(report.threads_retiring, 0);
        assert!(
            report.threads_joined >= 5,
            "four pooled + accept: {:?}",
            report
        );
        assert_eq!(server.num_connections(), 0);
        let wire = until_closed(&mut stream);
        assert!(
            wire.starts_with("HTTP/1.1 200 OK") && wire.ends_with("one"),
            "{}",
            wire
        );
    }

    #[test]
    fn a_request_held_past_the_deadline_is_cut_and_named() {
        let (server, addr) = server();
        let mut stream = client(addr, "GET /held HTTP/1.1\r\nHost: l\r\n\r\n");
        let peer = stream.local_addr().unwrap();
        let held = recv(&server);
        let report = server.shutdown(Instant::now() + Duration::from_millis(200));
        assert!(report.timed_out && !report.completed, "{:?}", report);
        assert_eq!(
            report.cut,
            vec![CutConnection {
                id: 1,
                peer: Some(peer)
            }]
        );
        // Nothing reaches the peer after the cut, not even the held response.
        let _ = held.respond(Response::from_string("late"));
        assert_eq!(until_closed(&mut stream), "");
        let report = report.wait();
        assert_eq!(report.threads_retiring, 0);
        assert_eq!(server.num_connections(), 0);
    }

    #[test]
    fn a_queued_request_is_answered_by_shutdown() {
        let (server, addr) = server();
        let mut stream = client(addr, "GET /never HTTP/1.1\r\nHost: l\r\n\r\n");
        let until = Instant::now() + CLIENT_WAIT;
        while server.messages.len() == 0 {
            assert!(Instant::now() < until, "request never reached the queue");
            std::thread::sleep(Duration::from_millis(1));
        }
        let report = server.shutdown(Instant::now() + Duration::from_secs(10));
        assert!(report.completed, "{:?}", report);
        assert_eq!(report.queued_requests_dropped, 1);
        assert!(until_closed(&mut stream).starts_with("HTTP/1.1 500"));
        assert!(server.recv().is_err(), "recv after shutdown");
        let again = server.shutdown(Instant::now());
        assert!(again.already_shut_down && !again.completed);
    }

    #[test]
    fn drop_returns_before_the_held_response_and_lets_it_finish() {
        let (server, addr) = server();
        let mut stream = client(addr, "GET /late HTTP/1.1\r\nHost: l\r\n\r\n");
        let held = recv(&server);
        let (dropped, responder_go) = mpsc::channel();
        let responder = std::thread::spawn(move || {
            // Responds only once drop has returned: a drop that waited for
            // this response would reach its budget and cut the connection.
            responder_go.recv().unwrap();
            held.respond(Response::from_string("after drop")).unwrap();
        });
        drop(server);
        dropped.send(()).unwrap();
        responder.join().unwrap();
        let wire = until_closed(&mut stream);
        assert!(wire.ends_with("after drop"), "{}", wire);
        let until = Instant::now() + Duration::from_secs(10);
        let report = loop {
            let joined = join_retirement(Instant::now() + Duration::from_secs(1));
            if let Some(report) = joined
                .dropped_server_reports
                .into_iter()
                .find(|report| report.threads_joined > 0)
            {
                break report;
            }
            assert!(Instant::now() < until, "dropped server never retired");
        };
        assert!(report.completed && report.cut.is_empty(), "{:?}", report);
    }

    #[cfg(unix)]
    #[test]
    fn unix_listener_path_is_gone_when_drop_returns() {
        let path = std::env::temp_dir().join(format!("tiny-http-drop-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let server = Server::http_unix(&path).unwrap();
        assert!(path.exists());
        drop(server);
        assert!(
            !path.exists(),
            "a process exiting now must not leave the socket file"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_listener_path_is_removed() {
        let dir = std::env::temp_dir().join(format!("tiny-http-retire-{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let server = Server::http_unix(&dir).unwrap();
        assert!(dir.exists());
        let report = server.shutdown(Instant::now() + Duration::from_secs(10));
        assert!(report.completed, "{:?}", report);
        assert!(!dir.exists());
    }
}
