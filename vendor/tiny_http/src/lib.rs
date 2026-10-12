//! # Simple usage
//!
//! ## Creating the server
//!
//! The easiest way to create a server is to call `Server::http()`.
//!
//! The `http()` function returns an `IoResult<Server>` which will return an error
//! in the case where the server creation fails (for example if the listening port is already
//! occupied).
//!
//! ```no_run
//! let server = tiny_http::Server::http("0.0.0.0:0").unwrap();
//! ```
//!
//! A newly-created `Server` will immediately start listening for incoming connections and HTTP
//! requests.
//!
//! ## Receiving requests
//!
//! Calling `server.recv()` will block until the next request is available.
//! This function returns an `IoResult<Request>`, so you need to handle the possible errors.
//!
//! ```no_run
//! # let server = tiny_http::Server::http("0.0.0.0:0").unwrap();
//!
//! loop {
//!     // blocks until the next request is received
//!     let request = match server.recv() {
//!         Ok(rq) => rq,
//!         Err(e) => { println!("error: {}", e); break }
//!     };
//!
//!     // do something with the request
//!     // ...
//! }
//! ```
//!
//! In a real-case scenario, you will probably want to spawn multiple worker tasks and call
//! `server.recv()` on all of them. Like this:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use std::thread;
//! # let server = tiny_http::Server::http("0.0.0.0:0").unwrap();
//! let server = Arc::new(server);
//! let mut guards = Vec::with_capacity(4);
//!
//! for _ in (0 .. 4) {
//!     let server = server.clone();
//!
//!     let guard = thread::spawn(move || {
//!         loop {
//!             let rq = server.recv().unwrap();
//!
//!             // ...
//!         }
//!     });
//!
//!     guards.push(guard);
//! }
//! ```
//!
//! If you don't want to block, you can call `server.try_recv()` instead.
//!
//! ## Handling requests
//!
//! The `Request` object returned by `server.recv()` contains informations about the client's request.
//! The most useful methods are probably `request.method()` and `request.url()` which return
//! the requested method (`GET`, `POST`, etc.) and url.
//!
//! To handle a request, you need to create a `Response` object. See the docs of this object for
//! more infos. Here is an example of creating a `Response` from a file:
//!
//! ```no_run
//! # use std::fs::File;
//! # use std::path::Path;
//! let response = tiny_http::Response::from_file(File::open(&Path::new("image.png")).unwrap());
//! ```
//!
//! All that remains to do is call `request.respond()`:
//!
//! ```no_run
//! # use std::fs::File;
//! # use std::path::Path;
//! # let server = tiny_http::Server::http("0.0.0.0:0").unwrap();
//! # let request = server.recv().unwrap();
//! # let response = tiny_http::Response::from_file(File::open(&Path::new("image.png")).unwrap());
//! let _ = request.respond(response);
//! ```
#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]
#![allow(clippy::match_like_matches_macro)]

#[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
use zeroize::Zeroizing;

use std::error::Error;
use std::io::Error as IoError;
use std::io::ErrorKind as IoErrorKind;
use std::io::Result as IoResult;
use std::net::ToSocketAddrs;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use client::ClientConnection;
use util::MessagesQueue;

pub use common::{HTTPVersion, Header, HeaderField, Method, StatusCode};
pub use connection::{ConfigListenAddr, ListenAddr, Listener};
pub use lifecycle::{
    join_retirement, CutConnection, RetirementReport, ShutdownReport, DROP_RETIREMENT_BUDGET,
};
pub use request::{ReadWrite, Request};
pub use response::{Response, ResponseBox};
pub use test::TestRequest;

mod lifecycle;
mod transport;
pub use transport::{ConnectionCancellation, TransportPolicy};
mod client;
mod common;
mod connection;
mod request;
mod response;
mod ssl;
mod test;
mod util;

/// The main class of this library.
///
/// [`Server::shutdown`] retires the server within a deadline: no new
/// connection is accepted, the reading part of every client connection is
/// closed, queued requests are answered by their drop, and requests already
/// returned by `recv()` may still respond until the deadline. Connections
/// still in use at the deadline are closed in both directions and named in
/// the report. Dropping the server does the same on the retirement service
/// with [`DROP_RETIREMENT_BUDGET`], without blocking the dropping thread; call
/// [`join_retirement`] before the process exits.
pub struct Server {
    // should be false as long as the server exists
    // when set to true, the accept thread stops accepting
    close: Arc<AtomicBool>,

    // queue for messages received by child threads
    messages: Arc<MessagesQueue<Message>>,

    // result of TcpListener::local_addr()
    listening_addr: ListenAddr,

    connections: Arc<lifecycle::Connections>,

    // taken by the first shutdown, or by Drop
    lifecycle: parking_lot::Mutex<Option<lifecycle::Lifecycle>>,
}

pub(crate) enum Message {
    Error(IoError),
    NewRequest(Request),
}

impl From<IoError> for Message {
    fn from(e: IoError) -> Message {
        Message::Error(e)
    }
}

impl From<Request> for Message {
    fn from(rq: Request) -> Message {
        Message::NewRequest(rq)
    }
}

// this trait is to make sure that Server implements Share and Send
#[doc(hidden)]
trait MustBeShareDummy: Sync + Send {}
#[doc(hidden)]
impl MustBeShareDummy for Server {}

pub struct IncomingRequests<'a> {
    server: &'a Server,
}

/// Represents the parameters required to create a server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// The addresses to try to listen to.
    pub addr: ConfigListenAddr,

    /// If `Some`, then the server will use SSL to encode the communications.
    pub ssl: Option<SslConfig>,
}

/// Configuration of the server for SSL.
#[derive(Debug, Clone)]
pub struct SslConfig {
    /// Contains the public certificate to send to clients.
    pub certificate: Vec<u8>,
    /// Contains the ultra-secret private key used to decode communications.
    pub private_key: Vec<u8>,
}

impl Server {
    /// Shortcut for a simple server on a specific address.
    #[inline]
    pub fn http<A>(addr: A) -> Result<Server, Box<dyn Error + Send + Sync + 'static>>
    where
        A: ToSocketAddrs,
    {
        Server::new(ServerConfig {
            addr: ConfigListenAddr::from_socket_addrs(addr)?,
            ssl: None,
        })
    }

    /// Opt-in bounded disposal/plain-HTTP I/O. Limits are explicit, including
    /// pipelined ownership waits; no application/model execution or thread-join
    /// guarantee follows. Legacy HTTP/HTTPS constructors remain unchanged.
    pub fn http_with_policy<A: ToSocketAddrs>(
        addr: A,
        policy: TransportPolicy,
    ) -> Result<Server, Box<dyn Error + Send + Sync + 'static>> {
        let policy = policy.validate()?;
        let listener = ConfigListenAddr::from_socket_addrs(addr)?.bind()?;
        Self::from_listener_inner(listener, None, Some(policy))
    }

    /// Shortcut for an HTTPS server on a specific address.
    #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
    #[inline]
    pub fn https<A>(
        addr: A,
        config: SslConfig,
    ) -> Result<Server, Box<dyn Error + Send + Sync + 'static>>
    where
        A: ToSocketAddrs,
    {
        Server::new(ServerConfig {
            addr: ConfigListenAddr::from_socket_addrs(addr)?,
            ssl: Some(config),
        })
    }

    #[cfg(unix)]
    #[inline]
    /// Shortcut for a UNIX socket server at a specific path
    pub fn http_unix(
        path: &std::path::Path,
    ) -> Result<Server, Box<dyn Error + Send + Sync + 'static>> {
        Server::new(ServerConfig {
            addr: ConfigListenAddr::unix_from_path(path),
            ssl: None,
        })
    }

    /// Builds a new server that listens on the specified address.
    pub fn new(config: ServerConfig) -> Result<Server, Box<dyn Error + Send + Sync + 'static>> {
        let listener = config.addr.bind()?;
        Self::from_listener(listener, config.ssl)
    }

    /// Builds a new server using the specified TCP listener.
    ///
    /// This is useful if you've constructed TcpListener using some less usual method
    /// such as from systemd. For other cases, you probably want the `new()` function.
    pub fn from_listener<L: Into<Listener>>(
        listener: L,
        ssl_config: Option<SslConfig>,
    ) -> Result<Server, Box<dyn Error + Send + Sync + 'static>> {
        Self::from_listener_inner(listener.into(), ssl_config, None)
    }
    fn from_listener_inner(
        listener: Listener,
        ssl_config: Option<SslConfig>,
        policy: Option<TransportPolicy>,
    ) -> Result<Server, Box<dyn Error + Send + Sync + 'static>> {
        // building the "close" variable
        let close_trigger = Arc::new(AtomicBool::new(false));

        // building the TcpListener
        let (server, local_addr) = {
            let local_addr = listener.local_addr()?;
            log::debug!("Server listening on {}", local_addr);
            (listener, local_addr)
        };

        // building the SSL capabilities
        #[cfg(all(feature = "ssl-openssl", feature = "ssl-rustls"))]
        compile_error!(
            "Features 'ssl-openssl' and 'ssl-rustls' must not be enabled at the same time"
        );
        #[cfg(not(any(feature = "ssl-openssl", feature = "ssl-rustls")))]
        type SslContext = ();
        #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
        type SslContext = crate::ssl::SslContextImpl;
        let ssl: Option<SslContext> = {
            match ssl_config {
                #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
                Some(config) => Some(SslContext::from_pem(
                    config.certificate,
                    Zeroizing::new(config.private_key),
                )?),
                #[cfg(not(any(feature = "ssl-openssl", feature = "ssl-rustls")))]
                Some(_) => return Err(
                    "Building a server with SSL requires enabling the `ssl` feature in tiny-http"
                        .into(),
                ),
                None => None,
            }
        };

        // creating a task where server.accept() is continuously called
        // and ClientConnection objects are pushed in the messages queue
        let messages = MessagesQueue::with_capacity(8);
        let pool = Arc::new(util::TaskPool::new());
        let connections = Arc::new(lifecycle::Connections::default());

        let inside_close_trigger = close_trigger.clone();
        let inside_messages = messages.clone();
        let inside_pool = pool.clone();
        let inside_connections = connections.clone();
        let accept = thread::Builder::new()
            .name("tiny-http-accept".into())
            .spawn(move || {
                log::debug!("Running accept thread");
                while !inside_close_trigger.load(SeqCst) {
                    let new_client = match server.accept() {
                        Ok((sock, peer)) => {
                            if inside_close_trigger.load(SeqCst) {
                                break;
                            }
                            use util::RefinedTcpStream;
                            // A connection the server cannot reach at shutdown
                            // is not accepted.
                            let live = match inside_connections.register(&sock, peer) {
                                Ok(live) => live,
                                Err(error) => {
                                    log::error!("Error registering new client: {}", error);
                                    continue;
                                }
                            };
                            let control = match policy {
                                Some(policy) => match ConnectionCancellation::new(&sock, policy) {
                                    Ok(control) => Some(control),
                                    Err(error) => {
                                        inside_messages.push(error.into());
                                        continue;
                                    }
                                },
                                None => None,
                            };
                            let (mut read_closable, mut write_closable) = match ssl {
                                None => match &control {
                                    Some(control) => {
                                        RefinedTcpStream::new_controlled(sock, control.clone())
                                    }
                                    None => RefinedTcpStream::new(sock),
                                },
                                #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
                                Some(ref ssl) => {
                                    // trying to apply SSL over the connection
                                    // if an error occurs, we just close the socket and resume listening
                                    let sock = match ssl.accept(sock) {
                                        Ok(s) => s,
                                        Err(_) => continue,
                                    };

                                    RefinedTcpStream::new(sock)
                                }
                                #[cfg(not(any(feature = "ssl-openssl", feature = "ssl-rustls")))]
                                Some(ref _ssl) => unreachable!(),
                            };
                            read_closable.attach_live(live.clone());
                            write_closable.attach_live(live);

                            Ok(ClientConnection::new(
                                write_closable,
                                read_closable,
                                control,
                            ))
                        }
                        Err(e) => Err(e),
                    };

                    match new_client {
                        Ok(client) => {
                            let messages = inside_messages.clone();
                            let close = inside_close_trigger.clone();
                            let mut client = Some(client);
                            inside_pool.spawn(Box::new(move || {
                                if let Some(client) = client.take() {
                                    serve_connection(client, &messages, &close);
                                }
                            }));
                        }

                        Err(e) => {
                            log::error!("Error accepting new client: {}", e);
                            inside_messages.push(e.into());
                            break;
                        }
                    }
                }
                log::debug!("Terminating accept thread");
            })?;

        // result
        Ok(Server {
            messages: messages.clone(),
            close: close_trigger.clone(),
            listening_addr: local_addr.clone(),
            connections: connections.clone(),
            lifecycle: parking_lot::Mutex::new(Some(lifecycle::Lifecycle {
                close: close_trigger,
                messages,
                listening_addr: local_addr,
                accept,
                pool,
                connections,
            })),
        })
    }

    /// Returns an iterator for all the incoming requests.
    ///
    /// The iterator will return `None` if the server socket is shutdown.
    #[inline]
    pub fn incoming_requests(&self) -> IncomingRequests<'_> {
        IncomingRequests { server: self }
    }

    /// Returns the address the server is listening to.
    #[inline]
    pub fn server_addr(&self) -> ListenAddr {
        self.listening_addr.clone()
    }

    /// Returns the number of client connections still in use: by a worker,
    /// a request not yet answered, or a stream taken out of a request.
    pub fn num_connections(&self) -> usize {
        self.connections.live_count()
    }

    /// Retires the server by `deadline` and reports what that took.
    ///
    /// Stops accepting, closes the reading part of every connection, drops
    /// the requests never received (each answers its client), and waits for
    /// the accept and worker threads and for every request or stream still
    /// using a connection. At the deadline the remaining connections are
    /// closed in both directions and listed in [`ShutdownReport::cut`];
    /// threads still running are then joined by the retirement service.
    /// Returns no later than the deadline plus one socket shutdown per
    /// remaining connection. Later calls, and `recv()` after this one, find
    /// the server closed.
    pub fn shutdown(&self, deadline: Instant) -> ShutdownReport {
        lifecycle::shutdown(self.lifecycle.lock().take(), deadline)
    }

    /// [`Server::shutdown`], then waits until every thread it left running
    /// has been joined.
    pub fn shutdown_blocking(&self, deadline: Instant) -> ShutdownReport {
        self.shutdown(deadline).wait()
    }

    /// Blocks until an HTTP request has been submitted and returns it.
    pub fn recv(&self) -> IoResult<Request> {
        match self.messages.pop() {
            Some(Message::Error(err)) => Err(err),
            Some(Message::NewRequest(rq)) => Ok(rq),
            None => Err(IoError::new(IoErrorKind::Other, "thread unblocked")),
        }
    }

    /// Same as `recv()` but doesn't block longer than timeout
    pub fn recv_timeout(&self, timeout: Duration) -> IoResult<Option<Request>> {
        match self.messages.pop_timeout(timeout) {
            Some(Message::Error(err)) => Err(err),
            Some(Message::NewRequest(rq)) => Ok(Some(rq)),
            None => Ok(None),
        }
    }

    /// Same as `recv()` but doesn't block.
    pub fn try_recv(&self) -> IoResult<Option<Request>> {
        match self.messages.try_pop() {
            Some(Message::Error(err)) => Err(err),
            Some(Message::NewRequest(rq)) => Ok(Some(rq)),
            None => Ok(None),
        }
    }

    /// Unblock thread stuck in recv() or incoming_requests().
    /// If there are several such threads, only one is unblocked.
    /// This method allows graceful shutdown of server.
    pub fn unblock(&self) {
        self.messages.unblock();
    }
}

impl Iterator for IncomingRequests<'_> {
    type Item = Request;
    fn next(&mut self) -> Option<Request> {
        self.server.recv().ok()
    }
}

/// Feeds one connection's requests to the queue until the client stops, the
/// queue closes, or (HTTPS) the server closes while a response is pending.
fn serve_connection(
    client: ClientConnection,
    messages: &MessagesQueue<Message>,
    close: &AtomicBool,
) {
    // Synchronization is needed for HTTPS requests to avoid a deadlock
    if client.secure() {
        let (sender, receiver) = mpsc::channel();
        for rq in client {
            if messages
                .try_push(rq.with_notify_sender(sender.clone()).into())
                .is_err()
            {
                return;
            }
            loop {
                match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(()) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) if !close.load(SeqCst) => (),
                    Err(_) => return,
                }
            }
        }
    } else {
        for rq in client {
            if messages.try_push(rq.into()).is_err() {
                return;
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.close.store(true, Relaxed);
        if let Some(lifecycle) = self.lifecycle.get_mut().take() {
            lifecycle::retire_in_background(lifecycle);
        }
    }
}
