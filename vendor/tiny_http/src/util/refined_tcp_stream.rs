use std::io::Result as IoResult;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr};

use crate::connection::Connection;
#[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
use crate::ssl::SslStream;

pub(crate) enum Stream {
    Http(Connection),
    #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
    Https(SslStream),
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl From<Connection> for Stream {
    fn from(tcp_stream: Connection) -> Self {
        Stream::Http(tcp_stream)
    }
}

impl Stream {
    fn try_clone(&self) -> IoResult<Self> {
        match self {
            Stream::Http(socket) => socket.try_clone().map(Stream::Http),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(stream) => Ok(Stream::Https(stream.clone())),
        }
    }

    fn secure(&self) -> bool {
        match self {
            Stream::Http(_) => false,
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(_) => true,
        }
    }

    fn peer_addr(&mut self) -> IoResult<Option<SocketAddr>> {
        match self {
            Stream::Http(tcp_stream) => tcp_stream.peer_addr(),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(ssl_stream) => ssl_stream.peer_addr(),
        }
    }

    fn shutdown(&mut self, how: Shutdown) -> IoResult<()> {
        match self {
            Stream::Http(tcp_stream) => tcp_stream.shutdown(how),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(ssl_stream) => ssl_stream.shutdown(how),
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        match self {
            Stream::Http(tcp_stream) => tcp_stream.read(buf),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(ssl_stream) => ssl_stream.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        match self {
            Stream::Http(tcp_stream) => tcp_stream.write(buf),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(ssl_stream) => ssl_stream.write(buf),
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        match self {
            Stream::Http(tcp_stream) => tcp_stream.flush(),
            #[cfg(any(feature = "ssl-openssl", feature = "ssl-rustls"))]
            Stream::Https(ssl_stream) => ssl_stream.flush(),
        }
    }
}

pub struct RefinedTcpStream {
    stream: Stream,
    close_read: bool,
    close_write: bool,
    control: Option<crate::ConnectionCancellation>,
    // Keeps the server's registry entry for this connection alive for as long
    // as any half, request or escaped stream still uses it.
    live: Option<std::sync::Arc<crate::lifecycle::LiveConnection>>,
}

impl RefinedTcpStream {
    pub(crate) fn new<S>(stream: S) -> (RefinedTcpStream, RefinedTcpStream)
    where
        S: Into<Stream>,
    {
        Self::try_split(stream.into(), None).unwrap()
    }

    fn try_split(
        stream: Stream,
        control: Option<crate::ConnectionCancellation>,
    ) -> IoResult<(Self, Self)> {
        let read = stream.try_clone()?;
        Ok((
            Self {
                stream: read,
                close_read: true,
                close_write: false,
                control: control.clone(),
                live: None,
            },
            Self {
                stream,
                close_read: false,
                close_write: true,
                control,
                live: None,
            },
        ))
    }

    pub(crate) fn try_new_controlled(
        socket: Connection,
        control: crate::ConnectionCancellation,
    ) -> IoResult<(Self, Self)> {
        Self::try_split(socket.into(), Some(control))
    }

    pub(crate) fn new_controlled(
        socket: Connection,
        control: crate::ConnectionCancellation,
    ) -> (Self, Self) {
        Self::try_new_controlled(socket, control).unwrap()
    }

    pub(crate) fn attach_live(&mut self, live: std::sync::Arc<crate::lifecycle::LiveConnection>) {
        self.live = Some(live);
    }

    /// Returns true if this struct wraps around a secure connection.
    #[inline]
    pub(crate) fn secure(&self) -> bool {
        self.stream.secure()
    }

    pub(crate) fn peer_addr(&mut self) -> IoResult<Option<SocketAddr>> {
        self.stream.peer_addr()
    }
}

impl Drop for RefinedTcpStream {
    fn drop(&mut self) {
        if self.close_read {
            self.stream.shutdown(Shutdown::Read).ok();
        }

        if self.close_write {
            self.stream.shutdown(Shutdown::Write).ok();
        }
    }
}

impl Read for RefinedTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if let Some(control) = &self.control {
            control.prepare_io(true)?;
            return control.after_io(self.stream.read(buf));
        }
        self.stream.read(buf)
    }
}

impl Write for RefinedTcpStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if let Some(control) = &self.control {
            control.prepare_io(false)?;
            return control.after_io(self.stream.write(buf));
        }
        self.stream.write(buf)
    }

    fn flush(&mut self) -> IoResult<()> {
        if let Some(control) = &self.control {
            control.prepare_io(false)?;
            return control.after_io(self.stream.flush());
        }
        self.stream.flush()
    }
}
