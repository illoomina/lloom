use std::io::Result as IoResult;
use std::io::{Read, Write};

use crate::ConnectionCancellation;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::sync::Mutex;

use std::mem;

pub struct SequentialReaderBuilder<R>
where
    R: Read + Send,
{
    inner: SequentialReaderBuilderInner<R>,
    control: Option<ConnectionCancellation>,
}

enum SequentialReaderBuilderInner<R>
where
    R: Read + Send,
{
    First(R),
    NotFirst(Receiver<R>),
}

pub struct SequentialReader<R>
where
    R: Read + Send,
{
    inner: SequentialReaderInner<R>,
    next: Sender<R>,
    control: Option<ConnectionCancellation>,
}

enum SequentialReaderInner<R>
where
    R: Read + Send,
{
    MyTurn(R),
    Waiting(Receiver<R>),
    Empty,
}

pub struct SequentialWriterBuilder<W>
where
    W: Write + Send,
{
    writer: Arc<Mutex<W>>,
    next_trigger: Option<Receiver<()>>,
    control: Option<ConnectionCancellation>,
}

pub struct SequentialWriter<W>
where
    W: Write + Send,
{
    trigger: Option<Receiver<()>>,
    writer: Arc<Mutex<W>>,
    on_finish: Sender<()>,
    control: Option<ConnectionCancellation>,
}

impl<R: Read + Send> SequentialReaderBuilder<R> {
    pub(crate) fn with_control(reader: R, control: Option<ConnectionCancellation>) -> Self {
        Self {
            inner: SequentialReaderBuilderInner::First(reader),
            control,
        }
    }
    pub fn new(reader: R) -> SequentialReaderBuilder<R> {
        SequentialReaderBuilder {
            inner: SequentialReaderBuilderInner::First(reader),
            control: None,
        }
    }
}

impl<W: Write + Send> SequentialWriterBuilder<W> {
    pub(crate) fn with_control(writer: W, control: Option<ConnectionCancellation>) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
            next_trigger: None,
            control,
        }
    }
    pub fn new(writer: W) -> SequentialWriterBuilder<W> {
        SequentialWriterBuilder {
            writer: Arc::new(Mutex::new(writer)),
            next_trigger: None,
            control: None,
        }
    }
}

impl<R: Read + Send> Iterator for SequentialReaderBuilder<R> {
    type Item = SequentialReader<R>;

    fn next(&mut self) -> Option<SequentialReader<R>> {
        let (tx, rx) = channel();

        let inner = mem::replace(&mut self.inner, SequentialReaderBuilderInner::NotFirst(rx));

        match inner {
            SequentialReaderBuilderInner::First(reader) => Some(SequentialReader {
                inner: SequentialReaderInner::MyTurn(reader),
                next: tx,
                control: self.control.clone(),
            }),

            SequentialReaderBuilderInner::NotFirst(previous) => Some(SequentialReader {
                inner: SequentialReaderInner::Waiting(previous),
                next: tx,
                control: self.control.clone(),
            }),
        }
    }
}

impl<W: Write + Send> Iterator for SequentialWriterBuilder<W> {
    type Item = SequentialWriter<W>;
    fn next(&mut self) -> Option<SequentialWriter<W>> {
        let (tx, rx) = channel();
        let mut next_next_trigger = Some(rx);
        ::std::mem::swap(&mut next_next_trigger, &mut self.next_trigger);

        Some(SequentialWriter {
            trigger: next_next_trigger,
            writer: self.writer.clone(),
            on_finish: tx,
            control: self.control.clone(),
        })
    }
}

impl<R: Read + Send> Read for SequentialReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if self.control.is_none() {
            // Preserve incumbent legacy ownership progression and disconnected
            // handoff panic. Only the explicit-policy route is fallible here.
            let mut reader = match self.inner {
                SequentialReaderInner::MyTurn(ref mut reader) => return reader.read(buf),
                SequentialReaderInner::Waiting(ref mut recv) => recv.recv().unwrap(),
                SequentialReaderInner::Empty => unreachable!(),
            };
            let result = reader.read(buf);
            self.inner = SequentialReaderInner::MyTurn(reader);
            return result;
        }
        if let Some(control) = &self.control {
            control.check()?;
        }
        if let SequentialReaderInner::Waiting(recv) = &self.inner {
            let reader = match &self.control {
                Some(control) => control.receive(recv)?,
                None => recv.recv().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::ConnectionAborted,
                        "ownership handoff closed",
                    )
                })?,
            };
            self.inner = SequentialReaderInner::MyTurn(reader);
        }
        match &mut self.inner {
            SequentialReaderInner::MyTurn(reader) => {
                let result = reader.read(buf);
                match &self.control {
                    Some(control) => control.after_io(result),
                    None => result,
                }
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "reader unavailable",
            )),
        }
    }
}
impl<W: Write + Send> SequentialWriter<W> {
    fn wait_turn(&mut self) -> IoResult<()> {
        if let Some(control) = &self.control {
            control.check()?;
        }
        if let Some(receiver) = &self.trigger {
            match &self.control {
                Some(control) => control.receive(receiver)?,
                None => receiver.recv().unwrap(),
            }
        }
        self.trigger = None;
        Ok(())
    }
}
impl<W: Write + Send> Write for SequentialWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        self.wait_turn()?;
        match &self.control {
            Some(control) => control.lock_writer(&self.writer)?.write(buf),
            None => self.writer.lock().unwrap().write(buf),
        }
    }
    fn flush(&mut self) -> IoResult<()> {
        self.wait_turn()?;
        match &self.control {
            Some(control) => control.lock_writer(&self.writer)?.flush(),
            None => self.writer.lock().unwrap().flush(),
        }
    }
}
impl<R: Read + Send> Drop for SequentialReader<R> {
    fn drop(&mut self) {
        // Cancellation discards handoff authority: undrained bytes cannot become
        // a next header, even if BufReader already contains them.
        if self
            .control
            .as_ref()
            .is_some_and(|control| control.check().is_err())
        {
            return;
        }
        let inner = mem::replace(&mut self.inner, SequentialReaderInner::Empty);
        match inner {
            SequentialReaderInner::MyTurn(reader) => {
                self.next.send(reader).ok();
            }
            SequentialReaderInner::Waiting(recv) => {
                let reader = match &self.control {
                    Some(control) => control.receive(&recv).ok(),
                    None => Some(recv.recv().unwrap()),
                };
                if let Some(reader) = reader {
                    self.next.send(reader).ok();
                }
            }
            SequentialReaderInner::Empty => (),
        }
    }
}

impl<W> Drop for SequentialWriter<W>
where
    W: Write + Send,
{
    fn drop(&mut self) {
        self.on_finish.send(()).ok();
    }
}
