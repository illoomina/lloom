use std::io::Read;
use std::io::Result as IoResult;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};

/// A `Reader` that reads exactly the number of bytes from a sub-reader.
///
/// If the limit is reached, it returns EOF. If the limit is not reached
/// when the destructor is called, the remaining bytes will be read and
/// thrown away.
pub struct EqualReader<R>
where
    R: Read,
{
    reader: R,
    size: usize,
    last_read_signal: Sender<IoResult<()>>,
    control: Option<crate::ConnectionCancellation>,
}

impl<R> EqualReader<R>
where
    R: Read,
{
    pub fn new(reader: R, size: usize) -> (EqualReader<R>, Receiver<IoResult<()>>) {
        Self::new_controlled(reader, size, None)
    }
    pub(crate) fn new_controlled(
        reader: R,
        size: usize,
        control: Option<crate::ConnectionCancellation>,
    ) -> (EqualReader<R>, Receiver<IoResult<()>>) {
        let (tx, rx) = channel();

        let r = EqualReader {
            reader,
            size,
            last_read_signal: tx,
            control,
        };

        (r, rx)
    }
}

impl<R> Read for EqualReader<R>
where
    R: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(control) = &self.control {
            control.check()?;
        }
        if self.size == 0 {
            return Ok(0);
        }

        let buf = if buf.len() < self.size {
            buf
        } else {
            &mut buf[..self.size]
        };

        match self.reader.read(buf) {
            Ok(0) if self.control.is_some() => {
                if let Some(control) = &self.control {
                    let _ = control.cancel();
                }
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "incomplete request body",
                ))
            }
            Ok(len) => {
                self.size -= len;
                if let Some(control) = &self.control {
                    control.check()?;
                }
                Ok(len)
            }
            err @ Err(_) => err,
        }
    }
}

impl<R> Drop for EqualReader<R>
where
    R: Read,
{
    fn drop(&mut self) {
        let control = self.control.clone();
        let _budget = control.as_ref().map(|c| c.begin_disposal());
        let mut buf = [0u8; 4096];
        if let Some(control) = &control {
            if self.size > control.policy().max_discard_bytes {
                let _ = control.cancel();
                let _ = self.last_read_signal.send(Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unread body exceeds disposal limit",
                )));
                return;
            }
        }
        while self.size > 0 {
            let count = self.size.min(buf.len());
            match self.read(&mut buf[..count]) {
                Err(e) => {
                    if let Some(control) = &control {
                        let _ = control.cancel();
                    }
                    self.last_read_signal.send(Err(e)).ok();
                    return;
                }
                Ok(0) => {
                    if let Some(control) = &control {
                        let _ = control.cancel();
                        self.last_read_signal
                            .send(Err(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "incomplete request body",
                            )))
                            .ok();
                    } else {
                        self.last_read_signal.send(Ok(())).ok();
                    }
                    return;
                }
                Ok(_) => (),
            }
        }
        if control.is_some() {
            self.last_read_signal.send(Ok(())).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EqualReader;
    use std::io::Read;

    #[test]
    fn test_limit() {
        use std::io::Cursor;

        let mut org_reader = Cursor::new("hello world".to_string().into_bytes());

        {
            let (mut equal_reader, _) = EqualReader::new(org_reader.by_ref(), 5);

            let mut string = String::new();
            equal_reader.read_to_string(&mut string).unwrap();
            assert_eq!(string, "hello");
        }

        let mut string = String::new();
        org_reader.read_to_string(&mut string).unwrap();
        assert_eq!(string, " world");
    }

    #[test]
    fn test_not_enough() {
        use std::io::Cursor;

        let mut org_reader = Cursor::new("hello world".to_string().into_bytes());

        {
            let (mut equal_reader, _) = EqualReader::new(org_reader.by_ref(), 5);

            let mut vec = [0];
            equal_reader.read_exact(&mut vec).unwrap();
            assert_eq!(vec[0], b'h');
        }

        let mut string = String::new();
        org_reader.read_to_string(&mut string).unwrap();
        assert_eq!(string, " world");
    }
}

#[cfg(test)]
mod bounded_disposal_tests {
    use super::EqualReader;
    use std::cell::Cell;
    use std::io::{self, Cursor, Read};
    use std::rc::Rc;

    struct EofProbe {
        max_buffer: Rc<Cell<usize>>,
        calls: Rc<Cell<usize>>,
    }
    impl Read for EofProbe {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.max_buffer.set(self.max_buffer.get().max(buffer.len()));
            self.calls.set(self.calls.get() + 1);
            Ok(0)
        }
    }
    #[test]
    fn actual_drop_workspace_is_chunk_bounded_for_larger_logical_body() {
        let max_buffer = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        let source = EofProbe {
            max_buffer: max_buffer.clone(),
            calls: calls.clone(),
        };
        let (reader, signal) = EqualReader::new(source, 4097);
        drop(reader);
        assert_eq!(
            calls.get(),
            1,
            "actual EOF disposal must perform one bounded source read"
        );
        assert_eq!(
            max_buffer.get(),
            4096,
            "actual Drop must use fixed chunk, not logical remainder"
        );
        assert!(
            signal.recv().unwrap().is_ok(),
            "legacy EOF signal remains compatible"
        );
    }
    #[test]
    fn actual_small_partial_drop_keeps_next_pipeline_bytes_exact() {
        let mut stream = Cursor::new(b"abcNEXT".to_vec());
        let (mut reader, signal) = EqualReader::new(&mut stream, 3);
        let mut first = [0];
        reader.read_exact(&mut first).unwrap();
        assert_eq!(first, [b'a']);
        drop(reader);
        assert_eq!(
            stream.position(),
            3,
            "drain consumes only the declared body's remainder"
        );
        assert!(
            signal.recv().is_err(),
            "legacy complete drain retains original disconnected-signal behavior"
        );
        let mut following = [0; 4];
        stream.read_exact(&mut following).unwrap();
        assert_eq!(
            &following, b"NEXT",
            "valid following pipeline remains byte-identical"
        );
    }
}
