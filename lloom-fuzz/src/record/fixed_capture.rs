//! The signal route's archive writer: the existing v1/v3 directory format,
//! read by the ordinary replay reader, written with descriptor-relative
//! syscalls and fixed stack buffers only. The caller supplies an already-open
//! parent directory, an exclusive coherent snapshot, the immutable identity
//! and a unique dump sequence. No allocation, std filesystem API or lock.

use sha2::{Digest, Sha256};

const FORMAT_V1: &[u8] = b"lloom-input-record-v1\n";
const FORMAT_V3: &[u8] = b"lloom-input-record-v3\n";

struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}
impl<const N: usize> Text<N> {
    fn new() -> Self {
        Self {
            bytes: [0; N],
            len: 0,
        }
    }
    fn append(&mut self, bytes: &[u8]) -> bool {
        let Some(end) = self.len.checked_add(bytes.len()) else {
            return false;
        };
        if end > N {
            return false;
        }
        self.bytes[self.len..end].copy_from_slice(bytes);
        self.len = end;
        true
    }
    fn decimal(&mut self, mut value: u64) -> bool {
        let mut digits = [0u8; 20];
        let mut start = digits.len();
        loop {
            start -= 1;
            digits[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        self.append(&digits[start..])
    }
    fn hex(&mut self, value: usize) -> bool {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut digits = [0u8; 2 + 2 * std::mem::size_of::<usize>()];
        digits[..2].copy_from_slice(b"0x");
        for (i, digit) in digits[2..].iter_mut().enumerate() {
            let shift = 4 * (2 * std::mem::size_of::<usize>() - 1 - i);
            *digit = HEX[(value >> shift) & 15];
        }
        self.append(&digits)
    }
    fn data(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn c_path(&mut self) -> Option<*const libc::c_char> {
        self.append(&[0]).then_some(self.bytes.as_ptr().cast())
    }
}

// A successful call advances by at least one byte. EINTR/zero/error leaves an
// explicitly incomplete archive; no unbounded retry on repeated interruption.
fn write_bytes(fd: libc::c_int, mut bytes: &[u8]) -> bool {
    while !bytes.is_empty() {
        // SAFETY: borrowed initialized bytes and caller-owned open descriptor.
        let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
        if written <= 0 {
            return false;
        }
        let written = written as usize;
        if written > bytes.len() {
            return false;
        }
        bytes = &bytes[written..];
    }
    true
}

struct Descriptor(libc::c_int);
impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: this descriptor was freshly opened and never transferred.
        // Do not retry close on EINTR: descriptor reuse must not close a successor.
        unsafe {
            libc::close(self.0);
        }
    }
}

fn new_file(dir: libc::c_int, name: *const libc::c_char) -> Option<Descriptor> {
    // SAFETY: NUL-terminated fixed name; parent descriptor remains caller-owned.
    // O_EXCL prevents overwriting any prior artifact, including symlinks.
    let fd = unsafe {
        libc::openat(
            dir,
            name,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600 as libc::c_uint,
        )
    };
    (fd >= 0).then_some(Descriptor(fd))
}

fn fixed_file(dir: libc::c_int, name: &'static std::ffi::CStr, contents: &[u8]) -> bool {
    let Some(fd) = new_file(dir, name.as_ptr()) else {
        return false;
    };
    write_bytes(fd.0, contents) && unsafe { libc::fsync(fd.0) } == 0
}

fn record_name(pid: u64, sequence: u64) -> Option<Text<64>> {
    let mut name = Text::<64>::new();
    (name.append(b"record-") && name.decimal(pid) && name.append(b"-") && name.decimal(sequence))
        .then_some(name)
}

/// Crash context beside a completed record: `bt.txt` with the faulting
/// address, the handler image's base and path, and one raw `pc=` line per
/// frame, innermost first. Best effort; the record does not depend on it.
pub(super) fn context(
    parent: libc::c_int,
    pid: u64,
    sequence: u64,
    si_addr: usize,
    image_base: usize,
    image_path: &[u8],
    frames: &[*mut libc::c_void],
) -> bool {
    let Some(mut path) = record_name(pid, sequence) else {
        return false;
    };
    if !path.append(b"/bt.txt") {
        return false;
    }
    let Some(path) = path.c_path() else {
        return false;
    };
    let Some(fd) = new_file(parent, path) else {
        return false;
    };
    let mut head = Text::<96>::new();
    if !(head.append(b"si_addr=")
        && head.hex(si_addr)
        && head.append(b"\nimage_base=")
        && head.hex(image_base)
        && head.append(b"\nimage_path="))
    {
        return false;
    }
    if !(write_bytes(fd.0, head.data())
        && write_bytes(fd.0, image_path)
        && write_bytes(fd.0, b"\n"))
    {
        return false;
    }
    for &frame in frames {
        let mut line = Text::<32>::new();
        if !(line.append(b"pc=")
            && line.hex(frame as usize)
            && line.append(b"\n")
            && write_bytes(fd.0, line.data()))
        {
            return false;
        }
    }
    // SAFETY: freshly opened descriptor owned by `fd`.
    unsafe { libc::fsync(fd.0) == 0 }
}

/// On failure all already-written artifacts remain inspectable. The completion
/// manifest is born LAST, only after every numbered input and reason is written
/// and synchronized. I/O may block: bounded memory is not a latency deadline.
pub(super) fn capture<'a, I>(
    parent: libc::c_int,
    pid: u64,
    sequence: u64,
    inputs: I,
    identity: Option<&[u8]>,
    reason: &[u8],
) -> bool
where
    I: Iterator<Item = &'a [u8]> + Clone,
{
    let count = inputs.clone().count();
    if count == 0 || identity.is_some_and(|id| id.is_empty() || id.len() > 65536) {
        return false;
    }
    let Some(mut name) = record_name(pid, sequence) else {
        return false;
    };
    let Some(name) = name.c_path() else {
        return false;
    };
    // SAFETY: caller-owned parent and fixed NUL-terminated fresh directory name.
    if unsafe { libc::mkdirat(parent, name, 0o700 as libc::mode_t) } != 0 {
        return false;
    }
    let dir = unsafe {
        libc::openat(
            parent,
            name,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if dir < 0 {
        return false;
    }
    let dir = Descriptor(dir);
    let format = if identity.is_some() {
        FORMAT_V3
    } else {
        FORMAT_V1
    };
    if !fixed_file(dir.0, c"record.format", format) {
        return false;
    }
    if let Some(id) = identity {
        if !fixed_file(dir.0, c"record.identity", id) {
            return false;
        }
    }
    // Manifest staging is one file, not an allocation-sized metadata vector.
    // A .pending filename cannot be mistaken for replay completion evidence.
    let Some(manifest) = new_file(dir.0, c"record.pending".as_ptr()) else {
        return false;
    };
    let mut count_line = Text::<24>::new();
    if !(count_line.decimal(count as u64)
        && count_line.append(b"\n")
        && write_bytes(manifest.0, format)
        && write_bytes(manifest.0, count_line.data()))
    {
        return false;
    }
    for (index, bytes) in inputs.enumerate() {
        if bytes.len() > 65536 {
            return false;
        }
        let mut number = Text::<20>::new();
        if !number.decimal(index as u64) {
            return false;
        }
        let mut filename = Text::<25>::new();
        for _ in number.len..20 {
            if !filename.append(b"0") {
                return false;
            }
        }
        if !(filename.append(number.data()) && filename.append(b".bin")) {
            return false;
        }
        let Some(filename) = filename.c_path() else {
            return false;
        };
        let Some(fd) = new_file(dir.0, filename) else {
            return false;
        };
        if !write_bytes(fd.0, bytes) || unsafe { libc::fsync(fd.0) } != 0 {
            return false;
        }
        let mut line = Text::<88>::new();
        if !line.decimal(bytes.len() as u64) {
            return false;
        }
        if identity.is_some() {
            if !line.append(b" ") {
                return false;
            }
            // sha2 builds with {default, std} only, which on aarch64 is its
            // portable compressor: no feature detection, no allocation.
            let digest = Sha256::digest(bytes);
            const HEX: &[u8; 16] = b"0123456789abcdef";
            for byte in digest {
                if !line.append(&[HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]]) {
                    return false;
                }
            }
        }
        if !line.append(b"\n") || !write_bytes(manifest.0, line.data()) {
            return false;
        }
    }
    if !fixed_file(dir.0, c"reason.txt", reason)
        || !write_bytes(manifest.0, b"END\n")
        || unsafe { libc::fsync(manifest.0) } != 0
    {
        return false;
    }
    // The directory is new and never exported while being built. linkat's
    // no-overwrite publication preserves inspectable pending bytes on failure.
    if unsafe {
        libc::linkat(
            dir.0,
            c"record.pending".as_ptr(),
            dir.0,
            c"record.complete".as_ptr(),
            0,
        )
    } != 0
    {
        return false;
    }
    (unsafe { libc::fsync(dir.0) }) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn parent() -> (std::path::PathBuf, std::fs::File) {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "lloom-fixed-capture-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fd = std::fs::File::open(&path).unwrap();
        (path, fd)
    }

    #[test]
    fn fixed_writer_replays_identified_and_legacy_without_overwriting() {
        let (parent, fd) = parent();
        let values = [b"B".as_slice(), b"", b"C"];
        for (sequence, identity) in [(0, Some(b"identity".as_slice())), (1, None)] {
            assert!(capture(
                fd.as_raw_fd(),
                7,
                sequence,
                values.into_iter(),
                identity,
                b"control"
            ));
            let dir = parent.join(format!("record-7-{sequence}"));
            let mut observed = Vec::new();
            assert_eq!(
                super::super::replay_with_identity(&dir, identity, false, |bytes| {
                    observed.push(bytes.to_vec());
                    Ok(())
                })
                .unwrap(),
                3,
                "fixed archive must replay through original reader"
            );
            assert_eq!(observed, values.map(|v| v.to_vec()));
            let original = std::fs::read(dir.join("record.complete")).unwrap();
            assert!(!capture(
                fd.as_raw_fd(),
                7,
                sequence,
                [b"new".as_slice()].into_iter(),
                identity,
                b"must not replace"
            ));
            assert_eq!(
                std::fs::read(dir.join("record.complete")).unwrap(),
                original
            );
            if identity.is_some() {
                std::fs::write(dir.join("00000000000000000000.bin"), b"X").unwrap();
                let mut calls = 0;
                assert!(
                    super::super::replay_with_identity(&dir, identity, false, |_| {
                        calls += 1;
                        Ok(())
                    })
                    .is_err()
                );
                assert_eq!(calls, 0, "changed payload cannot reach target");
            }
        }
    }

    #[test]
    fn fixed_writer_uses_real_wrapped_ring_snapshot_and_keeps_its_permit() {
        use super::super::inputs::{InputRing, RecordError};

        let (parent, fd) = parent();
        let ring = InputRing::new(2, 4);
        for input in [b"old".as_slice(), b"B", b""] {
            ring.push(input).unwrap();
        }
        let snapshot = ring.snapshot().unwrap();
        let inputs = snapshot.inputs();
        let independently_traversed = inputs.clone();
        assert!(capture(
            fd.as_raw_fd(),
            9,
            0,
            inputs,
            Some(b"identity"),
            b"ring"
        ));
        assert_eq!(
            independently_traversed.collect::<Vec<_>>(),
            [b"B".as_slice(), b""]
        );
        assert_eq!(
            ring.push(b"new"),
            Err(RecordError::Busy),
            "writer traversal must not release the original snapshot permit"
        );
        drop(snapshot);
        ring.push(b"new").unwrap();
        let mut replayed = Vec::new();
        assert_eq!(
            super::super::replay_with_identity(
                &parent.join("record-9-0"),
                Some(b"identity"),
                false,
                |bytes| {
                    replayed.push(bytes.to_vec());
                    Ok(())
                }
            )
            .unwrap(),
            2
        );
        assert_eq!(
            replayed,
            [b"B".to_vec(), Vec::new()],
            "real ring wrap/empty payload must survive original archive replay"
        );
    }

    #[test]
    fn fixed_writer_failure_never_publishes_completion() {
        let (parent, fd) = parent();
        assert!(!capture(
            -1,
            8,
            0,
            [b"B".as_slice()].into_iter(),
            None,
            b"bad fd"
        ));
        assert!(!parent.join("record-8-0").exists());
        let oversized = vec![0u8; 65537];
        assert!(!capture(
            fd.as_raw_fd(),
            8,
            1,
            [oversized.as_slice()].into_iter(),
            Some(b"identity"),
            b"too large"
        ));
        let dir = parent.join("record-8-1");
        assert!(
            dir.join("record.pending").exists(),
            "keep partial artifacts inspectable"
        );
        assert!(!dir.join("record.complete").exists());
        let mut calls = 0;
        assert!(
            super::super::replay_with_identity(&dir, Some(b"identity"), false, |_| {
                calls += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 0, "incomplete capture cannot reach target");
    }
}
