//! Bounded input history with nonblocking exclusive access.
//!
//! A process-global panic/signal hook may run on a different thread from the
//! campaign. Publication of a length alone cannot protect ordinary bytes from
//! overwrite. Every payload access here instead owns the same atomic permit.
//! Contention is reported, never spun on (a signal may interrupt the owner).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordError {
    NotInstalled,
    Busy,
    TooLarge { actual: usize, maximum: usize },
    SequenceExhausted,
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled => f.write_str("persistent record has not been installed"),
            Self::Busy => f.write_str("persistent record is already being read or written"),
            Self::TooLarge { actual, maximum } => write!(
                f,
                "persistent record input has {actual} bytes; maximum is {maximum}"
            ),
            Self::SequenceExhausted => f.write_str("persistent record sequence exhausted"),
        }
    }
}
impl std::error::Error for RecordError {}

struct Entries {
    slots: Vec<Box<[u8]>>,
    lengths: Vec<usize>,
    total: usize,
}

pub(super) struct InputRing {
    busy: AtomicBool,
    entries: UnsafeCell<Entries>,
    capacity: usize,
    maximum: usize,
}

// SAFETY: all access to entries, including reads, owns the unique permit.
// Snapshot borrows cannot outlive that permit; neither payload nor permit is
// exposed to callers. The immutable capacity/maximum need no synchronization.
unsafe impl Sync for InputRing {}

struct Permit<'a>(&'a AtomicBool);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl InputRing {
    pub(super) fn new(capacity: usize, maximum: usize) -> Self {
        assert!(capacity > 0);
        Self {
            busy: AtomicBool::new(false),
            entries: UnsafeCell::new(Entries {
                slots: (0..capacity)
                    .map(|_| vec![0; maximum].into_boxed_slice())
                    .collect(),
                lengths: vec![0; capacity],
                total: 0,
            }),
            capacity,
            maximum,
        }
    }

    fn acquire(&self) -> Result<Permit<'_>, RecordError> {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map(|_| Permit(&self.busy))
            .map_err(|_| RecordError::Busy)
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn push(&self, bytes: &[u8]) -> Result<(), RecordError> {
        if bytes.len() > self.maximum {
            return Err(RecordError::TooLarge {
                actual: bytes.len(),
                maximum: self.maximum,
            });
        }
        let _permit = self.acquire()?;
        // SAFETY: permit excludes all other readers and writers until scope exit.
        let entries = unsafe { &mut *self.entries.get() };
        let next = entries
            .total
            .checked_add(1)
            .ok_or(RecordError::SequenceExhausted)?;
        let index = entries.total % self.capacity;
        entries.slots[index][..bytes.len()].copy_from_slice(bytes);
        entries.lengths[index] = bytes.len();
        entries.total = next;
        Ok(())
    }

    pub(super) fn snapshot(&self) -> Result<Snapshot<'_>, RecordError> {
        let permit = self.acquire()?;
        // SAFETY: snapshot owns permit and exposes only borrows tied to itself.
        let entries = unsafe { &*self.entries.get() };
        Ok(Snapshot {
            entries,
            _permit: permit,
        })
    }
}

pub(super) struct Snapshot<'a> {
    entries: &'a Entries,
    _permit: Permit<'a>,
}
impl Snapshot<'_> {
    pub(super) fn inputs(&self) -> impl Iterator<Item = &[u8]> {
        let entries = self.entries;
        let capacity = entries.slots.len();
        let oldest = entries.total.saturating_sub(capacity);
        (oldest..entries.total).map(move |sequence| {
            let index = sequence % capacity;
            &entries.slots[index][..entries.lengths[index]]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_keeps_exact_bytes_in_execution_order() {
        let ring = InputRing::new(2, 4);
        for bytes in [b"old".as_slice(), b"", b"last"] {
            ring.push(bytes).unwrap();
        }
        let snapshot = ring.snapshot().unwrap();
        assert_eq!(
            snapshot.inputs().collect::<Vec<_>>(),
            vec![b"".as_slice(), b"last"],
            "record ordering changed"
        );
    }

    #[test]
    fn snapshot_excludes_overwrite_and_an_interrupted_writer_never_spins() {
        let ring = InputRing::new(1, 4);
        ring.push(b"old").unwrap();
        let snapshot = ring.snapshot().unwrap();
        assert_eq!(ring.push(b"new"), Err(RecordError::Busy));
        assert!(matches!(ring.snapshot(), Err(RecordError::Busy)));
        assert_eq!(snapshot.inputs().next(), Some(b"old".as_slice()));
        drop(snapshot);
        let writer = ring.acquire().unwrap();
        assert!(matches!(ring.snapshot(), Err(RecordError::Busy)));
        drop(writer);
        ring.push(b"new").unwrap();
        assert_eq!(
            ring.snapshot().unwrap().inputs().next(),
            Some(b"new".as_slice())
        );
    }

    #[test]
    fn oversize_is_refused_without_changing_history() {
        let ring = InputRing::new(1, 2);
        ring.push(b"ok").unwrap();
        assert_eq!(
            ring.push(b"long"),
            Err(RecordError::TooLarge {
                actual: 4,
                maximum: 2
            })
        );
        assert_eq!(
            ring.snapshot().unwrap().inputs().next(),
            Some(b"ok".as_slice())
        );
    }

    #[test]
    fn unwind_releases_snapshot_permit() {
        let ring = InputRing::new(1, 2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _snapshot = ring.snapshot().unwrap();
            panic!("planted reader unwind");
        }));
        assert!(result.is_err());
        ring.push(b"ok").unwrap();
    }
}
