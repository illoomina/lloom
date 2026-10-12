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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SetupError {
    InvalidCapacity,
    Overflow,
    Budget { requested: usize, limit: usize },
    HostAllocation,
    SlotSize,
}
impl std::fmt::Display for SetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCapacity => f.write_str("record capacity must be nonzero"),
            Self::Overflow => f.write_str("record ring allocation layout overflow"),
            Self::Budget { requested, limit } => write!(
                f,
                "record ring allocation budget exceeded: requested {requested}, limit {limit}"
            ),
            Self::HostAllocation => f.write_str("record ring host allocation refused"),
            Self::SlotSize => f.write_str("record slot allocator returned wrong size"),
        }
    }
}
impl std::error::Error for SetupError {}

fn zeroed_slot(size: usize) -> Result<Box<[u8]>, SetupError> {
    if size == 0 {
        return Ok(Box::default());
    }
    let layout = std::alloc::Layout::array::<u8>(size).map_err(|_| SetupError::Overflow)?;
    // SAFETY: positive validated layout, Global allocation, fully initialized
    // bytes. A null result is a fallible refusal, never Box::from_raw(null).
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
    if pointer.is_null() {
        return Err(SetupError::HostAllocation);
    }
    // SAFETY: exact requested slice layout and allocator; Box owns this unique
    // allocation and deallocates with that same layout. No shrink/reallocation.
    Ok(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer, size)) })
}

impl InputRing {
    /// Requested payload and two metadata-array layouts only; not allocator
    /// rounding, RSS, concurrent setup, path/identity storage or dump workspace.
    pub(super) fn requested_bytes(capacity: usize, maximum: usize) -> Result<usize, SetupError> {
        if capacity == 0 {
            return Err(SetupError::InvalidCapacity);
        }
        let slots = std::alloc::Layout::array::<Box<[u8]>>(capacity)
            .map_err(|_| SetupError::Overflow)?
            .size();
        let lengths = std::alloc::Layout::array::<usize>(capacity)
            .map_err(|_| SetupError::Overflow)?
            .size();
        let payload = std::alloc::Layout::array::<u8>(maximum)
            .map_err(|_| SetupError::Overflow)?
            .size();
        capacity
            .checked_mul(payload)
            .and_then(|n| n.checked_add(slots))
            .and_then(|n| n.checked_add(lengths))
            .ok_or(SetupError::Overflow)
    }

    pub(super) fn try_new(
        capacity: usize,
        maximum: usize,
        budget: usize,
    ) -> Result<Self, SetupError> {
        Self::try_new_with(capacity, maximum, budget, zeroed_slot)
    }

    fn try_new_with(
        capacity: usize,
        maximum: usize,
        budget: usize,
        mut allocate: impl FnMut(usize) -> Result<Box<[u8]>, SetupError>,
    ) -> Result<Self, SetupError> {
        let requested = Self::requested_bytes(capacity, maximum)?;
        if requested > budget {
            return Err(SetupError::Budget {
                requested,
                limit: budget,
            });
        }
        // Full quote/refusal precedes any metadata or slot allocation. Both
        // Vecs reserve once fallibly; resize/push cannot grow inside this loop.
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| SetupError::HostAllocation)?;
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(capacity)
            .map_err(|_| SetupError::HostAllocation)?;
        lengths.resize(capacity, 0);
        for _ in 0..capacity {
            let slot = allocate(maximum)?;
            if slot.len() != maximum {
                return Err(SetupError::SlotSize);
            }
            slots.push(slot);
        }
        Ok(Self {
            busy: AtomicBool::new(false),
            entries: UnsafeCell::new(Entries {
                slots,
                lengths,
                total: 0,
            }),
            capacity,
            maximum,
        })
    }
}

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
    #[test]
    fn setup_budget_refuses_before_slot_birth_and_explicit_budget_recovers() {
        let needed = InputRing::requested_bytes(2, 4).unwrap();
        let mut births = 0;
        let result = InputRing::try_new_with(2, 4, needed - 1, |_| {
            births += 1;
            zeroed_slot(4)
        });
        assert!(
            matches!(result, Err(SetupError::Budget { requested, limit })
            if requested == needed && limit == needed - 1)
        );
        assert_eq!(births, 0, "quota refusal must precede slot creation");
        let ring = InputRing::try_new(2, 4, needed).unwrap();
        for bytes in [b"old".as_slice(), b"", b"last"] {
            ring.push(bytes).unwrap();
        }
        assert_eq!(
            ring.snapshot().unwrap().inputs().collect::<Vec<_>>(),
            vec![b"".as_slice(), b"last"]
        );
    }

    #[test]
    fn setup_layout_overflow_and_zero_capacity_precede_allocator() {
        for (capacity, maximum, expected) in [
            (usize::MAX, 1, SetupError::Overflow),
            (1, usize::MAX, SetupError::Overflow),
            (0, 1, SetupError::InvalidCapacity),
        ] {
            let mut births = 0;
            let result = InputRing::try_new_with(capacity, maximum, usize::MAX, |_| {
                births += 1;
                zeroed_slot(1)
            });
            assert!(matches!(result, Err(error) if error == expected));
            assert_eq!(births, 0);
        }
    }

    #[test]
    fn setup_partial_slot_refusal_is_fallible_and_retry_preserves_ring() {
        let needed = InputRing::requested_bytes(3, 2).unwrap();
        let mut births = 0;
        let result = InputRing::try_new_with(3, 2, needed, |size| {
            births += 1;
            if births == 2 {
                Err(SetupError::HostAllocation)
            } else {
                zeroed_slot(size)
            }
        });
        assert!(matches!(result, Err(SetupError::HostAllocation)));
        assert_eq!(births, 2);
        let ring = InputRing::try_new(3, 2, needed).unwrap();
        ring.push(b"ok").unwrap();
        assert_eq!(
            ring.snapshot().unwrap().inputs().next(),
            Some(b"ok".as_slice())
        );
    }

    #[test]
    fn setup_zero_length_payload_keeps_empty_input_support() {
        let needed = InputRing::requested_bytes(2, 0).unwrap();
        let ring = InputRing::try_new(2, 0, needed).unwrap();
        ring.push(b"").unwrap();
        assert_eq!(
            ring.snapshot().unwrap().inputs().next(),
            Some(b"".as_slice())
        );
    }
}
