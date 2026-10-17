//! SOURCE PROPOSAL ONLY. Included only by the enabled diagnostic module.
//! No Tensor, native resource, user callback, global registry or GPU wait.
use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

pub const MAX_OBSERVERS: usize = 64;
const OPEN: u8 = 0;
const EOS: u8 = 1;
const CLOSED: u8 = 2;
const ERROR: u8 = 3;
const UNWIND: u8 = 4;

#[derive(Debug, Default)]
struct Observation {
    watch: AtomicBool,
    first_eos: AtomicU64,
    terminal: AtomicU8,
}

/// One reader's exact observation record; the group retains another strong
/// owner, so closing the reader never discards a recorded first sighting.
#[derive(Debug)]
pub struct ReaderObserver(Arc<Observation>);
impl ReaderObserver {
    pub fn mark_watch(&self) {
        self.0.watch.store(true, Ordering::Relaxed);
    }
    pub fn saw_eos(&self, now: f64) {
        // Error/close is terminal for this capture, even if a caller retries
        // the public RingReader afterwards. Never erase a prior read failure.
        if self.0.terminal.load(Ordering::Acquire) != OPEN {
            return;
        }
        let _ = self.0.first_eos.compare_exchange(
            0,
            now.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        let _ = self
            .0
            .terminal
            .compare_exchange(OPEN, EOS, Ordering::Release, Ordering::Relaxed);
    }
    pub fn read_error(&self) {
        let _ = self
            .0
            .terminal
            .compare_exchange(OPEN, ERROR, Ordering::Release, Ordering::Relaxed);
    }
}
impl Drop for ReaderObserver {
    fn drop(&mut self) {
        let terminal = if std::thread::panicking() {
            UNWIND
        } else {
            CLOSED
        };
        let _ =
            self.0
                .terminal
                .compare_exchange(OPEN, terminal, Ordering::Release, Ordering::Relaxed);
    }
}

#[derive(Debug, Default)]
struct GroupState {
    started: bool,
    sealed: bool,
    overflow: bool,
    readers: Vec<Arc<Observation>>,
}
#[derive(Debug, Default)]
pub struct ObserverGroup(Mutex<GroupState>);

#[derive(Clone, Copy, Debug, Default)]
pub struct ObservationSnapshot {
    pub registered: usize,
    pub open: usize,
    pub closed: usize,
    pub errors: usize,
    pub unwinds: usize,
    pub overflow: bool,
    pub sealed: bool,
    pub reader_eos: Option<f64>,
    pub watch_eos: Option<f64>,
}
impl ObservationSnapshot {
    pub fn terminal(&self) -> bool {
        self.sealed && self.open == 0
    }
    pub fn incomplete(&self) -> bool {
        self.overflow || self.errors != 0 || self.unwinds != 0
    }
}
impl ObserverGroup {
    /// This lock is cold setup only. Either registration enters before seal,
    /// or returns Err and the Ring slot chooses a fresh group; never append to
    /// a sealed run. Overflow is an explicit capture failure, not reader failure.
    fn register(&self) -> std::result::Result<Option<ReaderObserver>, ()> {
        let mut state = self.0.lock();
        if state.sealed {
            return Err(());
        }
        if state.readers.len() == MAX_OBSERVERS {
            state.overflow = true;
            return Ok(None);
        }
        let observation = Arc::new(Observation::default());
        state.readers.push(Arc::clone(&observation));
        Ok(Some(ReaderObserver(observation)))
    }
    fn start(&self) -> bool {
        let mut state = self.0.lock();
        if state.started || state.sealed {
            return false;
        }
        state.started = true;
        true
    }
    pub fn seal(&self) {
        self.0.lock().sealed = true;
    }
    pub fn snapshot(&self) -> ObservationSnapshot {
        let state = self.0.lock();
        let mut out = ObservationSnapshot {
            registered: state.readers.len(),
            sealed: state.sealed,
            overflow: state.overflow,
            ..ObservationSnapshot::default()
        };
        for record in &state.readers {
            match record.terminal.load(Ordering::Acquire) {
                OPEN => out.open += 1,
                EOS => {
                    let time = f64::from_bits(record.first_eos.load(Ordering::Relaxed));
                    let slot = if record.watch.load(Ordering::Relaxed) {
                        &mut out.watch_eos
                    } else {
                        &mut out.reader_eos
                    };
                    *slot = Some(slot.map_or(time, |old| old.min(time)));
                }
                CLOSED => out.closed += 1,
                ERROR => out.errors += 1,
                UNWIND => out.unwinds += 1,
                _ => unreachable!("private observation terminal state"),
            }
        }
        out
    }
}

/// Owned by the Ring, used only on its cold setup thread. Workers own scalar
/// observation records, NOT this RefCell. Prepared readers and a later begin
/// share one exact Arc group; old readers never acquire a newer group.
#[derive(Debug, Default)]
pub struct ObserverSlot(RefCell<Option<Arc<ObserverGroup>>>);
impl ObserverSlot {
    pub fn reader(&self, enabled: bool) -> Option<ReaderObserver> {
        if !enabled {
            return None;
        } // no borrow, lock, Arc or Vec on disabled path
        let mut slot = self.0.borrow_mut();
        loop {
            let group = slot.get_or_insert_with(|| Arc::new(ObserverGroup::default()));
            match group.register() {
                Ok(observer) => return observer,
                Err(()) => *slot = None,
            }
        }
    }
    pub fn begin(&self, enabled: bool) {
        if !enabled {
            return;
        }
        let mut slot = self.0.borrow_mut();
        let group = slot.get_or_insert_with(|| Arc::new(ObserverGroup::default()));
        if !group.start() {
            // A previous run sealed its group, or a direct begin superseded
            // an unbound start. No reader/observer is rebound to the new group.
            let next = Arc::new(ObserverGroup::default());
            assert!(next.start());
            *slot = Some(next);
        }
    }
    pub fn group(&self, enabled: bool) -> Option<Arc<ObserverGroup>> {
        if !enabled {
            return None;
        }
        self.0.borrow().as_ref().map(Arc::clone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_reader_records_late_eos_only_in_its_original_run() {
        let slot = ObserverSlot::default();
        let reader = slot.reader(true).unwrap(); // BEFORE decode starts
        slot.begin(true);
        let old = slot.group(true).unwrap();
        old.seal(); // producer return does NOT close a live observer
        assert!(!old.snapshot().terminal());
        let new_reader = slot.reader(true).unwrap();
        slot.begin(true);
        let new = slot.group(true).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        reader.saw_eos(3.0);
        assert!(old.snapshot().terminal());
        assert_eq!(old.snapshot().reader_eos, Some(3.0));
        assert_eq!(new.snapshot().reader_eos, None);
        drop(reader);
        assert_eq!(new.snapshot().open, 1);
        new.seal();
        drop(new_reader);
        assert!(new.snapshot().terminal());
        assert_eq!(new.snapshot().closed, 1);
    }

    #[test]
    fn zero_reader_and_error_unwind_closures_are_distinct() {
        let zero = ObserverGroup::default();
        assert!(zero.start());
        zero.seal();
        assert!(zero.snapshot().terminal());
        assert_eq!(zero.snapshot().reader_eos, None);
        let group = ObserverGroup::default();
        let error = group.register().unwrap().unwrap();
        let unwind = group.register().unwrap().unwrap();
        error.read_error();
        drop(error);
        let result = std::panic::catch_unwind(|| {
            let _unwind = unwind;
            panic!("controlled reader unwind");
        });
        assert!(result.is_err());
        group.seal();
        let snap = group.snapshot();
        assert!(snap.terminal() && snap.incomplete());
        assert_eq!((snap.errors, snap.unwinds, snap.closed), (1, 1, 0));
        assert_eq!(snap.reader_eos, None);
    }

    #[test]
    fn seal_and_registration_have_one_atomic_disposition() {
        let group = Arc::new(ObserverGroup::default());
        let gate = Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let group = Arc::clone(&group);
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                gate.wait();
                group.register()
            })
        };
        gate.wait();
        group.seal();
        let outcome = worker.join().unwrap();
        match outcome {
            Ok(Some(reader)) => {
                assert_eq!(group.snapshot().registered, 1);
                drop(reader);
            }
            Err(()) => assert_eq!(group.snapshot().registered, 0),
            Ok(None) => panic!("empty group cannot overflow"),
        }
        assert!(group.snapshot().terminal());
        assert!(group.register().is_err());
    }

    #[test]
    fn actual_eos_publication_takes_no_group_registration_lock() {
        let group = Arc::new(ObserverGroup::default());
        let reader = group.register().unwrap().unwrap();
        let held = group.0.lock();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            reader.saw_eos(7.0);
            send.send(()).unwrap();
        });
        let completed = receive.recv_timeout(std::time::Duration::from_secs(1));
        drop(held); // ALWAYS release before join, including the failing mutant path
        worker.join().unwrap();
        assert!(
            completed.is_ok(),
            "observer event tried to take registration lock"
        );
        group.seal();
        assert_eq!(group.snapshot().reader_eos, Some(7.0));
        assert!(group.snapshot().terminal());
    }

    #[test]
    fn overflow_is_bounded_and_disabled_slot_is_untouched() {
        let slot = ObserverSlot::default();
        assert!(slot.reader(false).is_none());
        slot.begin(false);
        assert!(slot.group(false).is_none());
        assert!(slot.0.borrow().is_none());
        let group = ObserverGroup::default();
        let mut readers = Vec::new();
        for _ in 0..MAX_OBSERVERS {
            readers.push(group.register().unwrap().unwrap());
        }
        assert!(group.register().unwrap().is_none());
        assert_eq!(group.snapshot().registered, MAX_OBSERVERS);
        assert!(group.snapshot().overflow);
        group.seal();
        drop(readers);
        assert!(group.snapshot().terminal() && group.snapshot().incomplete());
    }
}
