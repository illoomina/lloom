//! The library test binary's one global allocator. It forwards every call to
//! `System` unchanged and keeps the instruments the tests read: a thread-local,
//! opt-in count of requested bytes (`counted`); the allocation calls, live
//! requested bytes and live blocks, both for the calling thread (`here`) and
//! for the process (`process`); and, while `under_lock` watches a mutex on this
//! thread, the allocations and frees made with that mutex held.
//!
//! A `realloc` counts as one allocation call and moves the live bytes by the
//! size change, which is what the trait's default (allocate, copy, free) would
//! have counted.
//!
//! The process-wide counters include every thread's allocations: the test
//! harness's own (libtest's main thread records and waits for the test after
//! the test's thread has started) and every other test's under a runner that
//! shares the process. A window measuring work done on its own thread reads
//! `here`. A block freed on a thread other than the one that allocated it
//! moves both threads' receipts, so `here` measures only work that allocates
//! and frees on the reading thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering};

pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
pub static LIVE: AtomicIsize = AtomicIsize::new(0);
pub static BLOCKS: AtomicIsize = AtomicIsize::new(0);

static WATCHED: AtomicPtr<parking_lot::RawMutex> = AtomicPtr::new(std::ptr::null_mut());
static WATCHED_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static WATCHED_FREES: AtomicUsize = AtomicUsize::new(0);
static LOCKED_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static LOCKED_FREES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static REQUESTED: Cell<Option<usize>> = const { Cell::new(None) };
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static HERE_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static HERE_LIVE: Cell<isize> = const { Cell::new(0) };
    static HERE_BLOCKS: Cell<isize> = const { Cell::new(0) };
}

/// Allocation calls, live requested bytes and live blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Receipt {
    pub allocations: usize,
    pub live: isize,
    pub blocks: isize,
}

/// The calling thread's receipt: its allocations, and its allocations less
/// its frees.
pub(crate) fn here() -> Receipt {
    Receipt {
        allocations: HERE_ALLOCATIONS.with(Cell::get),
        live: HERE_LIVE.with(Cell::get),
        blocks: HERE_BLOCKS.with(Cell::get),
    }
}

/// Every thread's receipt.
pub(crate) fn process() -> Receipt {
    Receipt {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        live: LIVE.load(Ordering::Relaxed),
        blocks: BLOCKS.load(Ordering::Relaxed),
    }
}

/// Moves this thread's receipt; a thread whose locals are gone moves none.
fn here_add(allocations: usize, live: isize, blocks: isize) {
    let _ = HERE_ALLOCATIONS.try_with(|c| c.set(c.get() + allocations));
    let _ = HERE_LIVE.try_with(|c| c.set(c.get() + live));
    let _ = HERE_BLOCKS.try_with(|c| c.set(c.get() + blocks));
}

/// `Some(held)` while `under_lock` watches on this thread.
fn watched() -> Option<bool> {
    if !WATCHING.try_with(Cell::get).unwrap_or(false) {
        return None;
    }
    let raw = WATCHED.load(Ordering::Acquire);
    if raw.is_null() {
        return None;
    }
    // SAFETY: `under_lock` keeps the mutex borrowed while the pointer is set;
    // `is_locked` is one relaxed load of its state word.
    Some(unsafe { parking_lot::lock_api::RawMutex::is_locked(&*raw) })
}

fn watch(all: &AtomicUsize, locked: &AtomicUsize) {
    if let Some(held) = watched() {
        all.fetch_add(1, Ordering::Relaxed);
        if held {
            locked.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn record(bytes: usize) {
    let _ = REQUESTED.try_with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous.saturating_add(bytes)));
        }
    });
}

fn born(bytes: usize) {
    watch(&WATCHED_ALLOCATIONS, &LOCKED_ALLOCATIONS);
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    BLOCKS.fetch_add(1, Ordering::Relaxed);
    LIVE.fetch_add(bytes as isize, Ordering::Relaxed);
    here_add(1, bytes as isize, 1);
}

struct Counter;

#[global_allocator]
static ALLOCATOR: Counter = Counter;

// SAFETY: every call is forwarded unchanged to System; the instruments are
// atomics and const-initialized thread-local scalars, none of which
// allocates or dereferences an allocator pointer.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller's layout is forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            born(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller's layout is forwarded unchanged.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            born(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        // SAFETY: the caller's live allocation, its layout and the new size
        // are forwarded unchanged.
        let moved = unsafe { System.realloc(ptr, layout, size) };
        if !moved.is_null() {
            watch(&WATCHED_ALLOCATIONS, &LOCKED_ALLOCATIONS);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(size as isize - layout.size() as isize, Ordering::Relaxed);
            here_add(1, size as isize - layout.size() as isize, 0);
        }
        moved
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        watch(&WATCHED_FREES, &LOCKED_FREES);
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        BLOCKS.fetch_sub(1, Ordering::Relaxed);
        here_add(0, -(layout.size() as isize), -1);
        // SAFETY: the caller's allocation and its layout are forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Run `run` and return the bytes it requested on this thread.
pub(crate) fn counted<T>(run: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REQUESTED.with(|count| count.set(None));
        }
    }
    REQUESTED.with(|count| {
        assert!(count.get().is_none());
        count.set(Some(0));
    });
    let _reset = Reset;
    let result = run();
    let bytes = REQUESTED.with(|count| count.get().unwrap());
    (result, bytes)
}

/// This thread's allocations and frees inside an `under_lock` call, and those
/// of them made while the watched mutex was held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Watched {
    pub allocations: usize,
    pub frees: usize,
    pub locked_allocations: usize,
    pub locked_frees: usize,
}

/// Run `run`, counting this thread's allocations and frees, and those made
/// while `lock` is held.
pub(crate) fn under_lock<T, R>(
    lock: &parking_lot::Mutex<T>,
    run: impl FnOnce() -> R,
) -> (R, Watched) {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            WATCHING.with(|w| w.set(false));
            WATCHED.store(std::ptr::null_mut(), Ordering::Release);
        }
    }
    for counter in [
        &WATCHED_ALLOCATIONS,
        &WATCHED_FREES,
        &LOCKED_ALLOCATIONS,
        &LOCKED_FREES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    // SAFETY: the raw mutex is only read (`is_locked`), never locked or
    // unlocked through this pointer, and only while `lock` is borrowed here.
    let raw = unsafe { lock.raw() } as *const parking_lot::RawMutex as *mut parking_lot::RawMutex;
    WATCHED.store(raw, Ordering::Release);
    WATCHING.with(|w| w.set(true));
    let stop = Stop;
    let result = run();
    drop(stop);
    let seen = Watched {
        allocations: WATCHED_ALLOCATIONS.load(Ordering::Relaxed),
        frees: WATCHED_FREES.load(Ordering::Relaxed),
        locked_allocations: LOCKED_ALLOCATIONS.load(Ordering::Relaxed),
        locked_frees: LOCKED_FREES.load(Ordering::Relaxed),
    };
    (result, seen)
}
