//! The library test binary's one global allocator. It forwards every call to
//! `System` unchanged and keeps the instruments the tests read: a thread-local,
//! opt-in count of requested bytes (`counted`), and the process-wide
//! allocation calls, live requested bytes and live blocks; and, while
//! `under_lock` watches a mutex on this thread, the allocations and frees made
//! with that mutex held.
//!
//! A `realloc` counts as one allocation call and moves the live bytes by the
//! size change, which is what the trait's default (allocate, copy, free) would
//! have counted. The process-wide counters are shared by every test in the
//! process: a test reading them needs the runner's per-process isolation
//! (nextest) or `--test-threads=1`.

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
}

struct Counter;

#[global_allocator]
static ALLOCATOR: Counter = Counter;

// SAFETY: every call is forwarded unchanged to System; the instruments are
// atomics and a const-initialized thread-local scalar, none of which
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
        }
        moved
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        watch(&WATCHED_FREES, &LOCKED_FREES);
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        BLOCKS.fetch_sub(1, Ordering::Relaxed);
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
