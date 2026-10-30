//! The library test binary's one global allocator. It forwards every call to
//! `System` unchanged and keeps the instruments the tests read: a thread-local,
//! opt-in count of requested bytes (`counted`), and the process-wide
//! allocation calls, live requested bytes and live blocks.
//!
//! A `realloc` counts as one allocation call and moves the live bytes by the
//! size change, which is what the trait's default (allocate, copy, free) would
//! have counted. The process-wide counters are shared by every test in the
//! process: a test reading them needs the runner's per-process isolation
//! (nextest) or `--test-threads=1`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
pub static LIVE: AtomicIsize = AtomicIsize::new(0);
pub static BLOCKS: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    static REQUESTED: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record(bytes: usize) {
    let _ = REQUESTED.try_with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous.saturating_add(bytes)));
        }
    });
}

fn born(bytes: usize) {
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
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        moved
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
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
