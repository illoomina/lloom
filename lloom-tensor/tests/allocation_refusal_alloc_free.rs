//! A refused allocation allocates nothing on the host, and widens into
//! `lloom_tensor::Error::Allocation` unchanged (`memory_admission::AllocationError`).
//!
//! Its own binary because it installs a counting global allocator. The count
//! is per thread: only allocations made on the thread running the refusal are
//! counted, so the test harness's own threads cannot move it.

use lloom_tensor::memory_admission::{process_account, AllocationError, RefusalDomain};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

struct Counting;

static COUNTED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    if COUNTING.with(|c| c.get()) {
        COUNTED.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` unchanged; the count is a side
// effect on an atomic and a const-initialised thread-local, neither of which
// allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `f`'s result and the host allocations made on this thread while it ran.
fn counted<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = COUNTED.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let out = f();
    COUNTING.with(|c| c.set(false));
    (out, COUNTED.load(Ordering::Relaxed) - before)
}

/// The storage boundary's widening, as a caller writes it.
fn widen<T>(r: Result<T, AllocationError>) -> lloom_tensor::Result<T> {
    Ok(r?)
}

#[test]
fn a_refused_reserve_take_and_commit_allocate_nothing_and_widen_typed() {
    let domain = process_account().unadmitted_domain(Some(16)).unwrap();
    let pass = domain.reserve(16).unwrap();
    let context = pass.context();

    let (refused, n) = counted(|| widen(domain.reserve(1).map(drop)));
    assert_eq!(n, 0, "a refused reserve allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 1,
                available: 0,
                domain: RefusalDomain::Model,
            }))
        ),
        "{refused:?}"
    );

    let (refused, n) = counted(|| widen(context.take(17).map(drop)));
    assert_eq!(n, 0, "a refused take allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 17,
                available: 16,
                domain: RefusalDomain::Promise,
            }))
        ),
        "{refused:?}"
    );

    let mut ticket = context.take(8).unwrap();
    let (refused, n) = counted(|| widen(ticket.try_commit(9).map(drop)));
    assert_eq!(n, 0, "a refused commit allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 9,
                available: 8,
                domain: RefusalDomain::Promise,
            }))
        ),
        "{refused:?}"
    );
    drop(ticket);

    drop(pass);
    let (refused, n) = counted(|| widen(context.take(1).map(drop)));
    assert_eq!(n, 0, "a cancelled take allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Cancelled))
        ),
        "{refused:?}"
    );
}

/// The instrument can see an allocation: a positive control, so the zeros
/// above are a measurement and not a counter that never moves.
#[test]
fn the_counter_sees_a_formatted_refusal() {
    let (_, n) = counted(|| lloom_tensor::Error::Msg(format!("{}", AllocationError::Cancelled)));
    assert!(n > 0, "the counting allocator did not observe format!");
}
