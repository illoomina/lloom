//! A pool operation allocates nothing on the host once the pool is warm
//! (`DESIGN.md` §6.4e, CONTRIBUTING §3.3a "zero allocations per token").
//!
//! Its own binary because it installs a counting global allocator. The count
//! is per thread: only allocations made on the thread running a pool
//! operation are counted, so the test harness's own threads cannot move it.

#![cfg(feature = "metal")]

use lloom_tensor::kernels::metal::{BufferPool, Device, GpuClock};
use lloom_tensor::kernels::RESOURCE_OPTIONS;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

struct Counting;

static COUNTED: AtomicU64 = AtomicU64::new(0);
static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

thread_local! {
    static TRACING: Cell<bool> = const { Cell::new(false) };
}

/// `LLOOM_ALLOC_TRACE=1`: print where the first counted allocations came
/// from (a failure's diagnosis, off by default).
fn trace_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LLOOM_ALLOC_TRACE").is_some())
}

fn note() {
    if !COUNTING.with(|c| c.get()) {
        return;
    }
    let n = COUNTED.fetch_add(1, Ordering::Relaxed);
    if n < 4 && TRACE.load(Ordering::Relaxed) && !TRACING.with(|t| t.replace(true)) {
        COUNTING.with(|c| c.set(false));
        eprintln!(
            "counted allocation:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
        COUNTING.with(|c| c.set(true));
        TRACING.with(|t| t.set(false));
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

/// Host allocations made on this thread while `f` runs.
fn allocations_in(f: impl FnOnce()) -> u64 {
    TRACE.store(trace_wanted(), Ordering::Relaxed);
    let before = COUNTED.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    COUNTED.load(Ordering::Relaxed) - before
}

/// The decode shape: every step asks for the same sizes, releases them, and
/// the GPU completes the step before the next asks again. Warm, the pool
/// serves every request from its free list — and a hit must not allocate
/// on the host either: not a handle per acquire, not a bucket re-created
/// when one empties and refills, not a queue entry per release that nothing
/// ever takes back.
///
/// Absolute reference, with a control that pins the counter's level: the
/// same loop with every request a MISS allocates at least one host object
/// per request (the handle of the buffer it adopts), so a counter that read
/// 0 because it counted nothing fails the control.
#[test]
fn a_warm_pool_hit_allocates_nothing_on_the_host() {
    let dev = Device::system_default().expect("no Metal device");
    let clock = Arc::new(GpuClock::new());
    let pool = BufferPool::with_clock(Arc::clone(&clock));
    // Sizes a decode step asks for, several of each and a size that appears
    // once, so buckets empty and refill every step.
    let sizes: Vec<usize> = [4096, 4096, 4096, 16384, 16384, 65536, 1 << 20, 256]
        .into_iter()
        .collect();
    // The step fills a Vec the test owns, so the test's own bookkeeping
    // allocates nothing inside the counted window.
    let step = |pool: &BufferPool, held: &mut Vec<_>| {
        for &s in &sizes {
            let b = match pool.acquire(s) {
                Some(b) => b,
                None => {
                    let raw = dev.new_buffer(s, RESOURCE_OPTIONS).expect("allocation");
                    pool.adopt(raw, s)
                }
            };
            held.push(b);
        }
    };
    let mut held = Vec::with_capacity(sizes.len());
    // Warm: enough steps that every size has a free buffer at every point,
    // and that the eviction queue has reached its high-water mark (it is
    // compacted once its stale entries outnumber the free buffers, so its
    // capacity settles at a few steps' worth of releases).
    for _ in 0..32 {
        step(&pool, &mut held);
        held.clear();
        let e = clock.commit_epoch();
        clock.mark_completed(e);
    }
    let misses_before = pool.counters().allocations;
    let steps = 2000u64;
    let counted = allocations_in(|| {
        for _ in 0..steps {
            step(&pool, &mut held);
            held.clear();
            let e = clock.commit_epoch();
            clock.mark_completed(e);
        }
    });
    assert_eq!(
        pool.counters().allocations,
        misses_before,
        "the warm loop missed: this test is about hits"
    );
    assert_eq!(
        counted,
        0,
        "{counted} host allocations over {steps} warm steps of {} pool hits each",
        sizes.len()
    );

    // Control: a pool that never reuses (a fresh pool per step) must count
    // host allocations, or the zero above is the counter's.
    let counted_misses = allocations_in(|| {
        let cold = BufferPool::with_clock(Arc::new(GpuClock::new()));
        let raw = dev.new_buffer(4096, RESOURCE_OPTIONS).expect("allocation");
        drop(cold.adopt(raw, 4096));
    });
    assert!(
        counted_misses >= 1,
        "the counter saw nothing on a miss: the zero above is not evidence"
    );
}
